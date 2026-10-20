"""Static file server that sends COOP/COEP (cross-origin isolation) so onnxruntime-web multi-threaded wasm and
performance.measureUserAgentSpecificMemory() work. Stdlib only. Serves --root (default: this directory); read-only
(GET/HEAD, plus an OPT-IN loopback-only POST sink with --results-dir); binds 127.0.0.1. LAN use is not supported for
measurement (crypto.subtle and cross-origin isolation need a secure context).

    python serve_coi.py --root <dir with device_bench.html, models/, queries.json> --port 8765

COEP require-corp note: the pinned ort.min.js comes from cdn.jsdelivr.net with crossorigin="anonymous" (CORS), which
satisfies COEP. Offline, use a same-origin vendor/ copy of the pinned onnxruntime-web dist files.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

COI_HEADERS = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}


class CoiHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map: ClassVar[dict] = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript",
                      ".onnx": "application/octet-stream", ".json": "application/json"}

    def end_headers(self):
        for k, v in COI_HEADERS.items():
            self.send_header(k, v)
        super().end_headers()

    results_dir = None                    # set via make_server(); None = sink disabled
    SINK_PATH = "/__bench_result"
    MAX_BODY = 1_000_000
    TAG_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")

    def _read_only(self):
        self.send_error(405, "read-only server")

    do_PUT = do_DELETE = do_PATCH = _read_only

    def do_POST(self):
        """Loopback-only result sink: ONE path, loopback client, same-origin Origin, size-capped JSON with the expected
        schema, tag-validated file name inside results_dir. Everything else is 405. A body within the size cap is
        read before any check, so a refused request still gets its status instead of a connection reset."""
        try:
            n = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            n = -1
        body = self.rfile.read(n) if 0 <= n <= self.MAX_BODY else None
        u = urlparse(self.path)
        if u.path != self.SINK_PATH or self.results_dir is None:
            return self._read_only()
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            return self.send_error(403, "loopback only")
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + self.headers.get("Host", ""):
            return self.send_error(403, "same-origin only")
        tag = (parse_qs(u.query).get("tag") or [""])[0]
        if not self.TAG_RE.match(tag):
            return self.send_error(400, "bad tag")
        if body is None:
            return self.send_error(413, "body size")
        try:
            doc = json.loads(body)
        except ValueError:
            return self.send_error(400, "not json")
        if not isinstance(doc, dict) or doc.get("schema") != "ppsi.ondevice.device_bench.v1":
            return self.send_error(400, "bad schema")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / (tag + ".json")).write_text(json.dumps(doc, indent=1), encoding="utf-8")
        self.send_response(204)
        self.end_headers()


def make_server(root: Path, host: str = "127.0.0.1", port: int = 8765,
                results_dir=None) -> http.server.ThreadingHTTPServer:
    cls = type("BoundCoiHandler", (CoiHandler,), {"results_dir": Path(results_dir) if results_dir else None})
    handler = functools.partial(cls, directory=str(root))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    return http.server.ThreadingHTTPServer((host, port), handler)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--results-dir", default=None, help="enable the loopback-only POST sink writing <tag>.json here")
    a = ap.parse_args(argv)
    srv = make_server(Path(a.root), a.host, a.port, a.results_dir)
    print(f"serving {a.root} on http://{a.host}:{a.port}/ (COOP/COEP on); Ctrl+C to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
