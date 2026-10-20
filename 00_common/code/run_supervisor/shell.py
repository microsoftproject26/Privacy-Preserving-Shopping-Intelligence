"""Shell-pipeline wrapping (package 10 S4: "Don't mask subprocess failure by piping to tail
without pipefail."; "runs the subprocess with pipefail semantics and captures the exit code").

A registry phase normally gives `command` as a plain argv list, executed directly with no shell at
all -- the safest option, and immune to this problem by construction, since there is no pipe to
mask a failure in. `build_shell_argv` exists only for the rarer `shell_command` phase, whose
command string may itself contain a shell pipeline (e.g. `some_tool | tee out.log`): on POSIX it is
run as `bash -o pipefail -c CMD`, so the pipeline's exit code reflects the first stage that failed,
not just its last stage.
"""
from __future__ import annotations

import os


def build_shell_argv(shell_command: str) -> "list[str]":
    if os.name == "posix":
        shell = os.environ.get("SHELL") or "/bin/bash"
        return [shell, "-o", "pipefail", "-c", shell_command]
    # This project's production target is the Linux server (package 10); a shell_command phase is
    # not expected to run on Windows. PowerShell has no pipefail equivalent, so this is a
    # best-effort fallback for local development only, not a guarantee.
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command",
            f"$ErrorActionPreference = 'Stop'; {shell_command}"]
