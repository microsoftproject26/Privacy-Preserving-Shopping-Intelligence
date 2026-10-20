"""build_shell_argv: a shell_command phase runs with pipefail semantics on POSIX, so a failure
partway through a pipeline is not masked by the exit code of its last stage (package 10 S4:
"Don't mask subprocess failure by piping to tail without pipefail."). Pure function, no subprocess
is actually spawned here -- `os.name` is monkeypatched so this is testable on any platform.
"""
from __future__ import annotations

from v4sup.shell import build_shell_argv


def test_posix_uses_bash_pipefail(monkeypatch):
    monkeypatch.setattr("v4sup.shell.os.name", "posix")
    monkeypatch.delenv("SHELL", raising=False)
    argv = build_shell_argv("false | tee out.log")
    assert argv == ["/bin/bash", "-o", "pipefail", "-c", "false | tee out.log"]


def test_posix_honors_shell_env(monkeypatch):
    monkeypatch.setattr("v4sup.shell.os.name", "posix")
    monkeypatch.setenv("SHELL", "/bin/zsh")
    argv = build_shell_argv("echo hi")
    assert argv[0] == "/bin/zsh"
    assert argv[1:3] == ["-o", "pipefail"]


def test_non_posix_fallback_does_not_crash(monkeypatch):
    monkeypatch.setattr("v4sup.shell.os.name", "nt")
    argv = build_shell_argv("echo hi")
    assert "echo hi" in argv[-1]
