"""Restrict writes to explicit work/cache directories before executing a command.

Used inside the existing copied-project launcher. Read access is unaffected;
legacy tests cannot overwrite frozen receipts through old absolute paths.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
from pathlib import Path


def restrict(allowed):
    if platform.machine() != 'x86_64':
        raise RuntimeError('This verified server uses x86_64 Landlock syscalls')
    libc = ctypes.CDLL(None, use_errno=True)

    def check(result, operation):
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, operation + ': ' + os.strerror(code))
        return result

    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]

    if check(libc.syscall(444, 0, 0, 1), 'Landlock ABI query') < 3:
        raise RuntimeError('Landlock ABI >= 3 required')
    writes = (1 << 1) | sum(1 << b for b in range(4, 15))
    attr = Ruleset(writes)
    ruleset = check(libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0), 'create ruleset')
    try:
        rules = [(Path(p), writes) for p in allowed] + [(Path(p), (1 << 1) | (1 << 14)) for p in ['/dev', '/proc']]
        for path, access in rules:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, fd)
                check(libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0), 'add write rule')
            finally:
                os.close(fd)
        check(libc.prctl(38, 1, 0, 0, 0), 'set no_new_privs')
        check(libc.syscall(446, ruleset, 0), 'restrict writes')
    finally:
        os.close(ruleset)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allow', type=Path, action='append', required=True)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        ap.error('a command is required')
    restrict(args.allow)
    os.execvpe(command[0], command, os.environ.copy())


if __name__ == '__main__':
    main()
