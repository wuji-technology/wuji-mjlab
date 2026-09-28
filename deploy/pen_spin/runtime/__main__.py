# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""``python -m runtime`` — process-level concerns, kept out of main().

Two things belong to "being a process" rather than to "running a deployment",
and having them here is what keeps ``main()`` callable from a test:

  * os._exit instead of a normal return: with the hand SDK, zenoh and a GL
    renderer in the process, a non-daemon thread of theirs holds the interpreter
    open and leaves the operator's terminal unusable. Session.close() has already
    released everything we own, and logged it.
  * the last-chance stdout flush that os._exit skips
"""

from __future__ import annotations

import os
import sys
import traceback

from runtime.main import main

if __name__ == "__main__":
    try:
        _rc = main()
    except SystemExit as exc:
        _rc = exc.code
        if _rc is not None and not isinstance(_rc, int):
            print(_rc, file=sys.stderr)
            _rc = 1
    except KeyboardInterrupt:
        _rc = 130
    except BaseException:
        traceback.print_exc()
        _rc = 1
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0 if _rc is None else _rc)
