# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""One timestamped file per run, alongside the operator's console output.

stdout is for reading live and stays exactly as it is — a preflight report is
meant to be seen, not parsed. What stdout cannot do is survive the run: its
lines carry no time, so afterwards "the hand twitched about a second before it
stopped" cannot be placed against the watchdog's own record. So the control
loop's event sites go here as well, and nothing else does.

Deliberately NOT used by ``runtime/safety.py``: that code also runs on the Ctrl+C
path, where the hand must open before teardown, and it must not become able to
fail because a directory is not writable.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

LOGDIR = Path(os.environ.get("WUJI_RUNLOG_DIR") or "/tmp/wuji_pen_spin_runs")


def setup() -> Path | None:
    """Attach the run log; returns its path for the start-up banner.

    Returns None if the file cannot be opened. A missing log is worth a line in
    the banner, never a refused run.
    """
    log = logging.getLogger("run")
    if log.handlers:                          # a second call is a no-op
        return getattr(log.handlers[0], "baseFilename", None)
    try:
        LOGDIR.mkdir(parents=True, exist_ok=True)
        path = LOGDIR / f"{time.strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(path)
    except OSError:
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s",
                                           datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    return path
