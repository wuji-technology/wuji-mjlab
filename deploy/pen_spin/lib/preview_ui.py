# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Qt preview font configuration for the OpenCV wheel."""
import os
from pathlib import Path


def configure_qt_fonts():
    """Keep valid user settings and fall back to an installed system font."""
    configured = os.environ.get('QT_QPA_FONTDIR')
    if configured and Path(configured).is_dir():
        return configured
    for directory in ('/usr/share/fonts/truetype/dejavu',
                      '/usr/share/fonts/truetype/freefont'):
        if Path(directory).is_dir():
            os.environ['QT_QPA_FONTDIR'] = directory
            return directory
    return None
