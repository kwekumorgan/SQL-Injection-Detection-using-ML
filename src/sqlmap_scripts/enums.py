#!/usr/bin/env python
"""
Minimal shim of sqlmap's lib/core/enums.py -- only the PRIORITY class,
which is all space2comment.py and randomcomments.py require.
Original source: https://github.com/sqlmapproject/sqlmap/blob/master/lib/core/enums.py
Copyright (c) 2006-2026 sqlmap developers (https://sqlmap.org)
See the file 'LICENSE' for copying permission
"""


class PRIORITY(object):
    LOWEST = -100
    LOWER = -50
    LOW = -10
    NORMAL = 0
    HIGH = 10
    HIGHER = 50
    HIGHEST = 100
