#!/usr/bin/env python
"""
Minimal shim of sqlmap's lib/core/compat.py -- only the Python 2/3
`xrange` compatibility alias, which is all space2comment.py and
randomcomments.py require.
Original source: https://github.com/sqlmapproject/sqlmap/blob/master/lib/core/compat.py
Copyright (c) 2006-2026 sqlmap developers (https://sqlmap.org)
See the file 'LICENSE' for copying permission
"""

xrange = range
