#!/usr/bin/env python
"""
Minimal shim of sqlmap's lib/core/common.py -- only the randomRange()
helper, which is all randomcomments.py requires. Logic is unchanged
from the original (a straightforward random.randint wrapper); the
thread-local-seed branch used internally by sqlmap's HTTP layer is
omitted since it is not exercised by randomcomments.py.
Original source: https://github.com/sqlmapproject/sqlmap/blob/master/lib/core/common.py
Copyright (c) 2006-2026 sqlmap developers (https://sqlmap.org)
See the file 'LICENSE' for copying permission
"""

import random


def randomRange(start=0, stop=1000, seed=None):
    """
    Returns random integer value in given range
    """
    if seed is not None:
        rnd = random.Random(seed)
        randint = rnd.randint
    else:
        randint = random.randint

    return int(randint(start, stop))
