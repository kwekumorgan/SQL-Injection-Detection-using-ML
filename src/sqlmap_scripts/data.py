#!/usr/bin/env python
"""
Minimal shim of sqlmap's lib/core/data.py -- only the `kb` (knowledge
base) object's `.keywords` attribute, which randomcomments.py checks
against. In real sqlmap this is populated at startup via:
    kb.keywords = set(getFileItems(paths.SQL_KEYWORDS))
(see lib/core/option.py). We reproduce that exact logic here, reading
the same keywords.txt file vendored alongside this module.
Copyright (c) 2006-2026 sqlmap developers (https://sqlmap.org)
See the file 'LICENSE' for copying permission
"""

import os


def _load_keywords():
    keywords_path = os.path.join(os.path.dirname(__file__), "keywords.txt")
    keywords = set()
    with open(keywords_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                keywords.add(line.upper())
    return keywords


class _KnowledgeBase(object):
    def __init__(self):
        self.keywords = _load_keywords()


kb = _KnowledgeBase()
