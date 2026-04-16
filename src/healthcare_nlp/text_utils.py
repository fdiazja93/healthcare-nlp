from __future__ import annotations 

import re

_ws = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    """Lowercase text and collapse repeated whitespace."""
    s = str(s).strip().lower()
    return _ws.sub(" ", s)
