#!/usr/bin/env python3
"""Focused Unicode cleanup regression checks.

Run directly from the module root:
  python tools/tests/test_text_cleanup.py
"""

import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from chatter_text import cleanup_message  # noqa: E402


def test_hangul_and_cjk_text_are_preserved():
    message = "좋아, 이제 출발하자. 准备好了吗?"
    assert cleanup_message(message) == message


def test_actual_emoji_ranges_are_removed():
    message = "A\U000024C2B\U0001F201C\U0001F600D"
    assert cleanup_message(message) == "ABCD"


def main() -> int:
    test_hangul_and_cjk_text_are_preserved()
    test_actual_emoji_ranges_are_removed()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
