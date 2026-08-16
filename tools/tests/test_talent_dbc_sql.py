#!/usr/bin/env python3
"""Regression checks for the TalentTab SQL overrides.

Run directly from the module root:
  python tools/tests/test_talent_dbc_sql.py
"""

import re
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[2]
BASE_SQL = (
    MODULE_DIR
    / "data"
    / "sql"
    / "world"
    / "base"
    / "llm_chatter_talent_dbc.sql"
)
UPDATE_SQL = (
    MODULE_DIR
    / "data"
    / "sql"
    / "world"
    / "updates"
    / "20260816_fix_pet_talent_masks.sql"
)

PET_TABS = {
    409: ("Tenacity", 0, 0, 2, 0),
    410: ("Ferocity", 0, 0, 1, 0),
    411: ("Cunning", 0, 0, 4, 0),
}


def _base_pet_tabs() -> dict[int, tuple[str, int, int, int, int]]:
    pattern = re.compile(
        r"VALUES \((409|410|411), '([^']+)', "
        r"(\d+), (\d+), (\d+), (\d+)\)"
    )
    rows = {}
    for match in pattern.finditer(BASE_SQL.read_text(encoding="utf-8")):
        rows[int(match.group(1))] = (
            match.group(2),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
        )
    return rows


def test_base_pet_tabs_match_client_dbc():
    assert _base_pet_tabs() == PET_TABS


def test_existing_install_update_repairs_all_pet_tabs():
    sql = UPDATE_SQL.read_text(encoding="utf-8")
    for tab_id, (_, _, order, pet_mask, _) in PET_TABS.items():
        block = re.search(
            rf"UPDATE `talenttab_dbc`\s+SET ([^;]+?)"
            rf"\s+WHERE `ID` = {tab_id};",
            sql,
            flags=re.DOTALL,
        )
        assert block is not None
        assert f"`PetTalentMask` = {pet_mask}" in block.group(0)
        assert f"`OrderIndex` = {order}" in block.group(0)


def main() -> int:
    test_base_pet_tabs_match_client_dbc()
    test_existing_install_update_repairs_all_pet_tabs()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
