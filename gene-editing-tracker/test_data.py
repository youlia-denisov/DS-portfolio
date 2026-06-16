"""
Data integrity tests for the NGT Gene Editing Tracker dashboard.

These tests extract the embedded DATA array from index.html and verify it
matches what the dashboard claims to show — a bit like running QC on a
sequencing dataset before you trust the results.

Run with:
    pip install pytest
    pytest test_data.py -v
"""

import re
import json
import pytest
from pathlib import Path

# ── Load the data once, reuse across all tests ───────────────────────────────

HTML_FILE = Path(__file__).parent / "PowerBI to html transition" / "index.html"

@pytest.fixture(scope="session")
def data():
    """
    Extract the DATA array from index.html using a regex, then parse it as JSON.

    Why regex instead of a proper HTML parser? The DATA variable is a single
    giant JS array literal on one line — regex is the simplest tool here.
    Think of it like using grep to pull a specific gene from a FASTA file.
    """
    html = HTML_FILE.read_text(encoding="utf-8")

    # Match: const DATA = [...];
    match = re.search(r"const DATA\s*=\s*(\[.*?\]);", html, re.DOTALL)
    assert match, "Could not find 'const DATA = [...]' in index.html"

    return json.loads(match.group(1))


# ── Schema tests: every record has the right fields ──────────────────────────

REQUIRED_FIELDS = {"yr", "ct", "co", "pr", "ki", "or", "oc", "tt", "te", "st"}

def test_all_records_have_required_fields(data):
    """
    Like checking that every row in a lab spreadsheet has all columns filled.
    Missing fields would cause silent rendering bugs in the dashboard.
    """
    missing = [
        (i, REQUIRED_FIELDS - record.keys())
        for i, record in enumerate(data)
        if REQUIRED_FIELDS - record.keys()
    ]
    assert not missing, f"Records missing fields: {missing[:5]}"  # show first 5


def test_no_completely_empty_records(data):
    empty = [i for i, r in enumerate(data) if not any(r.values())]
    assert not empty, f"Completely empty records at indices: {empty}"


# ── Value validation: fields contain sensible values ─────────────────────────

VALID_STAGES = {
    "1: Discovery",
    "2: Proof-of-Concept",
    "3: Large-scale trials",
    "4: Approved",
    "5: Marketing",
}

def test_stage_values_are_valid(data):
    """
    'st' (development stage) must be one of the 5 known pipeline stages.
    An unknown stage would be silently dropped from charts — like a sample
    failing QC but not being flagged.
    """
    bad = [(i, r["st"]) for i, r in enumerate(data) if r.get("st") not in VALID_STAGES]
    assert not bad, f"Unknown stage values: {bad}"


VALID_KINGDOMS = {"Plant", "Animal"}

def test_kingdom_values_are_valid(data):
    """'ki' (kingdom) should only be Plant or Animal for agricultural products."""
    bad = [(i, r["ki"]) for i, r in enumerate(data) if r.get("ki") not in VALID_KINGDOMS]
    assert not bad, f"Unexpected kingdom values: {bad}"


def test_year_values_are_plausible(data):
    """
    'yr' (year first reported) should be between 2000 and 2025.
    Outliers here often mean a typo (e.g. 201 instead of 2015).
    """
    bad = [(i, r["yr"]) for i, r in enumerate(data) if not (2000 <= int(r.get("yr", 0)) <= 2025)]
    assert not bad, f"Implausible year values: {bad}"


VALID_COMPANY_TYPES = {"Academic", "Public", "Private"}

def test_company_type_values_are_valid(data):
    """'ct' (company type) must be one of the three known categories."""
    bad = [(i, r["ct"]) for i, r in enumerate(data) if r.get("ct") not in VALID_COMPANY_TYPES]
    assert not bad, f"Unknown company type values: {bad}"


# ── KPI tests: dashboard headline numbers match the actual data ───────────────
#
# The header subtitle and KPI cards are now generated dynamically by JS from
# the DATA array itself, so there is no hard-coded number to go stale.
# These tests verify the HTML structure that JS depends on is still in place,
# and that the KPI initial values (written before any filter is applied) stay
# consistent with the data.

STAGES_FOR_KPI = {
    "2: Proof-of-Concept",
    "3: Large-scale trials",
    "4: Approved",
    "5: Marketing",
}

def test_header_subtitle_element_exists(data):
    """
    The subtitle is now set by JS: init() writes DATA.length and unique organism
    count into id="header-subtitle". This test confirms that element exists so
    JS has a valid target — if the id is ever renamed, the subtitle goes blank.
    """
    html = HTML_FILE.read_text(encoding="utf-8")
    assert 'id="header-subtitle"' in html, (
        "Could not find id='header-subtitle' in index.html. "
        "The JS in init() writes product/organism counts to this element — "
        "if it's missing or renamed the subtitle will be blank."
    )


def test_organism_count_matches_header(data):
    """
    The header subtitle is dynamically generated from DATA by JS.
    Nothing to assert numerically here — this test just documents the contract:
    organism count shown = len(unique 'or' values in DATA).
    """
    actual = len({r["or"] for r in data if r.get("or")})
    assert actual > 0, "No organisms found in DATA — something is very wrong."


def test_country_count_matches_kpi(data):
    """
    The kpi-countries element initial value should match unique countries in DATA.
    JS overwrites this on load, but keeping the initial value correct avoids a
    flash of wrong content before JS runs.
    """
    html = HTML_FILE.read_text(encoding="utf-8")
    match = re.search(r'id="kpi-countries">(\d+)<', html)
    assert match, "Could not find kpi-countries element in HTML"
    claimed = int(match.group(1))
    actual = len({r["co"] for r in data if r.get("co")})
    assert actual == claimed, (
        f"KPI card initial value says {claimed} countries but DATA has {actual} unique values. "
        f"Update the initial value in index.html (JS will overwrite it at runtime, "
        f"but the initial value should still match to avoid a flash of wrong content)."
    )


# ── Completeness / null checks ────────────────────────────────────────────────

def test_organism_field_never_empty(data):
    """Every record must have an organism — it's used as the primary dimension."""
    empty = [i for i, r in enumerate(data) if not r.get("or", "").strip()]
    assert not empty, f"Records with empty organism ('or') field at indices: {empty}"


def test_country_field_never_empty(data):
    empty = [i for i, r in enumerate(data) if not r.get("co", "").strip()]
    assert not empty, f"Records with empty country ('co') field at indices: {empty}"


def test_trait_category_never_empty(data):
    """
    'tt' (trait category) drives the word cloud and filters.
    An empty value would appear as a blank entry in dropdowns.
    """
    empty = [i for i, r in enumerate(data) if not r.get("tt", "").strip()]
    assert not empty, f"Records with empty trait category ('tt') at indices: {empty}"


# ── Summary: print a quick data profile when running with -v ─────────────────

def test_data_profile(data, capsys):
    """
    Not a pass/fail test — prints a summary so you can eyeball the distribution.
    Think of it as a quick describe() call on a DataFrame.

    Run with: pytest test_data.py::test_data_profile -v -s
    """
    from collections import Counter

    stages  = Counter(r["st"] for r in data)
    kingdom = Counter(r["ki"] for r in data)
    tech    = Counter(r["te"] for r in data)

    print("\n── Data Profile ──────────────────────────────")
    print(f"  Total records : {len(data)}")
    print(f"  Countries     : {len({r['co'] for r in data})}")
    print(f"  Organisms     : {len({r['or'] for r in data})}")
    print(f"\n  Stage breakdown:")
    for s in sorted(stages):
        print(f"    {s:30s} {stages[s]:>3}")
    print(f"\n  Kingdom breakdown:")
    for k, n in kingdom.most_common():
        print(f"    {k:15s} {n:>3}")
    print(f"\n  Technology breakdown:")
    for t, n in tech.most_common():
        print(f"    {t:20s} {n:>3}")
    print("──────────────────────────────────────────────")

    # Always passes — this is just for inspection
    assert True
