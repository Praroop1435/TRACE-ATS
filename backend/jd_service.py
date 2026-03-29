"""
jd_service.py — Job Description file management.

Reads from the JD/ directory, categorizes filenames into
Fresher / Experienced / General, and groups by role.
"""

from pathlib import Path
from functools import lru_cache

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
JD_FOLDER = BASE_DIR / "JD"


def _classify(filename: str) -> str:
    """Classify a JD file as fresher, experienced, or general."""
    lower = filename.lower()
    # Check "nonfresher" BEFORE "fresher" (substring match issue)
    if "nonfresher" in lower or "non_fresher" in lower or "non-fresher" in lower:
        return "experienced"
    if "fresher" in lower:
        return "fresher"
    return "general"


def _role_name(filename: str) -> str:
    """Extract a human-readable role name from filename."""
    name = filename.replace(".txt", "")
    # Remove category suffixes
    for suffix in ["_Fresher", "_NonFresher", "-job-description"]:
        name = name.replace(suffix, "")
    # Convert separators to spaces and title-case
    name = name.replace("_", " ").replace("-", " ")
    return name.strip().title()


@lru_cache(maxsize=1)
def list_jds() -> dict:
    """
    Return all JD files grouped by category.

    {
      "fresher":    [{"filename": "...", "role": "..."}],
      "experienced":[{"filename": "...", "role": "..."}],
      "general":    [{"filename": "...", "role": "..."}],
    }
    """
    groups: dict[str, list[dict]] = {
        "fresher": [],
        "experienced": [],
        "general": [],
    }

    if not JD_FOLDER.exists():
        return groups

    for f in sorted(JD_FOLDER.iterdir()):
        if f.suffix == ".txt" and f.is_file():
            cat = _classify(f.name)
            groups[cat].append({
                "filename": f.name,
                "role": _role_name(f.name),
            })

    return groups


def get_jd_text(filename: str) -> str | None:
    """Read the text content of a JD file. Returns None if not found."""
    path = JD_FOLDER / filename
    if not path.exists() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def all_filenames() -> list[str]:
    """Flat list of every JD filename."""
    groups = list_jds()
    return [
        item["filename"]
        for cat in groups.values()
        for item in cat
    ]
