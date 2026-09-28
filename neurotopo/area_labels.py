"""Utilities for deriving ferret cortical area labels from SWC filenames."""
import re


def infer_area_label(filename: str) -> str:
    """Return ``21A``, ``PMLS``, ``PLLS``, or ``Unknown`` from an SWC name."""
    stem = re.sub(r"\.swc$", "", str(filename), flags=re.IGNORECASE)
    stem = re.sub(r"\.CNG$", "", stem, flags=re.IGNORECASE)
    match = re.search(r"(?:^|_)(21a|PMLS|PLLS)(?:_|$)", stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "Unknown"
