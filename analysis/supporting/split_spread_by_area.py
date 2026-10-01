"""Copy the Spread step-function figures into per-area folders.

The originals are left in place. Files are named
``{class}_{neuron}_spread.png`` where the class prefix is ``V1-V2`` or
``PMLS-PLLS``; the area is read from the neuron portion of the name, after the
prefix is removed, since the prefix itself contains both ``PMLS`` and ``PLLS``.
"""
import os
import re
import shutil

from _common import RUN

SOURCE = RUN / "plots" / "Descriptors" / "step_functions" / "spread_step_functions"
TARGET = RUN / "plots" / "Descriptors" / "step_functions" / "spread_step_functions_by_area"

EXPECTED = {"V1-V2": 67, "PMLS": 20, "PLLS": 20, "21A": 10}


def area_of(filename):
    name = filename
    for prefix in ("PMLS-PLLS_", "V1-V2_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    else:
        return None
    if re.search(r"21a", name, re.IGNORECASE):
        return "21A"
    if re.search(r"PLLS", name, re.IGNORECASE):
        return "PLLS"
    if re.search(r"PMLS", name, re.IGNORECASE):
        return "PMLS"
    return "V1-V2"


counts = {}
unmatched = []
for area in EXPECTED:
    os.makedirs(TARGET / area, exist_ok=True)

for filename in sorted(os.listdir(SOURCE)):
    if not filename.lower().endswith(".png"):
        continue
    area = area_of(filename)
    if area is None:
        unmatched.append(filename)
        continue
    shutil.copy2(SOURCE / filename, TARGET / area / filename)
    counts[area] = counts.get(area, 0) + 1

print(f"source: {SOURCE}")
print(f"target: {TARGET}\n")
for area, expected in EXPECTED.items():
    got = counts.get(area, 0)
    print(f"  {area:8s} {got:3d} copied   expected {expected:3d}   "
          f"{'ok' if got == expected else 'MISMATCH'}")
print(f"\n  total {sum(counts.values())}   unmatched {len(unmatched)}")
for name in unmatched:
    print("   ?", name)
print(f"\noriginal folder still holds "
      f"{len([f for f in os.listdir(SOURCE) if f.lower().endswith('.png')])} files")
