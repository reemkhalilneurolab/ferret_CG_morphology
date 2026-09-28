"""Render every reconstruction in the cohort with a calibrated scale bar."""
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import config

config.PLOT_STEP_FUNCTION = False

import neurotopo.neuron_processor as npp  # noqa: E402
import neurotopo.utils as utl  # noqa: E402

OUT = os.path.join(config.PROJECT_ROOT, "figures", "neuron_images_scalebar")
os.makedirs(OUT, exist_ok=True)

AREA = {"PMLS": "PMLS", "PLLS": "PLLS", "21a": "21A"}


def area_of(name):
    for key, label in AREA.items():
        if key.lower() in name.lower():
            return label
    return "V1-V2"


files = sorted(glob.glob(os.path.join(config.BASE_DIRECTORY, "**", "*.swc"),
                         recursive=True))
excluded = set(config.EXCLUDED_NEURONS)
rendered = 0
for path in files:
    name = os.path.basename(path).replace(".CNG.swc", "").replace(".swc", "")
    name = name.replace(" ", "_").replace(".", "_").replace("-", "_")
    if name in excluded:
        continue
    try:
        result = npp.process_neuron(path)
        utl.plot_swc(result[6], name, area_of(name), OUT)
        rendered += 1
    except Exception as error:  # noqa: BLE001
        print(f"{name}: {type(error).__name__}: {error}")

print(f"rendered {rendered} reconstructions to {OUT}")
