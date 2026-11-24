"""Check that renamed teaching notebooks retain valid local document links."""

import json
import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = [
    "lif/README.md",
    "lif/neuron_dynamics.ipynb",
    "lif/experiment1/threshold_effect.ipynb",
    "spike_encoding/README.md",
    "spike_encoding/task1/encoding.ipynb",
    "spike_encoding/task3/spikcommander_evaluation.ipynb",
    "spike_encoding/shd_baseline/README.md",
]


@pytest.mark.parametrize("relative", DOCUMENTS)
def test_teaching_document_local_links(relative):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        notebook = json.loads(text)
        text = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        target = unquote(target.split("#", 1)[0])
        assert (path.parent / target).exists(), f"Broken link in {relative}: {target}"
