import gzip
import shutil
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
raw = root / "data" / "MNIST" / "raw"
raw.mkdir(parents=True, exist_ok=True)
for name in (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
):
    p = raw / name
    if not p.exists():
        urllib.request.urlretrieve(
            "https://ossci-datasets.s3.amazonaws.com/mnist/" + name, p
        )
    unpacked = p.with_suffix("")
    if not unpacked.exists():
        with gzip.open(p, "rb") as src, unpacked.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    print(p, p.stat().st_size, flush=True)
