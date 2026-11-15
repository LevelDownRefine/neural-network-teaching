"""Download and verify the official SHD training and test archives."""

import argparse
import gzip
import hashlib
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "outputs" / "Section2_Task3_SHD"
BASE_URL = "https://zenkelab.org/datasets/shd/"
ARCHIVES = {
    "shd_train.h5.gz": "d47c9825dee33347913e8ce0f2be08b0",
    "shd_test.h5.gz": "3062a80ec0c5719404d5b02e166543b1",
}


def file_digest(path, algorithm="sha256"):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def download_verified(url, path, expected, checksum=file_digest):
    if path.exists():
        if checksum(path) != expected:
            raise ValueError(f"Checksum mismatch for {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, suffix=".download", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with urlopen(url, timeout=75) as source, temporary.open("wb") as handle:
            shutil.copyfileobj(source, handle)
        if checksum(temporary) != expected:
            raise ValueError(f"Checksum mismatch for {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def extract_archive(archive):
    raw = archive.with_suffix("")
    if raw.exists():
        return raw
    with NamedTemporaryFile(
        dir=raw.parent, suffix=".h5.partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with gzip.open(archive, "rb") as source, temporary.open("wb") as handle:
            shutil.copyfileobj(source, handle)
        temporary.replace(raw)
    finally:
        temporary.unlink(missing_ok=True)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for name, digest in ARCHIVES.items():
        archive = download_verified(
            BASE_URL + name,
            args.out / "data" / name,
            digest,
            lambda p: file_digest(p, "md5"),
        )
        print(extract_archive(archive), flush=True)


if __name__ == "__main__":
    main()
