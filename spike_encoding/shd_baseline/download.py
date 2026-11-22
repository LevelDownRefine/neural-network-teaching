"""Download SHD with the shared verified dataset preparation helpers."""

import sys
from pathlib import Path

# Support the documented direct script invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from spike_encoding.prepare_shd import (  # noqa: E402
    ARCHIVES,
    BASE_URL,
    download_verified,
    extract_archive,
    file_digest,
)

DATA = Path(__file__).resolve().parent / "shd_data"


def main():
    for name, expected in ARCHIVES.items():
        print(f"Preparing {name} ...", flush=True)
        archive = download_verified(
            BASE_URL + name,
            DATA / name,
            expected,
            checksum=lambda path: file_digest(path, "md5"),
        )
        print(extract_archive(archive), flush=True)


if __name__ == "__main__":
    main()
