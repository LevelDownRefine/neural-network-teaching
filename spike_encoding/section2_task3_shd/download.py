"""Download official SHD archives with progress and verify their checksums."""
import gzip
import hashlib
import shutil
import sys
import time
from pathlib import Path
from urllib.request import urlopen

DATA = Path(__file__).resolve().parent / 'shd_data'
BASE = 'https://zenkelab.org/datasets/'
ARCHIVES = {
    'shd_train.h5.gz': 'd47c9825dee33347913e8ce0f2be08b0',
    'shd_test.h5.gz': '3062a80ec0c5719404d5b02e166543b1',
}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'md5').hexdigest()


def download(name, expected):
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / name
    part = DATA / (name + '.part')
    if dest.exists():
        print(f'Checking existing file {dest} ...', flush=True)
        if digest(dest) != expected:
            raise ValueError(f'{dest} failed MD5 validation; move or delete it and retry')
        return dest
    url = BASE + name
    print(f'Connecting to {url} (timeout 30 seconds) ...', flush=True)
    # Write to a distinct temporary file; an interrupted download is never treated as complete.
    # Restart from zero on retry, since server Range behavior varies between mirrors.
    with urlopen(url, timeout=30) as source, part.open('wb') as target:
        total = int(source.headers.get('Content-Length') or 0)
        print(f'Connected. Saving to {part}; total: {total / 1048576:.1f} MiB' if total
              else f'Connected. Saving to {part}; size unknown', flush=True)
        count = 0
        start = last = time.monotonic()
        while True:
            chunk = source.read(64 * 1024)
            if not chunk:
                break
            target.write(chunk)
            count += len(chunk)
            now = time.monotonic()
            if now - last >= 1:
                percent = f' ({count / total:.1%})' if total else ''
                print(f'  {count / 1048576:.1f} MiB{percent}, {count / max(now-start, 0.001) / 1024:.1f} KiB/s', flush=True)
                last = now
    if total and count != total:
        raise ValueError(f'incomplete download: {count}/{total} bytes; retry the command')
    print(f'Downloaded {count / 1048576:.1f} MiB. Checking MD5 ...', flush=True)
    if digest(part) != expected:
        raise ValueError(f'MD5 mismatch for {part}; retry the command')
    part.replace(dest)
    return dest


def extract(archive):
    dest = archive.with_suffix('')
    if dest.exists():
        print(f'Already extracted: {dest}', flush=True)
        return
    temp = dest.with_name(dest.name + '.part')
    print(f'Decompressing {archive.name} ...', flush=True)
    with gzip.open(archive, 'rb') as source, temp.open('wb') as target:
        shutil.copyfileobj(source, target)
    temp.replace(dest)
    print(f'Ready: {dest}', flush=True)


def main():
    try:
        for name, expected in ARCHIVES.items():
            extract(download(name, expected))
    except (OSError, ValueError) as error:
        print(f'Download failed: {error}\nCheck network/proxy access and retry. '
              'Partial files are not valid datasets and will be restarted.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
