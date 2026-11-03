from pathlib import Path
import urllib.request
import zipfile

root = Path(__file__).resolve().parent
raw = root / 'data' / 'MNIST' / 'raw'
raw.mkdir(parents=True, exist_ok=True)
for name in ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'):
    p = raw / name
    if not p.exists():
        urllib.request.urlretrieve('https://ossci-datasets.s3.amazonaws.com/mnist/' + name, p)
    print(p, p.stat().st_size, flush=True)
source = root / 'sources'
source.mkdir(exist_ok=True)
for name, url in [('cnn_paper', 'https://codeload.github.com/ansh941/MnistSimpleCNN/zip/refs/heads/master')]:
    archive = source / (name + '.zip')
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(source)
    print(archive, archive.stat().st_size, flush=True)
