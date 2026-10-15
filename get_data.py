#!/usr/bin/env python3
"""Download this package's data/ folder from its GitHub release and check it.

  python get_data.py [--archive FILE] [--keep]

Downloads metasplice_v2_reproducibility_data.tar.gz (291,082,658 bytes) from release v2.0.1 of
https://github.com/xiaoming-liu/MetaSplice-reproducibility, checks its SHA-256, unpacks data/ next to this script, and
checks every data file against MD5SUMS. --archive FILE uses a copy you already have; --keep keeps the downloaded
archive. Python 3.8 or newer, standard library only.
"""
import argparse, hashlib, os, sys, tarfile, urllib.request

TAG = "v2.0.1"
NAME = "metasplice_v2_reproducibility_data.tar.gz"
URL = f"https://github.com/xiaoming-liu/MetaSplice-reproducibility/releases/download/{TAG}/{NAME}"
SHA256 = "79bbb3ffd8ed3ec1acc627364680bcc5a7f07b398ea161edfdf89740630fc9db"
HERE = os.path.dirname(os.path.abspath(__file__))


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0 and blocks % 400 == 0:
            print(f"  {min(done, total) / 1e6:,.0f} of {total / 1e6:,.0f} MB", flush=True)
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path + ".part", progress)
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", help="a copy of the data archive you already have")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded archive")
    args = ap.parse_args()

    archive = args.archive or os.path.join(HERE, NAME)
    if not args.archive and not (os.path.exists(archive) and digest(archive, "sha256") == SHA256):
        download(URL, archive)
    got = digest(archive, "sha256")
    if got != SHA256:
        sys.exit(f"{archive}: SHA-256 {got}, expected {SHA256}")

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        unsafe = [m.name for m in members if not (m.isfile() or m.isdir()) or m.name.startswith("/")
                  or ".." in m.name.split("/") or not (m.name == "data" or m.name.startswith("data/"))]
        if unsafe:
            sys.exit(f"{archive}: unexpected entries, e.g. {unsafe[0]}")
        extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(HERE, members=members, **extra)

    want = {}
    with open(os.path.join(HERE, "MD5SUMS")) as f:
        for line in f:
            h, rel = line.rstrip("\n").split("  ", 1)
            if rel.startswith("data/"):
                want[rel] = h
    bad = [rel for rel, h in sorted(want.items())
           if not os.path.exists(os.path.join(HERE, rel)) or digest(os.path.join(HERE, rel), "md5") != h]
    for rel in bad:
        print(f"  DIFFERS or missing: {rel}")
    if not args.archive and not args.keep:
        os.remove(archive)
    print(f"{len(want) - len(bad)} of {len(want)} data files match MD5SUMS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
