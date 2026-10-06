"""Download and extract LibriSpeech subsets from OpenSLR.

Usage:
    python scripts/download_librispeech.py --subset dev-clean          # 40 speakers, ~340 MB (quick test)
    python scripts/download_librispeech.py --subset train-clean-100    # 251 speakers, ~6.3 GB (baseline)
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://www.openslr.org/resources/12"
SUBSETS = {
    "dev-clean": "≈340 MB · 40 speakers",
    "test-clean": "≈350 MB · 40 speakers",
    "train-clean-100": "≈6.3 GB · 251 speakers",
    "train-clean-360": "≈23 GB · 921 speakers",
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"✓ {dest.name} already downloaded")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=SUBSETS, default="dev-clean")
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    target = args.root / "LibriSpeech" / args.subset
    if target.exists():
        print(f"✓ {target} already extracted")
        return

    print(f"Downloading {args.subset} ({SUBSETS[args.subset]})")
    archive = args.root / f"{args.subset}.tar.gz"
    download(f"{BASE_URL}/{args.subset}.tar.gz", archive)

    print("Extracting…")
    with tarfile.open(archive) as tar:
        tar.extractall(args.root, filter="data")
    if not args.keep_archive:
        archive.unlink()
    print(f"✓ Ready at {target}")


if __name__ == "__main__":
    main()
