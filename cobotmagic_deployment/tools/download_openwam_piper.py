"""Download the pinned official real-Piper release and verify LFS hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper"
REVISION = "12bfca6fb289174ab5a0f4f7eb363dc77ed265f0"
DEFAULT_DIR = "/workspace/project/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Real-RoboDojo-Piper"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_DIR)
    args = parser.parse_args()
    target = Path(args.output)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    snapshot_download(REPO, revision=REVISION, local_dir=str(target), max_workers=4)
    verified = []
    for entry in info.siblings:
        path = target / entry.rfilename
        if path.stat().st_size != entry.size:
            raise RuntimeError(f"size mismatch: {path}")
        record = {"path": entry.rfilename, "size": entry.size}
        if entry.lfs:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(16 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != entry.lfs.sha256:
                raise RuntimeError(f"SHA256 mismatch: {path}")
            record["sha256"] = digest.hexdigest()
        verified.append(record)
        print("verified", entry.rfilename, flush=True)
    (target / "verification.json").write_text(json.dumps(
        {"repo_id": REPO, "revision": REVISION, "files": verified}, indent=2) + "\n")


if __name__ == "__main__":
    main()
