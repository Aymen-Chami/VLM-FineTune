"""Upload a trained adapter / checkpoint folder to a *private* Hugging Face Hub repo.

    python scripts/push_to_hub.py --folder outputs/pg2_448/final --repo paligemma2-3b-docchart-lora-448

The repo is created under the account of the token in $HF_HOME/token if it doesn't exist.
Optimizer/scheduler states are skipped (only the weights + configs are uploaded).
"""
from __future__ import annotations

import argparse

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo", required=True, help="repo name (without user) or full user/name id")
    ap.add_argument("--message", default="upload adapter")
    args = ap.parse_args()

    api = HfApi()
    repo_id = args.repo if "/" in args.repo else f"{api.whoami()['name']}/{args.repo}"
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=args.folder, commit_message=args.message,
                      ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state*.pth", "*.bin.index.json"])
    print(f"uploaded {args.folder} -> https://huggingface.co/{repo_id} (private)")


if __name__ == "__main__":
    main()
