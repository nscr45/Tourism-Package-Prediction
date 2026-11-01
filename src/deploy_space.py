"""Deploy the Streamlit app to the existing Hugging Face Space.

Run from the repository root: python src/deploy_space.py
Only the files listed below are uploaded. Unrelated files already in the Space
are left untouched.
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

HF_USERNAME = "Sricharan451706"
SPACE_NAME = "Tourism-Package-Predictor-Space"
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"
ROOT = Path(__file__).resolve().parents[1]

FILES_TO_UPLOAD = {
    ROOT / "app" / "app.py": "app.py",
    ROOT / "app" / "requirements.txt": "requirements.txt",
    ROOT / "app" / "tourism_hero.svg": "tourism_hero.svg",
    ROOT / "models" / "model.joblib": "model.joblib",
}


def deploy() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Add a Hugging Face write token to your environment.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="streamlit",
        exist_ok=True,
    )

    uploaded = []
    skipped = []
    for local_path, remote_path in FILES_TO_UPLOAD.items():
        if not local_path.is_file():
            skipped.append(str(local_path.relative_to(ROOT)))
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=REPO_ID,
            repo_type="space",
        )
        uploaded.append(remote_path)
        print(f"Uploaded {remote_path}")

    print(f"Deployment upload complete. Updated: {', '.join(uploaded) or 'none'}")
    if skipped:
        print(f"Skipped missing optional files: {', '.join(skipped)}")
    print(f"Space: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    deploy()
