"""Create a Space and upload only the deployment files; never uploads local secrets."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi
from dotenv import dotenv_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("space_id", nargs="?", help="Default: authenticated user/IndicReason-Tamil-TAPO")
    parser.add_argument("--public", action="store_true", help="Create a public Space (default: private)")
    args = parser.parse_args()
    token = dotenv_values(Path(__file__).resolve().parent.parent / ".env").get("HF_TOKEN")
    api = HfApi(token=token)
    space_id = args.space_id or f"{api.whoami()['name']}/IndicReason-Tamil-TAPO"
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio",
                    space_hardware="zero-a10g", private=not args.public, exist_ok=True)
    api.request_space_hardware(repo_id=space_id, hardware="zero-a10g")
    api.upload_folder(
        repo_id=space_id, repo_type="space", folder_path=str(Path(__file__).parent),
        allow_patterns=["app.py", "requirements.txt", "README.md"],
        commit_message="Deploy Tamil TAPO math demo",
    )
    print(f"https://huggingface.co/spaces/{space_id}")
    runtime = api.get_space_runtime(space_id)
    print(f"Stage: {runtime.stage}; requested hardware: {runtime.requested_hardware}")


if __name__ == "__main__":
    main()
