from huggingface_hub import snapshot_download, login
import os

token = "YOUR_HF_TOKEN".strip()
repo_id = "ShapeNet/ShapeSplatsV1"
local_dir = "gs_data/shapesplat"

print(f"Attempting to download {repo_id} to {local_dir}...")

try:
    login(token=token, add_to_git_credential=False)
    print("Login successful.")
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        token=token,
        resume_download=True
    )
    print("Download completed successfully.")
except Exception as e:
    print(f"Download failed: {e}")
