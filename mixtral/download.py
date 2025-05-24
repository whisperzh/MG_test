from huggingface_hub import snapshot_download
SAVED_DIR = "/root/MG_test/checkpoints" # Specify the saved directory
# Download HF checkpoints
snapshot_download(repo_id="mistralai/Mixtral-8x7B-v0.1", ignore_patterns=["*.pt"], local_dir=SAVED_DIR, local_dir_use_symlinks=False)