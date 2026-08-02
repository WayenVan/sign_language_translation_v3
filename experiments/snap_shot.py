from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="nvidia/C-RADIOv4-H",
    local_dir="./outputs/C-RADIOv4-H-code",
    allow_patterns=[
        "*.py",
        "*.json",
        "README.md",
    ],
)

print(local_path)
