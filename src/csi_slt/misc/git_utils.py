import subprocess
import os
import datetime
import tarfile
import argparse


def run_git_cmd(args, capture_output=True):
    """运行 git 命令"""
    result = subprocess.run(
        ["git"] + args, capture_output=capture_output, text=True, check=True
    )
    if capture_output:
        return result.stdout.strip()
    return None


def save_git_state(state_dir=None):
    if state_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        state_dir = f"git_state_{timestamp}"

    os.makedirs(state_dir, exist_ok=True)

    # commit
    commit = run_git_cmd(["rev-parse", "HEAD"])
    with open(os.path.join(state_dir, "commit.txt"), "w") as f:
        f.write(commit + "\n")

    # branch
    branch = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
    with open(os.path.join(state_dir, "branch.txt"), "w") as f:
        f.write(branch + "\n")

    # unstaged diff
    unstaged_patch = run_git_cmd(["diff"])
    with open(os.path.join(state_dir, "unstaged.patch"), "w") as f:
        f.write(unstaged_patch)

    # staged diff
    staged_patch = run_git_cmd(["diff", "--cached"])
    with open(os.path.join(state_dir, "staged.patch"), "w") as f:
        f.write(staged_patch)

    # untracked files
    untracked_files = run_git_cmd(["ls-files", "--others", "--exclude-standard"])
    if untracked_files.strip():
        tar_path = os.path.join(state_dir, "untracked.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for filepath in untracked_files.splitlines():
                tar.add(filepath)

    print(f"✅ Git state saved in {state_dir}")


def restore_git_state(state_dir):
    if not os.path.isdir(state_dir):
        raise FileNotFoundError(f"{state_dir} does not exist")

    # commit
    commit_file = os.path.join(state_dir, "commit.txt")
    if os.path.exists(commit_file):
        commit = open(commit_file).read().strip()
        print(f"➡️ Checkout commit {commit}")
        run_git_cmd(["checkout", commit])

    # branch
    branch_file = os.path.join(state_dir, "branch.txt")
    if os.path.exists(branch_file):
        branch = open(branch_file).read().strip()
        print(f"➡️ Switching to branch {branch}")
        run_git_cmd(["checkout", branch])

    # unstaged patch
    unstaged_patch = os.path.join(state_dir, "unstaged.patch")
    if os.path.exists(unstaged_patch) and os.path.getsize(unstaged_patch) > 0:
        print("➡️ Applying unstaged patch")
        run_git_cmd(["apply", unstaged_patch])

    # staged patch
    staged_patch = os.path.join(state_dir, "staged.patch")
    if os.path.exists(staged_patch) and os.path.getsize(staged_patch) > 0:
        print("➡️ Applying staged patch (index only)")
        run_git_cmd(["apply", "--cached", staged_patch])

    # untracked files
    untracked_tar = os.path.join(state_dir, "untracked.tar.gz")
    if os.path.exists(untracked_tar):
        print("➡️ Restoring untracked files")
        with tarfile.open(untracked_tar, "r:gz") as tar:
            tar.extractall(path=".")

    print("✅ Git state restored!")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Restore git state from saved directory"
#     )
#     parser.add_argument("state_dir", help="Path to git_state_YYYYMMDD_HHMMSS directory")
#     args = parser.parse_args()
#     restore_git_state(args.state_dir)

if __name__ == "__main__":
    save_git_state("outputs/git_info")
