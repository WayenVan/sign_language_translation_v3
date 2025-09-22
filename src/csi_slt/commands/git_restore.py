from ..misc.git_utils import restore_git_state
import argparse


def main(args):
    state_dir = getattr(args, "state_dir")

    if not isinstance(state_dir, str):
        raise ValueError("state_dir must be a string")

    restore_git_state(state_dir)
    print("✅ Git state restored!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restore the git state of the current repository from a specified directory."
    )
    parser.add_argument(
        "--state_dir",
        type=str,
        required=True,
        help="Directory where the git state is saved.",
    )
    args = parser.parse_args()
    main(args)
