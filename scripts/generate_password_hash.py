from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.auth import hash_password


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a SamVision AI admin password hash."
    )
    parser.add_argument(
        "--password",
        help="Password to hash. Omit this option to enter it securely.",
    )
    args = parser.parse_args()

    password = args.password
    if password is None:
        password = getpass.getpass("Admin password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match.", file=sys.stderr)
            return 1

    print(hash_password(password))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
