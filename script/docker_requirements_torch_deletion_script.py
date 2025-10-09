# Ugly script to remove torch from the dependencies since
# - it's already installed in the container
# - it makes the container heavier

import argparse
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "original_requirements_path",
    type=pathlib.Path,
    help="Path to the requirements exported by the Vectorizer package",
)
parser.add_argument(
    "torch_free_requirements_path",
    type=pathlib.Path,
    help="Path where the torch free requirements will be stored",
)

args = parser.parse_args()

with open(args.original_requirements_path, "r") as f_in, open(
    args.torch_free_requirements_path, "w"
) as f_out:
    for line in f_in:
        if line.startswith("torch") or line.startswith("nvidia-"):
            continue
        f_out.write(line)

sys.exit(0)