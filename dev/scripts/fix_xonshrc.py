"""
Fix .xonshrc to work around an indentation bug related to
https://github.com/conda-incubator/setup-miniconda/blame/f77e237980e9445f93e2033f0c4a695a697a967a/src/conda.ts#L387-L397
and possibly "conda init".
"""

import re
from pathlib import Path

# Looking for Conda Setup including odd indentation.
PATTERN = re.compile("""  # ----------------------------------------------------------------------------
  # Conda Setup Action: Basic configuration""")


def fix_xonshrc(xonshrc):
    if not xonshrc.exists():
        print("No .xonshrc")
        return

    split = PATTERN.split(xonshrc.read_text())
    if len(split) != 2:
        print("Broken .xonshrc not found")
        return  # already fixed or not as expected

    pattern_lines = PATTERN.pattern.splitlines()
    pattern_lines[1:1] = ["if True:"]

    split[1:1] = pattern_lines

    fixed = "\n".join(split)
    xonshrc.write_text(fixed)

    print(".xonshrc is now", fixed)


if __name__ == "__main__":
    fix_xonshrc(Path.home() / ".xonshrc")
