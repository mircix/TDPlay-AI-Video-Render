"""Show the end of a log as an error annotation on the workflow run, readable without opening the logs.

    python .github/annotate.py TITLE LOG [LOG...]
"""

import sys
from pathlib import Path


def main() -> None:
    title, paths = sys.argv[1], sys.argv[2:]
    for path in map(Path, paths):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        tail = "\n".join(lines[-80:])
        message = tail.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error title={title} ({path.name})::{message}")


if __name__ == "__main__":
    main()
