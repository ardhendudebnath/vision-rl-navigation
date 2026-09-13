"""Verify every relative link and repo path named in the docs resolves.

A committee skims the repo before it reads anything, so a dead link costs more
than any single result in it is worth. This has already caught two module
paths written in shorthand that would fail a reader's copy-paste.

    python scripts/check_docs.py

Exits non-zero on the first broken reference, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import re
import sys

#: Inline code that looks like a repo path. Anything matching this must exist,
#: unless it is listed as external below.
CODE_PATH = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|sh|yaml|yml|md|json|gif|mp4))`")
MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

#: Paths that belong to other packages and are correctly absent from this
#: repository. Listed explicitly so the checker stays strict about everything
#: else rather than being loosened with a wildcard.
EXTERNAL = {
    "nav2_bringup/navigation_launch.py",
}


def docs() -> list[str]:
    return (["README.md"]
            + sorted(glob.glob("docs/*.md"))
            + sorted(glob.glob("ros2_bridge/*.md"))
            + sorted(glob.glob("results/*.md")))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    broken, checked = [], 0
    for doc in docs():
        base = os.path.dirname(doc)
        text = io.open(doc, encoding="utf-8").read()

        for label, target in MD_LINK.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            checked += 1
            path = os.path.normpath(os.path.join(base, target.split("#")[0]))
            if not os.path.exists(path):
                broken.append((doc, "link", target, label[:40]))

        for target in sorted(set(CODE_PATH.findall(text))):
            if "/" not in target or target in EXTERNAL:
                continue
            checked += 1
            if not (os.path.exists(target)
                    or os.path.exists(os.path.normpath(os.path.join(base, target)))):
                broken.append((doc, "path", target, ""))

    if not args.quiet:
        print("checked {} references across {} docs".format(checked, len(docs())))
    for doc, kind, target, label in broken:
        print("  BROKEN {:5s} {:36s} in {}  {}".format(kind, target, doc, label),
              file=sys.stderr)
    if broken:
        print("\n{} broken reference(s)".format(len(broken)), file=sys.stderr)
        return 1
    if not args.quiet:
        print("all resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
