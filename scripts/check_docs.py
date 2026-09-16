"""Verify every relative link and repo path named in the docs resolves, and
that no generated prose still contains an unrendered format placeholder.

A committee skims the repo before it reads anything, so a dead link costs more
than any single result in it is worth. This has already caught two module
paths written in shorthand that would fail a reader's copy-paste.

    python scripts/check_docs.py

Exits non-zero on the first broken reference, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import glob
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

#: An f-string placeholder that reached a document unrendered. Result sections
#: are written by scripts that interpolate numbers out of results/*.json, and
#: one of those templates was missing its `f` prefix -- so the report shipped
#: "({f['delta']:+.3f}, p = {f['p']:.3f})" as literal text, in the paragraph
#: stating the headline of an experiment. Nothing else caught it: the links
#: resolved, the tests passed, and check_numbers.py compares result files
#: against hand-typed literals rather than reading the prose. Two shapes, both
#: unambiguous in Markdown: a subscript like `{d['k']}` and a format spec like
#: `{x:+.3f}`.
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\s*\[[^\]]*\][^}]*\}"
                         r"|\{[A-Za-z_][A-Za-z0-9_]*:[<>^+\-0-9.,]*[dfegs%]\}")


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
        text = open(doc, encoding="utf-8").read()

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

        for line_no, line in enumerate(text.splitlines(), start=1):
            for hit in PLACEHOLDER.findall(line):
                broken.append((doc, "fstr", hit[:36], f"line {line_no}"))
        checked += 1

    if not args.quiet:
        print(f"checked {checked} references across {len(docs())} docs")
    for doc, kind, target, label in broken:
        print(f"  BROKEN {kind:5s} {target:36s} in {doc}  {label}",
              file=sys.stderr)
    if broken:
        print(f"\n{len(broken)} broken reference(s)", file=sys.stderr)
        return 1
    if not args.quiet:
        print("all resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
