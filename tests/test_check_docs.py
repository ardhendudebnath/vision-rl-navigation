"""Tests for the documentation checker.

The placeholder check exists because the report shipped a paragraph reading
"({f['delta']:+.3f}, p = {f['p']:.3f})" as literal text -- the headline
sentence of an experiment, with the numbers missing. Result sections are
written by scripts that interpolate out of results/*.json, and one template was
missing its `f` prefix.

Nothing else in the repo could have caught it. The links resolved, the tests
passed, and check_numbers.py compares result files against hand-typed literals
rather than reading the prose, so it verifies the transcription *intent* and
never notices that the document says something else entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_docs import PLACEHOLDER, main  # noqa: E402


@pytest.mark.parametrize("text", [
    "({f['delta']:+.3f}, p = {f['p']:.3f})",       # exactly what shipped
    "costs {abl['mean_delta']:+.3f} success",      # subscript, other variable
    "the arm scored {mean:.3f} overall",           # bare format spec
    "{rate:.1%} of episodes",                      # percentage spec
    "{cond['dynamic']['delta']:+.3f}",             # nested subscript
])
def test_unrendered_placeholders_are_caught(text):
    assert PLACEHOLDER.search(text), f"missed: {text}"


@pytest.mark.parametrize("text", [
    "Set `{'a': 1}` in the config",                # a dict in prose
    "use {x} as a variable name",                  # a brace-wrapped word
    "the set {1, 2, 3} is closed",                 # set notation
    "success rate 0.652 and p = 0.017",            # ordinary numbers
    "`dynamic_fast` raises them to 0.8-1.5 m/s",   # ordinary prose
])
def test_ordinary_prose_is_not_flagged(text):
    assert not PLACEHOLDER.search(text), f"false positive: {text}"


def test_the_real_docs_are_clean():
    """The checker must pass on the repository as committed."""
    assert main(["--quiet"]) == 0
