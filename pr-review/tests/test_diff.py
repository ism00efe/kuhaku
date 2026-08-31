from pr_review.diff import parse_diff


def test_parses_files_hunks_and_line_numbers(sample_diff):
    files = parse_diff(sample_diff)
    assert [f.path for f in files] == ["src/calc.py", "pyproject.toml"]
    calc = files[0]
    assert calc.status == "modified"
    added = calc.added_lines()
    assert ("def modulo(a, b):" in [t for _, t in added])
    # new line numbers are tracked
    nums = [n for n, t in added if t == "def modulo(a, b):"]
    assert nums and nums[0] == 5


def test_added_and_deleted_files():
    d = (
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+import os\n"
        "+print(os)\n"
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-x = 1\n"
    )
    files = parse_diff(d)
    assert files[0].status == "added"
    assert files[1].status == "deleted"
