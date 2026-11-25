"""Unit and CLI tests for the wordfreq package."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wordfreq.cli import count_words, format_table, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _run_module(*args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wordfreq", *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


# --- count_words / format_table -------------------------------------------


def test_count_words_basic_ranking() -> None:
    text = "Apple banana apple, cherry! Banana apple."
    assert count_words(text) == [("apple", 3), ("banana", 2), ("cherry", 1)]


def test_count_words_breaks_ties_alphabetically() -> None:
    assert count_words("zeta alpha beta alpha beta") == [
        ("alpha", 2),
        ("beta", 2),
        ("zeta", 1),
    ]


def test_count_words_tie_order_ignores_input_order() -> None:
    forward = count_words("dog cat ant dog cat ant")
    reverse = count_words("ant cat dog ant cat dog")
    assert forward == reverse == [("ant", 2), ("cat", 2), ("dog", 2)]


def test_count_words_top_is_applied_after_tie_break() -> None:
    assert count_words("zeta alpha beta alpha beta", top=2) == [
        ("alpha", 2),
        ("beta", 2),
    ]


def test_count_words_top_zero_returns_empty_list() -> None:
    assert count_words("a b", top=0) == []


def test_count_words_empty_text_returns_empty_list() -> None:
    assert count_words("") == []


def test_count_words_lowercases_by_default() -> None:
    assert count_words("Apple APPLE apple") == [("apple", 3)]


def test_count_words_case_sensitive_preserves_case() -> None:
    # Sorted by (-count, word) with ASCII ordering: "apple" (2) first, then
    # "APPLE" < "Apple" as ties at count 1.
    assert count_words("Apple apple APPLE apple", ignore_case=False) == [
        ("apple", 2),
        ("APPLE", 1),
        ("Apple", 1),
    ]


def test_count_words_token_pattern_splits_on_punctuation_and_underscore() -> None:
    assert count_words("don't re-use_it") == [
        ("don", 1),
        ("it", 1),
        ("re", 1),
        ("t", 1),
        ("use", 1),
    ]


def test_count_words_min_len_filters_short_tokens() -> None:
    assert count_words("a bb ccc bb", min_len=2) == [("bb", 2), ("ccc", 1)]


def test_count_words_rejects_min_len_below_one() -> None:
    with pytest.raises(ValueError, match=r"^min_len must be >= 1$"):
        count_words("a", min_len=0)


def test_count_words_rejects_negative_top() -> None:
    with pytest.raises(ValueError, match=r"^top must be >= 0$"):
        count_words("a", top=-1)


def test_format_table_tsv() -> None:
    assert format_table([("apple", 3), ("banana", 2)]) == "apple\t3\nbanana\t2"


def test_format_table_single_row_has_no_trailing_newline() -> None:
    assert format_table([("apple", 3)]) == "apple\t3"


def test_format_table_empty_returns_empty_string() -> None:
    assert format_table([]) == ""


# --- main: stdin and file input -------------------------------------------


def test_cli_top_and_min_len(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("to be or not to be that is the question", encoding="utf-8")
    rc = main([str(sample), "--min-len", "3", "--top", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines() == ["not\t1", "question\t1"]


def test_cli_reads_stdin_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("the cat the\n"))
    assert main([]) == 0
    assert capsys.readouterr().out == "the\t2\ncat\t1\n"


def test_cli_dash_path_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("the cat the\n"))
    assert main(["-"]) == 0
    assert capsys.readouterr().out == "the\t2\ncat\t1\n"


def test_cli_tie_breaking_is_alphabetical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_text(tmp_path / "sample.txt", "zeta alpha beta alpha beta\n")
    assert main([str(sample)]) == 0
    assert capsys.readouterr().out == "alpha\t2\nbeta\t2\nzeta\t1\n"


def test_cli_case_sensitive_flag_preserves_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_text(tmp_path / "sample.txt", "Apple apple apple\n")
    assert main([str(sample), "--case-sensitive"]) == 0
    assert capsys.readouterr().out == "apple\t2\nApple\t1\n"


def test_cli_default_lowercases_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_text(tmp_path / "sample.txt", "Apple apple apple\n")
    assert main([str(sample)]) == 0
    assert capsys.readouterr().out == "apple\t3\n"


def test_cli_tsv_output_ends_with_single_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_text(tmp_path / "sample.txt", "apple apple banana\n")
    assert main([str(sample)]) == 0
    out = capsys.readouterr().out
    assert out == "apple\t2\nbanana\t1\n"
    assert out.endswith("\n")
    assert not out.endswith("\n\n")


def test_cli_empty_input_prints_nothing_for_tsv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_text(tmp_path / "empty.txt", "")
    assert main([str(sample)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --- main: --json output ---------------------------------------------------


def test_cli_json_output_ends_with_single_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_text(tmp_path / "sample.txt", "apple apple banana\n")
    assert main([str(sample), "--json"]) == 0
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert not out.endswith("\n\n")
    assert json.loads(out) == [
        {"word": "apple", "count": 2},
        {"word": "banana", "count": 1},
    ]


def test_cli_json_empty_input_is_empty_array(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_text(tmp_path / "empty.txt", "")
    assert main([str(sample), "--json"]) == 0
    assert capsys.readouterr().out == "[]\n"


def test_cli_json_honors_case_sensitive_top_and_min_len(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_text(tmp_path / "sample.txt", "a bb Bb cc cc\n")
    rc = main(
        [str(sample), "--json", "--case-sensitive", "--min-len", "2", "--top", "2"]
    )
    assert rc == 0
    # Case preserved: "Bb" and "bb" stay distinct; "a" is dropped by min_len.
    assert json.loads(capsys.readouterr().out) == [
        {"word": "cc", "count": 2},
        {"word": "Bb", "count": 1},
    ]


# --- argument validation and errors ----------------------------------------


def test_cli_min_len_zero_reports_message_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _write_text(tmp_path / "sample.txt", "apple\n")
    assert main([str(sample), "--min-len", "0"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "min_len must be >= 1" in captured.err


@pytest.mark.parametrize(
    "extra_args",
    [["--top", "-1"], ["--top", "abc"], ["--min-len", "abc"], ["--bogus"]],
    ids=["negative-top", "non-integer-top", "non-integer-min-len", "unknown-flag"],
)
def test_cli_invalid_arguments_return_2_with_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra_args: list[str]
) -> None:
    sample = _write_text(tmp_path / "sample.txt", "apple\n")
    assert main([str(sample), *extra_args]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != ""


def test_cli_missing_file_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does-not-exist.txt"
    assert main([str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read input" in captured.err


# --- python -m wordfreq ----------------------------------------------------


def test_python_m_wordfreq_reads_stdin_by_default() -> None:
    result = _run_module(stdin="the cat the\n")
    assert result.returncode == 0
    assert result.stdout == "the\t2\ncat\t1\n"


def test_python_m_wordfreq_reads_file_path(tmp_path: Path) -> None:
    sample = _write_text(tmp_path / "sample.txt", "zeta alpha beta alpha beta\n")
    result = _run_module(str(sample))
    assert result.returncode == 0
    assert result.stdout == "alpha\t2\nbeta\t2\nzeta\t1\n"


def test_python_m_wordfreq_json_output_ends_with_newline(tmp_path: Path) -> None:
    sample = _write_text(tmp_path / "sample.txt", "apple apple banana\n")
    result = _run_module(str(sample), "--json")
    assert result.returncode == 0
    assert result.stdout.endswith("\n")
    assert not result.stdout.endswith("\n\n")
    assert json.loads(result.stdout) == [
        {"word": "apple", "count": 2},
        {"word": "banana", "count": 1},
    ]


def test_python_m_wordfreq_empty_stdin_prints_nothing_for_tsv() -> None:
    result = _run_module(stdin="")
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_python_m_wordfreq_missing_file_returns_2(tmp_path: Path) -> None:
    result = _run_module(str(tmp_path / "missing.txt"))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr != ""
