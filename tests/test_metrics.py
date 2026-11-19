import math

from src.evaluation.metrics import (
    anls, anls_single, levenshtein, postprocess_prediction,
    relaxed_accuracy, relaxed_match,
)


# ---------------------------------------------------------------- levenshtein
def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("flaw", "lawn") == 2


# ---------------------------------------------------------------- ANLS
def test_anls_exact_and_case_whitespace():
    assert anls_single("Hello World", ["hello world"]) == 1.0
    assert anls_single("  hello   world ", ["Hello World"]) == 1.0


def test_anls_partial_and_threshold():
    # 1 edit out of 5 chars -> NL = 0.2 -> score 0.8
    assert math.isclose(anls_single("abcdx", ["abcde"]), 0.8)
    # NL = 0.5 exactly -> not < 0.5 -> 0
    assert anls_single("abxx", ["abcd"]) == 0.0
    # completely wrong -> 0
    assert anls_single("zzz", ["abc"]) == 0.0


def test_anls_multiple_golds_takes_max():
    assert anls_single("1999", ["the year 1999", "1999"]) == 1.0


def test_anls_empty_prediction():
    assert anls_single("", ["abc"]) == 0.0


def test_anls_corpus_mean():
    assert math.isclose(anls(["a", "b"], [["a"], ["c"]]), 0.5)


# ---------------------------------------------------------------- relaxed accuracy
def test_relaxed_numeric_tolerance():
    assert relaxed_match("104.9", "100")       # 4.9% error
    assert not relaxed_match("105.1", "100")   # 5.1% error
    assert relaxed_match("95", "100")          # exactly 5%


def test_relaxed_percent_commas_dollar():
    assert relaxed_match("45%", "45")
    assert relaxed_match("1,000", "1000")
    assert relaxed_match("$12", "12")


def test_relaxed_gold_zero():
    assert relaxed_match("0", "0")
    assert relaxed_match("0.0", "0")
    assert not relaxed_match("0.01", "0")


def test_relaxed_text_answers_exact_match():
    assert relaxed_match("Yes", "yes")
    assert not relaxed_match("No", "yes")
    assert not relaxed_match("2015 and 2016", "2015")


def test_relaxed_corpus_mean():
    assert relaxed_accuracy(["10", "x"], [["10"], ["y"]]) == 0.5


# ---------------------------------------------------------------- postprocessing
def test_postprocess():
    assert postprocess_prediction("  42. ") == "42"
    assert postprocess_prediction("Yes") == "Yes"
