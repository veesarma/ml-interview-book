import numpy as np

from mlbook.evaluation import text_metrics as tm


def test_edit_distance_kitten_sitting():
    d, s, dl, i = tm.edit_distance(list("kitten"), list("sitting"))
    assert d == 3 and s == 2 and i == 1 and dl == 0


def test_cer_and_wer():
    assert np.isclose(tm.cer("hello", "hallo"), 1 / 5)
    assert np.isclose(tm.wer("the quick brown fox", "the quick fox"), 1 / 4)
    assert tm.cer("abc", "abc") == 0.0 and tm.cer("", "x") == 1.0


def test_text_spotting_f1():
    pb = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=float)
    gb = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    p, r, f = tm.text_spotting_f1(pb, ["Stop", "exit", "noise"], gb, ["STOP", "EXlT"])
    assert np.isclose(p, 1 / 3) and np.isclose(r, 1 / 2) and np.isclose(f, 0.4)
