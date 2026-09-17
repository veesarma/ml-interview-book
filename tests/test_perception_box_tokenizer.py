import numpy as np

from mlbook.perception.box_tokenizer import BoxTokenizer


def test_encode_layout_and_vocab():
    tok = BoxTokenizer(n_bins=100, num_classes=3, image_hw=(200, 400))
    boxes = np.array([[40.0, 20.0, 200.0, 100.0]])
    seq = tok.encode(boxes, np.array([2]))
    assert seq.tolist() == [10, 10, 50, 50, 102, 103]  # y0 x0 y1 x1 class EOS
    assert tok.vocab_size == 104 and tok.eos == 103


def test_round_trip_within_quantisation_error():
    rng = np.random.default_rng(0)
    tok = BoxTokenizer(n_bins=1024, num_classes=10, image_hw=(480, 640))
    xy0 = rng.uniform(0, 300, size=(20, 2))
    boxes = np.concatenate([xy0, xy0 + rng.uniform(5, 100, size=(20, 2))], axis=1)
    labels = rng.integers(0, 10, size=20)
    out_boxes, out_labels = tok.decode(tok.encode(boxes, labels))
    assert np.array_equal(out_labels, labels)
    tol = np.array([640, 480, 640, 480]) / (2 * 1023) + 1e-9
    assert (np.abs(out_boxes - boxes) <= tol).all()


def test_decode_stops_at_eos_and_drops_malformed_tail():
    tok = BoxTokenizer(n_bins=100, num_classes=3, image_hw=(100, 100))
    good = [10, 10, 50, 50, 101]
    bad_class = [10, 10, 50, 50, 55]  # fifth token is a location token, not a class
    seq = np.array(good + bad_class + [tok.eos] + good)
    boxes, labels = tok.decode(seq)
    assert boxes.shape == (1, 4) and labels.tolist() == [1]


def test_to_text_paligemma_style():
    tok = BoxTokenizer(n_bins=1024, num_classes=2, image_hw=(1024, 1024))
    s = tok.to_text(np.array([[0.0, 0.0, 1024.0, 512.0]]), np.array([1]), ["cat", "dog"])
    assert s == "<loc0000><loc0000><loc0512><loc1023> dog"
