"""GPT: overfits a tiny sequence; KV-cached generation == uncached; parameter formula; sampling."""
import torch

torch.set_num_threads(1)  # tiny CPU models: one thread is faster than oversubscribed BLAS threads

from mlbook.transformer.generation import generate, sample_next_token
from mlbook.transformer.gpt import GPT, GPTConfig, count_parameters, gpt_param_count, inference_flops_per_token, training_flops_per_token


def _tiny(positional: str = "learned", **kw) -> GPT:
    cfg = GPTConfig(vocab_size=20, block_size=32, n_layers=2, n_heads=4, d_model=32, positional=positional, **kw)
    return GPT(cfg)


def test_gpt_param_count_formula_matches_module():
    for cfg in (
        GPTConfig(vocab_size=100, block_size=64, n_layers=3, n_heads=4, d_model=48),
        GPTConfig(vocab_size=100, block_size=64, n_layers=2, n_heads=2, d_model=16, tie_weights=False),
        GPTConfig(vocab_size=50, block_size=8, n_layers=1, n_heads=1, d_model=8, positional="rope"),
    ):
        assert count_parameters(GPT(cfg)) == gpt_param_count(cfg)
    assert training_flops_per_token(10) == 60 and inference_flops_per_token(10) == 20


def test_gpt_overfits_tiny_sequence():
    model = _tiny()
    seq = torch.tensor([[3, 7, 1, 9, 4, 4, 12, 0, 5, 6, 3, 7]])
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
    model.eval()
    logits, _ = model(x)
    assert torch.equal(logits.argmax(-1), y)


def test_gpt_is_causal():
    model = _tiny().eval()
    x = torch.randint(0, 20, (1, 10))
    logits, _ = model(x)
    x2 = x.clone()
    x2[0, 6] = (x2[0, 6] + 1) % 20
    logits2, _ = model(x2)
    torch.testing.assert_close(logits[:, :6], logits2[:, :6])
    assert not torch.allclose(logits[:, 6:], logits2[:, 6:])


def test_kv_cached_generation_equals_uncached():
    for positional in ("learned", "rope"):
        model = _tiny(positional).eval()
        prompt = torch.randint(0, 20, (2, 5))
        a = generate(model, prompt, max_new_tokens=12, greedy=True, use_cache=True)
        b = generate(model, prompt, max_new_tokens=12, greedy=True, use_cache=False)
        assert torch.equal(a, b), positional
        # and the cached logits themselves match the full forward at every step
        cache = model.new_cache(2)
        model(prompt, cache=cache)
        for t in range(5, a.shape[1] - 1):
            step_logits, _ = model(a[:, t : t + 1], cache=cache)
            full_logits, _ = model(a[:, : t + 1])
            torch.testing.assert_close(step_logits[:, -1], full_logits[:, -1], atol=1e-5, rtol=1e-4)


def test_sample_next_token_modes():
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.0]])
    assert sample_next_token(logits, greedy=True).item() == 1
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        assert sample_next_token(logits, top_k=1, generator=g).item() == 1
        assert sample_next_token(logits, top_p=0.5, generator=g).item() == 1  # nucleus = {1} (p_1 ~ 0.93)
    seen = {sample_next_token(logits, temperature=5.0, generator=g).item() for _ in range(200)}
    assert len(seen) >= 3  # high temperature explores
