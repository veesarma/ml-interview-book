"""Test-time compute utilities: best-of-N, self-consistency voting, beam search, KL bound."""
import math

import torch

from mlbook.posttrain.chat_template import ToyTokenizer, prompt_ids
from mlbook.posttrain.test_time import beam_search, best_of_n, bon_kl_bound, expected_max_of_n_gaussian, self_consistency
from mlbook.posttrain.toy_lm import TinyCausalLM, ToyLMConfig
from mlbook.posttrain.verifiers import response_text, target_token_reward


def _model_and_prompt():
    torch.manual_seed(0)
    tok = ToyTokenizer()
    cfg = ToyLMConfig(vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, max_len=32)
    model = TinyCausalLM(cfg).eval()
    prompt = torch.tensor(prompt_ids([{"role": "user", "content": "hello"}], tok))
    return tok, model, prompt


def test_bon_kl_bound_values():
    assert bon_kl_bound(1) == 0.0
    assert abs(bon_kl_bound(4) - (math.log(4) - 0.75)) < 1e-12
    assert bon_kl_bound(64) > bon_kl_bound(16) > bon_kl_bound(4)
    # E[max of N] grows with N and is 0 for N = 1
    assert abs(expected_max_of_n_gaussian(1)) < 0.05
    assert expected_max_of_n_gaussian(16) > expected_max_of_n_gaussian(4) > 0.9


def test_best_of_n_returns_argmax_reward():
    tok, model, prompt = _model_and_prompt()
    target = tok.id_of["good"]
    reward_fn = lambda t, m: target_token_reward(t, m, target)
    best_tokens, best_mask, rewards = best_of_n(model, prompt, n=32, reward_fn=reward_fn, eos_id=tok.eos_id, max_new_tokens=6)
    assert rewards.shape == (32,)
    assert best_tokens.shape == best_mask.shape == (prompt.shape[0] + 6,)
    assert target_token_reward(best_tokens.unsqueeze(0), best_mask.unsqueeze(0), target)[0] == rewards.max()
    assert best_mask[: prompt.shape[0]].sum() == 0  # prompt tokens are never "response"


def test_self_consistency_majority_vote():
    tok, model, prompt = _model_and_prompt()
    extract = lambda t, m: response_text(t, m, tok).split(" ")[0] if response_text(t, m, tok) else ""
    answer, votes = self_consistency(model, prompt, n=40, extract_answer=extract, eos_id=tok.eos_id, max_new_tokens=1)
    assert sum(votes.values()) == 40
    assert votes[answer] == max(votes.values())


def test_beam_search_matches_exhaustive_search():
    """With beam width >= V^L the beam contains the true top sequences; compare against brute force for L = 2."""
    tok, model, prompt = _model_and_prompt()
    V = tok.vocab_size
    beams = beam_search(model, prompt, beam_width=8, max_new_tokens=2, eos_id=tok.eos_id)
    assert len(beams) == 8 and all(len(g) <= 2 for g, _ in beams)
    assert all(beams[i][1] >= beams[i + 1][1] for i in range(len(beams) - 1))
    # brute force over all V*V two-token continuations (plus one-token eos sequences)
    with torch.no_grad():
        logp1 = torch.log_softmax(model(prompt.unsqueeze(0))[0, -1], dim=-1)  # (V,)
        best = -1e9
        for a in range(V):
            if a == tok.eos_id:
                best = max(best, float(logp1[a]))
                continue
            seq = torch.cat([prompt, torch.tensor([a])]).unsqueeze(0)
            logp2 = torch.log_softmax(model(seq)[0, -1], dim=-1)
            best = max(best, float(logp1[a] + logp2.max()))
    assert abs(beams[0][1] - best) < 1e-4
