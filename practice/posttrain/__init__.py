"""Part VII -- post-training: SFT, reward models, PPO-RLHF, DPO, GRPO and test-time compute.

All algorithms run on :class:`mlbook.posttrain.toy_lm.TinyCausalLM`, a self-contained
toy causal LM, so that every chapter's code trains on a CPU in seconds.
"""

from mlbook.posttrain import chat_template, dpo, grpo, ppo_lm, reward_model, sft, test_time, toy_lm, verifiers

__all__ = ["chat_template", "dpo", "grpo", "ppo_lm", "reward_model", "sft", "test_time", "toy_lm", "verifiers"]
