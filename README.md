# GCG replication: Universal and Transferable Adversarial Attacks on Aligned Language Models

A from-scratch reimplementation of the Greedy Coordinate Gradient (GCG) attack from Zou et al., 2023 ([arXiv:2307.15043](https://arxiv.org/abs/2307.15043)), run against Vicuna-7B v1.5.

I wrote this to understand the method by rebuilding it rather than by reading about it. The attack code is mine; nothing in `llm-attack-replication/` was generated.

## Why this paper

Jailbreaks were a human craft before this paper. Automating them is hard because the input space is discrete tokens, so you cannot descend a gradient through it the way you can in the vision setting. GCG gets around that by using the gradient only to shortlist candidate substitutions, then evaluating those candidates directly and greedily keeping what lowers the loss.

The target is an affirmative prefix. You optimize a suffix so that the model begins its reply with "Sure, here is ...", on the reasoning that an aligned model is still a probabilistic continuation engine, and that the most likely continuation of an affirmative opening is the content itself.

## What is implemented

- `llm-attack-replication/maximize_prompt.py` - the attack loop. Gradient-shortlisted candidate selection over the one-hot token coordinates, greedy substitution, and loss ranking.
- Multi-prompt loss aggregation. Loss and gradient are averaged across a set of queries so the optimization targets a single suffix that works on all of them, rather than one suffix per query. This is the universality mechanism in the paper.
- Incremental query set expansion. New prompts are added after earlier ones have had time to converge, which the paper reports as more effective than optimizing the full set from the start.
- Running-average loss (`compute_loss_avg`) rather than a per-step reset, so the objective reflects performance across the whole query set.
- `llm-attack-replication/prompt_build_tools.py` - builds the attack prompt, inserts the suffix, and handles Vicuna's conversation format via FastChat.
- Checkpointing. The current suffix is written to `checkpoints/step_XXXX.txt` every step, and the running best to `store.txt`, so a run can be inspected or resumed.

## Setup

```
git clone --recurse-submodules https://github.com/thesuperRL/universal-llm-attacks
pip install -r llm-attack-replication/requirements.txt
```

Model weights are gitignored. Place Vicuna-7B v1.5 weights in `models/vicuna-7b-v1.5/`; the config and tokenizer are committed. Then:

```
python llm-attack-replication/maximize_prompt.py
```

## What I took from it

Two results in the paper's discussion mattered more to me than the jailbreak.

**Stacking a detector in front of the model does not compose into a defense.** If a classifier screens inputs before they reach the LLM, the attack simply becomes a joint attack on the classifier and the model, which the paper finds about equally easy. The same layering is harder to defeat in the vision domain. This generalizes badly for anyone whose safety architecture is a filter in front of a model, which is an architecture I have also built.

**Optimizing longer makes the suffix worse.** Extended runs overfit to the models in the optimization set and lose transfer to models outside it. The attack that generalizes is not the strongest attack against any single target.

Two smaller things worth recording: suffixes transfer across models with different tokenizers, which is not obvious if you think of the suffix as a token-level artifact; and concatenating multiple GCG suffixes raises attack success up to a point, past which the input length degrades the model's comprehension and success drops.

## Limitations

This is an implementation and reading exercise, not a result.

- Runs were short. The committed checkpoints cover the first optimization steps only, and I did not train to convergence.
- No transfer evaluation. The paper's central claim is universality and transfer across models; I optimized against a single model and did not measure attack success rate on held-out models.
- No benchmark numbers. I did not reproduce the harmful-strings or harmful-behaviors evaluations, so nothing here should be read as confirming or disputing the paper's reported rates.

The thing I would build next is the measurement side rather than more optimization: a held-out model set and an attack success rate harness, so that a run produces a number instead of a suffix.
