# Training and fine-tuning

## Scope

Use training only when required for model support or an optimization artifact: adapter fine-tuning, quantization-aware training, speculative drafter/head training, distillation, architecture adaptation, or task validation.

## MLX training loop requirements

- explicit lazy evaluation boundaries;
- optimizer/model/random state captured correctly for compilation;
- gradient accumulation semantics verified;
- train/eval mode and dropout behavior tested;
- checkpoint save/load and resume parity;
- deterministic small-batch overfit test;
- memory and graph-retention monitoring.

## Adapter path

Prefer LoRA or another parameter-efficient adapter for compatibility experiments. Record target modules, rank, scaling, dropout, base revision, tokenizer, dataset, seed, and merge status.

## Speculative artifacts

Medusa/MTP/EAGLE/diffusion drafters are trained components, not runtime switches. Define:

- target model and frozen/trainable parts;
- feature/logit interface;
- training data and on/off-policy behavior;
- loss and acceptance objective;
- tokenizer/vocabulary compatibility;
- evaluation across tasks and sampling regimes;
- artifact license and target revision lock.

## Quantization-aware training

Use when post-training quantization fails a quality target and a compatible low-bit kernel exists. Do not train for a format that the intended MLX runtime cannot execute efficiently.

## Audio fine-tuning

Preserve sample rate, loudness policy, segmentation, speaker splits, text normalization, phonemization, and codec version. Evaluate content leakage and voice-cloning consent/licensing where relevant.
