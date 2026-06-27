# Runbook: time-series forecasting

## Applies to

Forecasting models such as Time Series Transformer, Chronos-style sequence
models, TimesFM-style foundation forecasters, PatchTST, Informer, Autoformer,
and related architectures that consume historical values plus time/static
covariates to predict future horizons. Route tabular classifiers/regressors,
ranking, retrieval, and recommender systems to separate runbooks until they have
their own fixtures and task gates.

## Architecture fingerprint

Confirm:

- context length, prediction length, lag sequence, seasonal frequency, and
  patching or tokenization policy;
- target dimensionality, observed-mask behavior, missing-value policy, and
  scaling/normalization state;
- known-past, known-future, static categorical, static real, and dynamic
  covariate schemas;
- encoder/decoder, causal, patch, or direct multi-horizon output structure;
- distribution head, quantile head, deterministic regression head, and sampling
  defaults;
- forecast output shape, horizon indexing, and metric contract.

## Source oracle checkpoints

Capture raw series, observed mask, scaler state, normalized context, lagged
features, time/static covariate tensors, encoder or patch embeddings, one block
output, distribution or quantile parameters, sampled or deterministic forecast,
and final metric inputs.

## Weight conversion

- preserve scaler parameters and whether they are learned, per item, or computed
  from the context window;
- preserve lag ordering and any zero/seasonal padding convention;
- map value, time-feature, static categorical, and projection embeddings
  separately;
- preserve distribution-head parameter order and positivity transforms;
- record whether sampling uses fixed seeds, ancestral sampling, quantile lookup,
  or deterministic mean/median outputs.

## Minimal MLX path

1. Port scaler, observed-mask handling, and lag construction.
2. Port value/time/static feature embeddings.
3. Port one encoder/decoder or patch block with fixed features.
4. Port the distribution, quantile, or regression head.
5. Compare fixed-horizon forecasts and task metrics.
6. Add batching only after single-series parity is stable.

## Parity traps

- leakage from known-future covariates into the past context;
- using prediction window values during scaler fitting;
- off-by-one lag offsets or horizon indexing;
- observed mask inverted or missing values treated as zeros;
- static categorical vocabularies reordered;
- distribution parameters returned in a different order;
- stochastic samples compared instead of deterministic head parameters;
- multivariate target axes swapped with batch or horizon axes.

## Optimization ladder

1. Establish scaler, lag, and covariate parity before optimizing model blocks.
2. Compile stable encoder/decoder or patch blocks for fixed context/prediction
   buckets.
3. Bucket by context length, prediction horizon, and covariate schema to avoid
   retracing.
4. Batch independent series only after scaler state and masks stay per item.
5. Quantize large linear paths only after forecast-error and quantile coverage
   checks.
6. Keep data loading, calendar feature construction, and metric aggregation
   outside hot compiled paths unless they are array-pure and shape-stable.

## Completion gates

- scaler/normalizer parity passes on fixed series with missing values;
- lag and context/prediction split match the source exactly;
- observed-mask and known-future covariate leakage probes pass;
- block outputs and distribution/quantile parameters match source checkpoints;
- deterministic forecast or quantile tensors match before aggregate metrics;
- task metrics such as MAE/RMSE/MAPE/SMAPE/CRPS or quantile loss match the
  source evaluation contract.
