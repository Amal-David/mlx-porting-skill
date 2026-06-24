# Failure atlas

| Symptom | Likely causes | First checks |
|---|---|---|
| Divergence from first layer | preprocessing, embedding scale, dtype, weight map | saved input, embedding weights, token IDs |
| Good full pass, bad incremental decode | cache axis/order, position offset, mask, rotating cache | one-token replay versus full recompute |
| NaNs after several blocks | normalization epsilon, softmax precision, invalid mask, recurrence accumulation | FP32 sensitive path, first NaN checkpoint |
| Correct logits, wrong text | tokenizer/chat template, special IDs, sampler, stop rules | token-by-token IDs and processors |
| Good short context, bad long context | RoPE scaling, cache truncation, integer overflow, mask construction | positions at boundary and cache length |
| MoE quality collapse | routing top-k, normalization, expert ordering, shared expert, gather/scatter | router logits and selected expert IDs |
| Quantized model much slower | unsupported shape/kernel, dequant overhead, small matrices | per-layer profile and unquantized baseline |
| Compile gives no win | retracing, tiny regions, graph breaks, dynamic containers | compile cache/retrace logs and region size |
| Memory grows each step | retained lazy graph, cache copies, Python references, repeated recompilation | explicit eval, object lifetime, active/cache memory |
| Audio clicks at chunks | overlap-add window, state reset, receptive field, phase mismatch | boundary waveform and final flush |
| TTS intelligible but wrong voice | reference preprocessing, speaker encoder, conditioning normalization | source/MLX speaker embeddings |
| Codec reconstruction noisy | codebook order, delay, quantizer distance, decoder padding/layout | codes before waveform and one-codebook test |
| ASR timestamps drift | frontend frame rate, padding, chunk offset, tokenizer timestamps | feature frames and offset arithmetic |
| Conv path unexpectedly slow | layout transposes, unsupported kernel shape, tiny dispatches | operator profile and boundary layout |
| Custom kernel only works on one size | missing bounds, alignment, tile assumptions | odd/nonmultiple and minimal shapes |
| Save/reload changes output | missing state/config, tied weights lost, quant metadata absent | manifest diff and parameter tree |
