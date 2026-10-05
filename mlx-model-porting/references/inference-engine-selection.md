# Inference engine selection and method transfer

Reviewed 2026-10-05. The canonical [engine inventory](../assets/inference_engines.json)
and [offline advisor](../scripts/inference_engine_advisor.py) hold exact source pins,
review depth, project boundaries and per-method correctness/rollback contracts.
The repository-root `INFERENCE_ENGINE_SURVEY.md` is generated from that inventory.
This is a separately versioned serving survey: it does not silently repin the
historical source registry, modify recorded experiments, or promote speed claims.

## Choose the layer before choosing the brand

MLX-LM and MLX-VLM supply model implementations and generation primitives.
Current MLX-VLM also has continuous batching, prefix/disk caching, hybrid-state
restoration and speculative paths. oMLX and vllm-mlx frequently build on these
libraries: their distinguishing work includes admission, scheduling, persistence,
model residency and API behavior. A comparison with an older unbatched MLX-VLM
version would answer the wrong question.

vLLM Metal is a distinct upstream vLLM hardware plugin; its scheduler/block
manager and native paged variable-length attention are not the same project as
vllm-mlx. Native paths such as Zig mlx-serve, rMLX via Rust/mlx-c, and the Swift
runtime work reduce language/runtime integration overhead but still need full
model and platform qualification. Splash specializes its own Metal execution
around selected target/draft pairs: accepting MLX-format weights does not by
itself make a runtime MLX-based. MTPLX and dflash-mlx concentrate on speculative
execution, target verification and state rollback.

Use exo for multi-device placement research, not as evidence that a single Mac
became faster. llama.cpp, mistral.rs and MLC LLM are non-MLX Metal comparators.
BaseRT's surrounding repository is public but its engine is proprietary.
Flash-MoE is a model-specialized C/Metal SSD-streaming reference with unresolved
root license coverage at the inspected pin. The MLXFast competition engine has
separate organizer boundaries. None is a blanket drop-in architecture generator.

Every statement above is scoped to the immutable README/code locators in the
engine inventory. An upstream feature declaration is not a local execution test.

## Agent decision workflow

1. Run static model/project intake and retain every canonical blocker. A server
   claiming a model family does not override unsafe serialization, custom model
   code, license uncertainty, incomplete weights or an ambiguous hybrid route.
2. Select the workload: interactive text, coding-agent, multi-user throughput,
   multimodal, audio, embeddings, distributed or specialized research. Use the
   advisor's alphabetical shortlist as research candidates, not a ranking.
3. Qualify the exact bundle: target/draft source, tokenizer/template/processor,
   image/audio path, adapter, weight and KV formats, model cache topology, ABI,
   macOS/chip/kernel availability and license notices. Reject unknown capability
   combinations rather than accepting a model-type string as proof.
4. Establish an unoptimized reference and task-quality gate. A new engine can
   be integrated as a separately pinned adapter; do not rewrite known-correct
   model math just to imitate another server's architecture.
5. Profile the user-visible bottleneck, select one method contract, test its
   negative cases, and keep the old path as a selectable fallback. Optimize
   preparation/tuning stages explicitly, not by searching on every request.
6. Publish evidence only for the exact measured context. Advice cannot promote
   itself. Unknown and inconclusive outcomes remain first-class results.

```bash
python3 mlx-model-porting/scripts/inference_engine_advisor.py --workload coding-agent
python3 mlx-model-porting/scripts/inference_engine_advisor.py --workload multimodal
python3 mlx-model-porting/scripts/inference_engine_advisor.py --workload distributed
```

## Priority order for this repository and auto-mlx

**P0: execution and measurement contracts.** Pin a complete runnable environment,
not only a copied interpreter executable. Interpreter relocation, native-library
lookup and ambient site packages are part of correctness. Probe the actual
workload in the intended sandbox. Packaged/precompiled kernels can reduce JIT
requirements, but this does not authorize a writable global compiler cache or
prove that the existing auto-mlx GPU blocker has been resolved.

Report request first-content latency separately from engine-prefill and model
loading. A heartbeat, HTTP header or reasoning-start event is not a content token.
Count physical prefill tokens separately from restored tokens; report the actual
model batch width separately from concurrent HTTP connections. Record errors and
cancellations in latency distributions, not only successful requests.

**P1: resident lifecycle, capabilities, bounded admission and state identity.**
Keep the model loaded when safe, but bound queues and reserve headroom for cache,
activations and competing processes. Model-pool LRU needs pinned-model/lease
awareness. Return overload explicitly. Expose unsupported and unknown API/model
capabilities accurately; a tool parser existing does not prove a model is trained
for tools or that a client round trip works.

**P1/P2: stateful prefix reuse and fair prefill.** Prefer a small qualified hot
cache before adding SSD persistence. Dense KV, sliding windows, recurrent GDN/SSM
state, pooled state and native-MTP history have different restoration rules.
Use immutable snapshots, copy-on-write and leases; partial trimming of ordinary
KV cannot rewind a cumulative recurrent state. Capture at a precise token
boundary and validate cold/warm continuations, including generated-token reuse.
Chunk long prefills so existing decodes progress; per-request sampler, masks,
stop conditions and cancellation must survive batch compaction.

**P2: verified speculation and kernels.** MTP uses a compatible target head;
DFlash/EAGLE/DSpark need their own matching drafts and verification contracts.
Mathematical rejection sampling, greedy verification and numerical equivalence
of batched versus serial target execution are different claims. Optional typical
or alternative acceptance modes must not inherit the exact mode's quality label.
Measure accepted target tokens per verify cost and rollback, not merely draft
acceptance. QMM small-M verification, paged attention, GDN recurrence, MoE gather
and fusion are candidates only after a profile and strict shape/device guards.

**P3: active-expert SSD or multi-Mac execution.** These address capacity and
communication/I/O constraints. KV offload, weight/expert streaming and memory
mapping are distinct mechanisms. SSD stalls, transient buffers, system reserve
and unbounded requests still create OOM risk. Multi-Mac tensor/pipeline placement
must include transport and failure costs, not just aggregate installed RAM.

## Hard negative cases to retain

- Warm logical-prefix throughput compared with cold physically computed prefill.
- Four HTTP clients labelled a four-row compute batch without instrumentation.
- A packed KV codec counted as compressed resident memory while BF16 shadows remain.
- Quantized weights, fewer experts, sparse prefill or compacted history labelled
  lossless because generation still returns plausible text.
- A speculative theorem used as proof of a new target kernel's numerical parity.
- MTP sidecars attached by shape matching without exact trunk/head provenance.
- Snapshot reuse after adapter, tokenizer, template, tenant, draft or media changes.
- CUDA-only kernel evidence promoted as an Apple Silicon implementation.
- Multi-Mac capacity or SSD streaming advertised as universal decode speed.
- Metadata-only Doctor output, source-code tests, or old-revision benchmark
  receipts labelled current model/GPU deployment approval.

## Evidence and update policy

The new engine registry is an independent, explicit source inventory. Every
entry binds repository, commit, file path, fetched-byte digest, review scope,
method candidates, limitations and `local_validation=not-run`. Existing historical
source pins and benchmark receipts are unchanged. The generated report and site
page must match the registry. New methods here are design/research candidates,
not additions to the promoted optimization catalogue.

Before copying third-party implementation code, review that file's license and
NOTICE, preserve required attributions and check model-weight terms separately.
The present update contains original contracts and summaries, not imported kernels.

For speech, carry the same discipline through the frontend, acoustic model,
codec/vocoder and playback queue: first-audio latency, real-time factor defined
as compute seconds / audio seconds, intelligibility and boundary continuity are
required. Text token throughput cannot stand in for audio usability.
