window.MLX_PORTING_SITE_DATA = {
  "architectures": {
    "families": [
      {
        "id": "automatic-speech-recognition",
        "label": "Automatic speech recognition",
        "runbook": "references/runbook-asr.md"
      },
      {
        "id": "autoregressive-audio-lm",
        "label": "Autoregressive audio LM and codec-token TTS",
        "runbook": "references/runbook-autoregressive-audio.md"
      },
      {
        "id": "dense-decoder-transformer",
        "label": "Dense decoder Transformer",
        "runbook": "references/runbook-decoder-transformer.md"
      },
      {
        "id": "diffusion-flow",
        "label": "Diffusion and flow models",
        "runbook": "references/runbook-diffusion-flow.md"
      },
      {
        "id": "encoder-decoder-transformer",
        "label": "Encoder-decoder Transformer",
        "runbook": "references/runbook-encoder-decoder.md"
      },
      {
        "id": "encoder-transformer",
        "label": "Encoder Transformer",
        "runbook": "references/runbook-encoder-transformer.md"
      },
      {
        "id": "flow-diffusion-tts",
        "label": "Flow, diffusion, and non-autoregressive TTS",
        "runbook": "references/runbook-flow-tts.md"
      },
      {
        "id": "graph-message-passing",
        "label": "Graph message-passing models",
        "runbook": "references/runbook-graph-message-passing.md"
      },
      {
        "id": "moe-decoder-transformer",
        "label": "Mixture-of-Experts Transformer",
        "runbook": "references/runbook-moe-transformer.md"
      },
      {
        "id": "neural-audio-codec",
        "label": "Neural audio codec and speech tokenizer",
        "runbook": "references/runbook-audio-codec.md"
      },
      {
        "id": "non-generative-cv-backbone",
        "label": "Non-generative CV backbones",
        "runbook": "references/runbook-non-generative-cv.md"
      },
      {
        "id": "separation-enhancement",
        "label": "Audio separation and enhancement",
        "runbook": "references/runbook-separation-enhancement.md"
      },
      {
        "id": "ssm-recurrent-hybrid",
        "label": "State-space, recurrent, and hybrid models",
        "runbook": "references/runbook-ssm-hybrid.md"
      },
      {
        "id": "streaming-speech",
        "label": "Streaming speech and speech-to-speech",
        "runbook": "references/runbook-streaming-speech.md"
      },
      {
        "id": "time-series-forecasting",
        "label": "Time-series forecasting",
        "runbook": "references/runbook-time-series-forecasting.md"
      },
      {
        "id": "vision-language-omni",
        "label": "Vision-language, audio-language, and omni models",
        "runbook": "references/runbook-multimodal-omni.md"
      },
      {
        "id": "vocoder-waveform-decoder",
        "label": "Neural vocoder and waveform decoder",
        "runbook": "references/runbook-vocoder.md"
      }
    ],
    "total": 17
  },
  "benchmarks": {
    "by_classification": {
      "performance_observation": 12,
      "rejected": 1
    },
    "promotion_ready": 0,
    "total": 13
  },
  "effective_claims": {
    "by_state": {
      "withheld": 10
    },
    "total": 10
  },
  "guidance": {
    "by_status": {
      "native-mlx": 7,
      "official-mlx-project": 3,
      "proven-mlx-port": 9,
      "rejected-or-superseded": 2,
      "research-candidate": 7
    },
    "total": 28
  },
  "learning": {
    "checkpoint_nodes": [
      {
        "concept": "Identify artifacts, inputs, outputs, modality, architecture, state, and source format.",
        "evidence_state": "required",
        "id": "inspect",
        "inspect": "Config, class graph, tensor inventory, preprocessing, generation or scheduler code, and custom operations.",
        "outcome": "A pinned semantic inventory",
        "prerequisite": "A pinned local source revision with remote model code disabled.",
        "proof": "The port plan records evidence and uncertainty for every selected family.",
        "title": "Inspect",
        "why_mlx_differs": "MLX route selection depends on the actual computation, not a filename or task label."
      },
      {
        "concept": "Record source tensors, state, and outputs before translating implementation details.",
        "evidence_state": "required",
        "id": "oracle",
        "inspect": "Inputs, RNG, preprocessing, primitive boundaries, repeated blocks, state, logits, and task output.",
        "outcome": "Deterministic source checkpoints",
        "prerequisite": "A safe source runtime and a small representative input.",
        "proof": "Captured artifacts are bounded, named, reproducible, and tied to the source revision.",
        "title": "Capture oracle",
        "why_mlx_differs": "The source oracle is the behavioral contract used to judge the new runtime."
      },
      {
        "concept": "Reproduce source semantics with native MLX modules and operations.",
        "evidence_state": "required",
        "id": "implement",
        "inspect": "Operator semantics, masks, positions, residual order, activation functions, state, and output contracts.",
        "outcome": "A readable eager MLX graph",
        "prerequisite": "Source oracle and architecture route.",
        "proof": "Small synthetic inputs exercise every critical branch before real weights are loaded.",
        "title": "Rebuild",
        "why_mlx_differs": "Lazy execution, layout, state, and numerical roles need explicit MLX choices."
      },
      {
        "concept": "Declare how every source tensor becomes a target parameter.",
        "evidence_state": "required",
        "id": "map",
        "inspect": "Rename, transpose, reshape, split, merge, dtype, destination shape, and source-key coverage.",
        "outcome": "Deterministic source-to-MLX transforms",
        "prerequisite": "Target module tree and a complete source tensor inventory.",
        "proof": "The converter reports complete coverage, expected shapes, and no unexplained tensor drop.",
        "title": "Map weights",
        "why_mlx_differs": "Names, layouts, fused projections, and dtype policies can differ from the source framework."
      },
      {
        "concept": "Compare staged values from primitives through the final task.",
        "evidence_state": "gate",
        "id": "parity",
        "inspect": "Shapes, primitive outputs, blocks, final outputs, cache or recurrent state, and task behavior.",
        "outcome": "The first trustworthy MLX path",
        "prerequisite": "Source oracle, eager implementation, and complete weight map.",
        "proof": "Every rung passes a declared tolerance or exact-match rule; the first divergence is explained.",
        "title": "Prove parity",
        "why_mlx_differs": "Only parity distinguishes a semantic port from a checkpoint that merely loads."
      },
      {
        "concept": "Separate the workload into phases and find the limiting one.",
        "evidence_state": "gate",
        "id": "profile",
        "inspect": "Prefill, decode, preprocessing, repeated steps, memory, streaming, synchronization, and serving behavior.",
        "outcome": "A measured bottleneck",
        "prerequisite": "A parity-passing eager path and representative workload.",
        "proof": "The receipt includes evaluation boundaries, hardware, software, shapes, warm state, and quality.",
        "title": "Profile",
        "why_mlx_differs": "Lazy scheduling and unified memory change where work and synchronization are observed."
      },
      {
        "concept": "Choose a technique that matches the measured bottleneck and architecture.",
        "evidence_state": "gated",
        "id": "optimize",
        "inspect": "Applicability, prerequisite, evidence status, quality gate, benchmark, and rollback.",
        "outcome": "A faster or smaller path with proof",
        "prerequisite": "Parity and a measured bottleneck; this node is gated until both exist.",
        "proof": "Parity and quality remain valid and the representative end-to-end workload improves.",
        "title": "Optimize",
        "why_mlx_differs": "Native operators, compilation, caches, compression, algorithms, and serving each have different contracts."
      },
      {
        "concept": "Package code, provenance, transforms, validation, measurements, limitations, and rollback.",
        "evidence_state": "required",
        "id": "publish",
        "inspect": "Source pin, license, plan, weight map, parity report, benchmark receipts, runbook, and known gaps.",
        "outcome": "An inspectable port packet",
        "prerequisite": "All claims have canonical evidence and unsupported paths remain labeled.",
        "proof": "A clean environment can reproduce the documented route without remote model code.",
        "title": "Publish",
        "why_mlx_differs": "A reusable MLX route needs an honest boundary around model, target, and evidence."
      }
    ],
    "checkpoint_order": [
      "inspect",
      "oracle",
      "implement",
      "map",
      "parity",
      "profile",
      "optimize",
      "publish"
    ],
    "foundations": [
      {
        "common_trap": "Treating unified memory as a promise that every framework bridge is zero-copy.",
        "example": "Create arrays and modules first; call mx.eval only when a dependency, comparison, or measurement needs realized results.",
        "id": "what-is-mlx",
        "mlx_translation": "Build with mx.array and mlx.nn, let operations select device and stream, and evaluate only at deliberate boundaries.",
        "next_step": "Learn what a port must preserve beyond weights.",
        "plain_language": "MLX is an array framework for machine learning on Apple silicon with a NumPy-like API.",
        "proof_check": "You can explain when an MLX expression is built and when its result is actually needed.",
        "pytorch_cuda": "PyTorch projects commonly think in eager tensors plus explicit CPU-to-CUDA movement.",
        "source_ids": [
          "mlx-docs",
          "mlx-doc-quick-start",
          "mlx-doc-unified-memory",
          "mlx-doc-lazy",
          "mlx-doc-streams",
          "mlx-doc-neural-networks",
          "mlx-doc-function-transforms"
        ],
        "title": "What MLX is",
        "why_it_matters": "Its lazy execution and unified-memory model change where porters place evaluation, state, and measurement boundaries."
      },
      {
        "common_trap": "Loading tensors successfully and calling the model ported before output parity exists.",
        "example": "Match tokenizer inputs, one primitive, one block, full logits, stateful decode, and the task result in order.",
        "id": "what-is-porting",
        "mlx_translation": "Rebuild the computation in MLX, transform every weight explicitly, and compare staged outputs against a source oracle.",
        "next_step": "Read the model before choosing an implementation route.",
        "plain_language": "A port reproduces a source model's behavior in a new runtime.",
        "proof_check": "You can name source semantics, weight transforms, state, and outputs that need evidence.",
        "pytorch_cuda": "The source implementation and its outputs are the behavioral contract, not only state_dict keys.",
        "source_ids": [
          "mlx-repo",
          "mlx-examples-repo"
        ],
        "title": "What porting means",
        "why_it_matters": "Checkpoint conversion alone cannot preserve control flow, layouts, masks, cache semantics, preprocessing, or generation behavior."
      },
      {
        "common_trap": "Using modality as the architecture family or treating a composed model as one homogeneous block.",
        "example": "Whisper is speech input and an encoder-decoder architecture; LLaVA composes vision, projection, and language components.",
        "id": "read-the-model",
        "mlx_translation": "Route every component through its native MLX family and record uncertainty instead of forcing one label.",
        "next_step": "Capture a source oracle before rewriting anything.",
        "plain_language": "Inventory the model's artifacts, inputs, outputs, repeated blocks, state, and surrounding pipeline.",
        "proof_check": "The port plan lists source format, modality, architecture components, state, preprocessing, and output contract.",
        "pytorch_cuda": "A Transformers class name may hide a vision tower, audio frontend, decoder, scheduler, or custom cache.",
        "source_ids": [
          "mlx-examples-repo"
        ],
        "title": "Read the model before code",
        "why_it_matters": "Modality, architecture, and workload are separate decisions that select different runbooks and proof gates."
      },
      {
        "common_trap": "Optimizing or compiling before an eager path has parity.",
        "example": "A wrong attention mask should fail at attention output before it is hidden by later residual blocks.",
        "id": "correctness-rail",
        "mlx_translation": "Implement an eager readable MLX path, map weights, then compare primitive, block, output, state, and task rungs.",
        "next_step": "Learn how to debug the first failed rung.",
        "plain_language": "Move from the smallest comparable value to the complete task, stopping at the first divergence.",
        "proof_check": "Every passed rung names inputs, shapes, dtype policy, tolerance, and source artifact.",
        "pytorch_cuda": "Capture deterministic source tensors and state at named checkpoints.",
        "source_ids": [
          "mlx-repo"
        ],
        "title": "The correctness rail",
        "why_it_matters": "A late end-to-end failure gives too little information to locate a semantic mismatch."
      },
      {
        "common_trap": "Loosening tolerances until the failure disappears without explaining the numerical policy.",
        "example": "If projected vision embeddings match but logits do not, inspect token placement and decoder inputs next.",
        "id": "first-divergence",
        "mlx_translation": "Compare the matching MLX primitive, then the enclosing block, output head, recurrent/cache state, and task result.",
        "next_step": "Profile only after the correctness rail is stable.",
        "plain_language": "The earliest failing comparison is the highest-signal place to investigate.",
        "proof_check": "A fix makes the intended rung pass while all earlier rungs remain green.",
        "pytorch_cuda": "Inspect source shapes, layouts, dtypes, masks, position rules, and state mutation at the failing boundary.",
        "source_ids": [
          "mlx-repo"
        ],
        "title": "Debug the first divergence",
        "why_it_matters": "Later errors compound and can make a local layout or state bug look like a global model failure."
      },
      {
        "common_trap": "Timing graph construction while the computation is realized outside the measured region.",
        "example": "Separate prompt prefill from token decode instead of reporting one blended generation time.",
        "id": "profile-bottleneck",
        "mlx_translation": "Place evaluation around measured regions so lazy work is included exactly once and synchronization is intentional.",
        "next_step": "Select a technique from the measured bottleneck branch.",
        "plain_language": "Measure the phase that limits the target workload before choosing an optimization.",
        "proof_check": "The receipt names workload, cold or warm state, evaluation boundaries, hardware, software, and quality gate.",
        "pytorch_cuda": "CUDA intuition about a hot kernel does not prove the same bottleneck exists on MLX and Apple silicon.",
        "source_ids": [
          "mlx-doc-lazy",
          "mlx-doc-compile"
        ],
        "title": "Profile the real bottleneck",
        "why_it_matters": "Prefill, decode, preprocessing, denoising, streaming, memory, and serving can demand different techniques."
      },
      {
        "common_trap": "Publishing a source-reported or local speed range as an effective recommendation.",
        "example": "Compare the same prompt, decode length, cache state, dtype policy, and output-quality gate.",
        "id": "benchmark-honestly",
        "mlx_translation": "Force the intended lazy work inside the timed region and preserve raw evidence plus rollback conditions.",
        "next_step": "Package the port and its proof boundary together.",
        "plain_language": "A benchmark is a scoped observation, not a portable promise.",
        "proof_check": "Another engineer can reproduce the exact workload and see why the number is or is not promotion-ready.",
        "pytorch_cuda": "Framework benchmark habits still need explicit warmup, synchronization, inputs, and quality checks.",
        "source_ids": [
          "mlx-doc-lazy"
        ],
        "title": "Benchmark without fooling yourself",
        "why_it_matters": "Shape, cache state, dtype, model revision, quality, and measurement boundaries can change the result."
      },
      {
        "common_trap": "Promoting indexed research or a passing smoke test into supported guidance.",
        "example": "Label one pinned Qwen checkpoint proven while keeping neighboring Qwen architectures unclaimed.",
        "id": "publish-proof",
        "mlx_translation": "Record source pin, weight map, parity report, benchmark metadata, supported route, and known gaps.",
        "next_step": "Use the field manual to execute the same rail on a real model.",
        "plain_language": "Ship the implementation together with provenance, validation, limitations, and a rollback story.",
        "proof_check": "Every claim points to evidence and every risky change names a rollback condition.",
        "pytorch_cuda": "A converted artifact without its source revision, transforms, and validation cannot be audited.",
        "source_ids": [
          "mlx-repo"
        ],
        "title": "Publish the proof boundary",
        "why_it_matters": "Users need to know exactly which checkpoint, workload, and behavior were reproduced."
      }
    ],
    "glossary": [
      {
        "definition": "Apple's array framework for machine learning on Apple silicon, with lazy evaluation, composable transforms, and unified-memory operation.",
        "term": "MLX"
      },
      {
        "definition": "The simplest readable MLX implementation used to prove semantics before compilation or specialized optimization.",
        "term": "eager graph"
      },
      {
        "definition": "Deterministic tensors, state, and outputs captured from the pinned source implementation for comparison.",
        "term": "source oracle"
      },
      {
        "definition": "Evidence that source and MLX behavior agree under a declared exact or numerical comparison policy.",
        "term": "parity"
      },
      {
        "definition": "The selected architecture-family and capability path that determines runbooks, risks, and proof gates.",
        "term": "route"
      },
      {
        "definition": "Architecture-specific implementation and validation guidance with supported, experimental, and blocked boundaries.",
        "term": "runbook"
      },
      {
        "definition": "A named intermediate tensor, state, or output used to localize correctness across the port.",
        "term": "checkpoint"
      },
      {
        "definition": "The explicit rename, transpose, reshape, split, merge, dtype, and destination contract for source tensors.",
        "term": "weight map"
      },
      {
        "definition": "Stored attention keys and values reused across autoregressive decoding steps.",
        "term": "KV cache"
      },
      {
        "definition": "The repository's structured weight-map and conversion contract with explicit tensor transforms and coverage.",
        "term": "schema-2"
      },
      {
        "definition": "A small representative input and expected behavior used as a stable end-to-end validation target.",
        "term": "golden scenario"
      },
      {
        "definition": "Structured raw evidence for one bounded benchmark or validation run, including its exact context.",
        "term": "receipt"
      },
      {
        "definition": "An evidence state that has satisfied the repository's provenance, workload, quality, attestation, and scope gates.",
        "term": "promotion-ready"
      },
      {
        "definition": "Evidence binding a measurement to its command, dependencies, outputs, and trusted verification boundary.",
        "term": "execution attestation"
      },
      {
        "definition": "The controlled hardware, software, model, workload, and capability context used for recommendation eligibility.",
        "term": "TargetProfile"
      },
      {
        "definition": "The proportion and disposition of source checkpoint keys accounted for by the weight-map contract.",
        "term": "source-key coverage"
      },
      {
        "definition": "The exact model, revision, target, workload, behavior, and evidence scope a claim is allowed to cover.",
        "term": "proof boundary"
      }
    ],
    "guidance_methods": [
      {
        "advisor": {
          "description": "Promising contributor, blog, repository, paper, package, or research-loop learning that is not promotion-ready for supported guidance.",
          "id": "experimental-approach",
          "label": "Experimental approach",
          "requires_user_opt_in": true
        },
        "applies_to": [
          "dense-decoder-transformer",
          "moe-decoder-transformer",
          "long-context"
        ],
        "canonical_source": {
          "claim_types": [],
          "id": "paper-2502-04420",
          "review_depth": "synthesized",
          "role": "papers",
          "support_scope": "unspecified",
          "title": "KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization",
          "url": "https://arxiv.org/abs/2502.04420"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [],
            "id": "paper-2502-04420",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization",
            "url": "https://arxiv.org/abs/2502.04420"
          },
          {
            "claim_types": [],
            "id": "paper-2502-15075",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "More for Keys, Less for Values: Adaptive KV Cache Quantization",
            "url": "https://arxiv.org/abs/2502.15075"
          },
          {
            "claim_types": [],
            "id": "paper-2602-23200",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "InnerQ: Hardware-Aware Tuning-Free Quantization of KV Cache for Large Language Models",
            "url": "https://arxiv.org/abs/2602.23200"
          },
          {
            "claim_types": [],
            "id": "paper-2604-04722",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs",
            "url": "https://arxiv.org/abs/2604.04722"
          },
          {
            "claim_types": [],
            "id": "paper-2606-20474",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "UltraQuant: 4-bit KV Caching for Context-Heavy Agents",
            "url": "https://arxiv.org/abs/2606.20474"
          },
          {
            "claim_types": [],
            "id": "paper-2606-24033",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "RoPE-Aware Bit Allocation for KV-Cache Quantization",
            "url": "https://arxiv.org/abs/2606.24033"
          }
        ],
        "expected_effect": "Targets peak memory, long context. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "state-memory",
        "id": "adaptive-kv-quantization",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "peak-memory",
          "long-context"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Weights or KV state exceed the useful memory budget or create avoidable transfer and cache cost.",
        "proof_gate": "State update, position, reset, isolation, and representative memory accounting are validated.",
        "quality_gate": [
          "matched bit budget comparison",
          "long-context quality"
        ],
        "quality_gated": true,
        "recommendation": "Treat layerwise, key/value, RoPE-aware, and hardware-aware KV bit allocation as experiments until reproduced in MLX.",
        "rollback_conditions": [
          "no quality advantage over uniform KV quantization",
          "decode overhead dominates"
        ],
        "status": "research-candidate",
        "technique_id": "mixed-precision-kv",
        "title": "Layerwise/key-value/adaptive mixed-precision KV",
        "tradeoffs": [
          "Calibration/search complexity",
          "possible decode overhead",
          "hardware-specific assumptions may not transfer to Apple Silicon."
        ],
        "validation_gates": [
          "MLX implementation",
          "matched bit budget comparison",
          "long-context quality",
          "bytes/token/layer",
          "decode latency",
          "fallback mode"
        ]
      },
      {
        "advisor": {
          "description": "Safe to try after parity, but no speedup or memory number may be claimed until measured on target hardware and workload.",
          "id": "benchmark-required",
          "label": "Benchmark required",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "autoregressive-audio-lm",
          "tts",
          "voice-cloning"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation",
            "performance",
            "audio_quality"
          ],
          "id": "mlx-audio-qwen3-tts-docs",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "third_party_pinned",
          "title": "MLX-Audio Qwen3-TTS model guide",
          "url": "https://github.com/Blaizzy/mlx-audio/blob/a7ef98604cfd752e9e5c9011bcee8ec8c67228be/docs/models/tts/qwen3-tts.md"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation",
              "performance",
              "audio_quality"
            ],
            "id": "mlx-audio-qwen3-tts-docs",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "third_party_pinned",
            "title": "MLX-Audio Qwen3-TTS model guide",
            "url": "https://github.com/Blaizzy/mlx-audio/blob/a7ef98604cfd752e9e5c9011bcee8ec8c67228be/docs/models/tts/qwen3-tts.md"
          },
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "mlx-audio-release-044",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "MLX-Audio v0.4.4 release",
            "url": "https://github.com/Blaizzy/mlx-audio/releases/tag/v0.4.4"
          }
        ],
        "expected_effect": "Targets first audio latency, rtf. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "audio-reference-conditioning-cache",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "first-audio-latency",
          "rtf"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Cache repeated reference/ICL conditioning only with model/codec/quantization/speaker fingerprint keys.",
        "rollback_conditions": [
          "speaker identity or prosody drifts",
          "privacy namespace missing",
          "memory pressure causes unacceptable eviction behavior"
        ],
        "status": "proven-mlx-port",
        "technique_id": "audio-reference-conditioning-cache",
        "title": "Audio reference / ICL conditioning cache",
        "tradeoffs": [
          "Privacy-sensitive speaker/reference material can leak through cache sharing.",
          "Wrong keying can preserve stale voice/prosody state."
        ],
        "validation_gates": [
          "cold/warm parity",
          "speaker similarity/prosody check",
          "cache hit/miss accounting",
          "tenant isolation",
          "memory pressure behavior"
        ]
      },
      {
        "advisor": {
          "description": "Safe to try after parity, but no speedup or memory number may be claimed until measured on target hardware and workload.",
          "id": "benchmark-required",
          "label": "Benchmark required",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "tts",
          "streaming-speech",
          "audio-codec",
          "vocoder"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation"
          ],
          "id": "mlx-audio-release-044",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "third_party_pinned",
          "title": "MLX-Audio v0.4.4 release",
          "url": "https://github.com/Blaizzy/mlx-audio/releases/tag/v0.4.4"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "mlx-audio-release-044",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "MLX-Audio v0.4.4 release",
            "url": "https://github.com/Blaizzy/mlx-audio/releases/tag/v0.4.4"
          },
          {
            "claim_types": [],
            "id": "mlx-audio-codecs",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX-Audio codec implementations",
            "url": "https://github.com/Blaizzy/mlx-audio/tree/412cf7cd381c2a3f6a8189af04a95af24cb415b6/mlx_audio/codec/models"
          },
          {
            "claim_types": [],
            "id": "mlx-audio-models",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX-Audio model implementations",
            "url": "https://github.com/Blaizzy/mlx-audio/tree/412cf7cd381c2a3f6a8189af04a95af24cb415b6/mlx_audio"
          },
          {
            "claim_types": [],
            "id": "mlx-audio-repo",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX-Audio repository",
            "url": "https://github.com/Blaizzy/mlx-audio/tree/412cf7cd381c2a3f6a8189af04a95af24cb415b6"
          },
          {
            "claim_types": [],
            "id": "paper-2505-15380",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding",
            "url": "https://arxiv.org/abs/2505.15380"
          }
        ],
        "expected_effect": "Targets first audio latency, rtf, streaming quality. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "audio-streaming-and-cache",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "first-audio-latency",
          "rtf",
          "streaming-quality"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [
          "chunk-boundary continuity",
          "speaker similarity/intelligibility"
        ],
        "quality_gated": true,
        "recommendation": "Use cache-aware streaming, overlap-add, and chunked codec/vocoder execution only when chunk boundaries preserve audio quality.",
        "rollback_conditions": [
          "audible discontinuity",
          "quality metric regression",
          "RTF not improved"
        ],
        "status": "proven-mlx-port",
        "technique_id": "audio-overlap-add",
        "title": "Overlap-add mid-generation streaming",
        "tradeoffs": [
          "Chunk seams, speaker drift, codec delay, and ASR intelligibility can regress even when latency improves."
        ],
        "validation_gates": [
          "first-audio latency",
          "RTF",
          "chunk-boundary continuity",
          "speaker similarity/intelligibility",
          "peak memory"
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "linear-heavy-models",
          "dense-decoder-transformer"
        ],
        "canonical_source": {
          "claim_types": [
            "api_support"
          ],
          "id": "mlx-docs",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "official_mlx",
          "title": "MLX documentation",
          "url": "https://ml-explore.github.io/mlx/build/html/index.html"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-docs",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX documentation",
            "url": "https://ml-explore.github.io/mlx/build/html/index.html"
          },
          {
            "claim_types": [],
            "id": "mlx-repo",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX array framework repository",
            "url": "https://github.com/ml-explore/mlx/tree/96296e9c3075a2389bc5c0f078bf01b5aa377cd9"
          }
        ],
        "expected_effect": "Targets model size, latency. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "compression",
        "id": "bf16-weight-cast",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "model-size",
          "latency"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Weight, state, or token representation dominates memory or bandwidth after baseline parity.",
        "proof_gate": "The compressed artifact passes task-specific quality and memory gates on the representative workload.",
        "quality_gate": [
          "task-specific quality against F32"
        ],
        "quality_gated": true,
        "recommendation": "Cast a controlled F32 MLX port to BF16 only when the exact target workload retains its declared quality window; keep timing results observation-only until every promotion gate, including external signed attestation, passes.",
        "rollback_conditions": [
          "quality output leaves the declared window",
          "external attestation signature is missing or invalid",
          "wall-time gain falls within noise",
          "target model, workload, or software fingerprint changes"
        ],
        "status": "native-mlx",
        "technique_id": "weight-bf16-cast",
        "title": "BF16 weight casting for MLX inference",
        "tradeoffs": [
          "Reduced precision may change generated outputs beyond the validated six-token window.",
          "The observed wall-time ratio includes model hashing and evidence capture, so it must not be presented as pure decode speed."
        ],
        "validation_gates": [
          "deterministic F32-to-BF16 conversion manifest",
          "task-specific quality against F32",
          "isolated parent-measured end-to-end benchmark",
          "external signature over commit/tree, challenge, reviewed dependencies, raw output, policy, and timing, verified against an out-of-repository trust anchor",
          "stability and noise-floor clearance"
        ]
      },
      {
        "advisor": {
          "description": "Safe to try after parity, but no speedup or memory number may be claimed until measured on target hardware and workload.",
          "id": "benchmark-required",
          "label": "Benchmark required",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "diffusion-flow",
          "flow-diffusion-tts",
          "vision-language-omni"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation",
            "memory"
          ],
          "id": "dgrauet-ltx2-block-streaming-source",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "third_party_pinned",
          "title": "ltx-2-mlx BlockStreamer implementation",
          "url": "https://github.com/dgrauet/ltx-2-mlx/blob/3f5897a9582b5c14379c8f88216ae0fd6e55741d/packages/ltx-core-mlx/src/ltx_core_mlx/loader/block_streaming.py"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation",
              "memory"
            ],
            "id": "dgrauet-ltx2-block-streaming-source",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "ltx-2-mlx BlockStreamer implementation",
            "url": "https://github.com/dgrauet/ltx-2-mlx/blob/3f5897a9582b5c14379c8f88216ae0fd6e55741d/packages/ltx-core-mlx/src/ltx_core_mlx/loader/block_streaming.py"
          },
          {
            "claim_types": [],
            "id": "dgrauet-ltx2-block-streaming-tests",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "ltx-2-mlx block streaming tests",
            "url": "https://github.com/dgrauet/ltx-2-mlx/blob/3f5897a9582b5c14379c8f88216ae0fd6e55741d/tests/test_block_streaming.py"
          },
          {
            "claim_types": [],
            "id": "dgrauet-ltx2-regression-report",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "ltx-2-mlx v0.11.0 regression report",
            "url": "https://github.com/dgrauet/ltx-2-mlx/blob/3f5897a9582b5c14379c8f88216ae0fd6e55741d/docs/REGRESSION_TESTS_v0.11.0.md"
          }
        ],
        "expected_effect": "Targets peak memory. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "state-memory",
        "id": "block-weight-streaming",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "peak-memory"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Weights or KV state exceed the useful memory budget or create avoidable transfer and cache cost.",
        "proof_gate": "State update, position, reset, isolation, and representative memory accounting are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "For repeated-block models that exceed unified memory, stream block weights from inspectable safetensors into a shared block only after eager parity is established.",
        "rollback_conditions": [
          "streamed outputs drift",
          "key coverage is incomplete",
          "eviction reloads stale weights or adapters",
          "measured memory win is absent",
          "latency becomes unacceptable"
        ],
        "status": "proven-mlx-port",
        "technique_id": "block-weight-streaming",
        "title": "Repeated-block weight streaming",
        "tradeoffs": [
          "Page faults and disk reads can erase the memory win.",
          "Requires exact per-block key maps, strict loading, eviction tests, and an eager fallback.",
          "Risky for autoregressive decode if every token reloads every block."
        ],
        "validation_gates": [
          "block key coverage",
          "streamed-vs-eager primitive and full-model parity",
          "eviction/reload consistency",
          "peak and steady memory measurement",
          "wall-time benchmark",
          "eager fallback path"
        ]
      },
      {
        "advisor": {
          "description": "Promising contributor, blog, repository, paper, package, or research-loop learning that is not promotion-ready for supported guidance.",
          "id": "experimental-approach",
          "label": "Experimental approach",
          "requires_user_opt_in": true
        },
        "applies_to": [
          "dense-decoder-transformer",
          "moe-decoder-transformer",
          "vision-language-omni",
          "server",
          "rag",
          "agents"
        ],
        "canonical_source": {
          "claim_types": [],
          "id": "paper-2606-21842",
          "review_depth": "screened",
          "role": "papers",
          "support_scope": "unspecified",
          "title": "Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG",
          "url": "https://arxiv.org/abs/2606.21842"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [],
            "id": "paper-2606-21842",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG",
            "url": "https://arxiv.org/abs/2606.21842"
          },
          {
            "claim_types": [],
            "id": "vllm-blog-anatomy",
            "review_depth": "screened",
            "role": "technical_blogs",
            "support_scope": "unspecified",
            "title": "Inside vLLM: Anatomy of a High-Throughput LLM Inference System",
            "url": "https://vllm.ai/blog/2025-09-05-anatomy-of-vllm"
          },
          {
            "claim_types": [],
            "id": "vllm-mlx-repo",
            "review_depth": "screened",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "vLLM-MLX serving repository",
            "url": "https://github.com/waybarrios/vllm-mlx"
          }
        ],
        "expected_effect": "Targets serving safety. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "cache-privacy-and-isolation",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "serving-safety"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Require namespace isolation, opt-out, padding/constant-time policy review, and leakage tests before cross-user cache reuse.",
        "rollback_conditions": [
          "unbounded cross-tenant reuse",
          "detectable side channel",
          "missing deletion/persistence policy"
        ],
        "status": "research-candidate",
        "technique_id": "cache-privacy-review",
        "title": "Cache privacy and tenant isolation review",
        "tradeoffs": [
          "Isolation and padding can reduce cache hit rate or add latency, but avoid privacy leaks."
        ],
        "validation_gates": [
          "tenant namespace tests",
          "timing/leakage review",
          "cache persistence policy",
          "opt-out path"
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "all"
        ],
        "canonical_source": {
          "claim_types": [
            "api_support"
          ],
          "id": "mlx-doc-compile",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "official_mlx",
          "title": "MLX compilation documentation",
          "url": "https://ml-explore.github.io/mlx/build/html/usage/compile.html"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-doc-compile",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX compilation documentation",
            "url": "https://ml-explore.github.io/mlx/build/html/usage/compile.html"
          },
          {
            "claim_types": [],
            "id": "mlx-repo",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX array framework repository",
            "url": "https://github.com/ml-explore/mlx/tree/96296e9c3075a2389bc5c0f078bf01b5aa377cd9"
          },
          {
            "claim_types": [],
            "id": "apple-wwdc25-mlx",
            "review_depth": "screened",
            "role": "technical_blogs",
            "support_scope": "unspecified",
            "title": "Get started with MLX for Apple silicon",
            "url": "https://developer.apple.com/videos/play/wwdc2025/315/"
          }
        ],
        "expected_effect": "Targets latency, throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "native-operators-compilation",
        "id": "compile-stable-region",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "latency",
          "throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: A compatible repeated operator or stable graph region dominates runtime.",
        "proof_gate": "Eager parity exists and the native or compiled path matches it across representative shapes and state.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Compile pure repeated regions after parity is established and shapes/state containers are stable.",
        "rollback_conditions": [
          "hidden retracing",
          "cold latency is unacceptable",
          "warm end-to-end win is absent"
        ],
        "status": "native-mlx",
        "technique_id": "stable-region-compile",
        "title": "Compile stable repeated regions",
        "tradeoffs": [
          "Dynamic Python control, container mutation, dtype changes, or rank changes can retrace.",
          "Cold and warm metrics must be separated."
        ],
        "validation_gates": [
          "compile count or cold/warm timing captured",
          "parity before and after compile",
          "multiple legal lengths tested when shapeless compile is used"
        ]
      },
      {
        "advisor": {
          "description": "Incompatible, unsafe, contradicted, license-blocked, CUDA-only, or superseded.",
          "id": "rejected-do-not-use",
          "label": "Rejected / do not use",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "vision-language-omni",
          "vlm"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation",
            "serving_semantics"
          ],
          "id": "mlx-vlm-readme",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "third_party_pinned",
          "title": "MLX-VLM serving, caching, and speculative decoding documentation",
          "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/README.md"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation",
              "serving_semantics"
            ],
            "id": "mlx-vlm-readme",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "MLX-VLM serving, caching, and speculative decoding documentation",
            "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/README.md"
          },
          {
            "claim_types": [
              "performance",
              "serving_semantics"
            ],
            "id": "paper-2601-19139",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "paper_only",
            "title": "Native LLM and MLLM Inference at Scale on Apple Silicon",
            "url": "https://arxiv.org/abs/2601.19139"
          },
          {
            "claim_types": [],
            "id": "vllm-mlx-repo",
            "review_depth": "screened",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "vLLM-MLX serving repository",
            "url": "https://github.com/waybarrios/vllm-mlx"
          }
        ],
        "expected_effect": "Targets ttft, throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "content-prefix-cache-vlm",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "ttft",
          "throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Superseded by multimodal-content-prefix-cache; do not count this alias as a separate optimization or evidence line.",
        "rollback_conditions": [
          "false cache hit",
          "poor hit rate",
          "privacy/storage policy failure"
        ],
        "status": "rejected-or-superseded",
        "technique_id": "automatic-prefix-cache",
        "title": "Block-level automatic prefix caching",
        "tradeoffs": [
          "Hashing/normalization must not collapse distinct inputs.",
          "Cache memory grows with repeated media and privacy policy matters."
        ],
        "validation_gates": [
          "same-image/different-encoding hit tests",
          "near-duplicate miss tests",
          "processor revision key tests",
          "TTFT and memory distribution"
        ]
      },
      {
        "advisor": {
          "description": "Safe to try after parity, but no speedup or memory number may be claimed until measured on target hardware and workload.",
          "id": "benchmark-required",
          "label": "Benchmark required",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "server",
          "autoregressive-transformer",
          "vision-language-omni"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation",
            "serving_semantics"
          ],
          "id": "mlx-vlm-readme",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "third_party_pinned",
          "title": "MLX-VLM serving, caching, and speculative decoding documentation",
          "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/README.md"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation",
              "serving_semantics"
            ],
            "id": "mlx-vlm-readme",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "MLX-VLM serving, caching, and speculative decoding documentation",
            "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/README.md"
          },
          {
            "claim_types": [],
            "id": "orca-osdi22",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Orca: A Distributed Serving System for Transformer-Based Generative Models",
            "url": "https://www.usenix.org/conference/osdi22/presentation/yu"
          },
          {
            "claim_types": [],
            "id": "paper-2309-06180",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention",
            "url": "https://arxiv.org/abs/2309.06180"
          },
          {
            "claim_types": [
              "performance",
              "serving_semantics"
            ],
            "id": "paper-2601-19139",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "paper_only",
            "title": "Native LLM and MLLM Inference at Scale on Apple Silicon",
            "url": "https://arxiv.org/abs/2601.19139"
          },
          {
            "claim_types": [],
            "id": "vllm-blog-anatomy",
            "review_depth": "screened",
            "role": "technical_blogs",
            "support_scope": "unspecified",
            "title": "Inside vLLM: Anatomy of a High-Throughput LLM Inference System",
            "url": "https://vllm.ai/blog/2025-09-05-anatomy-of-vllm"
          },
          {
            "claim_types": [],
            "id": "vllm-mlx-repo",
            "review_depth": "screened",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "vLLM-MLX serving repository",
            "url": "https://github.com/waybarrios/vllm-mlx"
          }
        ],
        "expected_effect": "Targets throughput, tail latency, concurrency. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "continuous-batching-serving",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "throughput",
          "tail-latency",
          "concurrency"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Use only for concurrent serving where scheduler complexity is justified.",
        "rollback_conditions": [
          "tail latency regression",
          "state corruption",
          "scheduler overhead dominates"
        ],
        "status": "proven-mlx-port",
        "technique_id": "continuous-batching",
        "title": "Continuous batching",
        "tradeoffs": [
          "Can hurt isolated single-user latency.",
          "Requires robust cancellation, compaction, cache ownership, and per-request sampling."
        ],
        "validation_gates": [
          "concurrency=1 and many",
          "mixed prompt/output lengths",
          "cancel/error tests",
          "P50/P95/P99 latency",
          "throughput and memory distributions"
        ]
      },
      {
        "advisor": {
          "description": "Incompatible, unsafe, contradicted, license-blocked, CUDA-only, or superseded.",
          "id": "rejected-do-not-use",
          "label": "Rejected / do not use",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "dense-decoder-transformer",
          "moe-decoder-transformer"
        ],
        "canonical_source": {
          "claim_types": [
            "api_support"
          ],
          "id": "mlx-doc-compile",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "official_mlx",
          "title": "MLX compilation documentation",
          "url": "https://ml-explore.github.io/mlx/build/html/usage/compile.html"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-doc-compile",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX compilation documentation",
            "url": "https://ml-explore.github.io/mlx/build/html/usage/compile.html"
          },
          {
            "claim_types": [
              "api_support",
              "risk_or_negative"
            ],
            "id": "pytorch-doc-cuda-graphs",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "context_only",
            "title": "PyTorch CUDA Graphs semantics",
            "url": "https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs"
          }
        ],
        "expected_effect": "Targets decode latency. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "custom-backend",
        "id": "cuda-graphs-decode-capture",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "decode-latency"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: A proven hot operation has no adequate native MLX path after safer options are exhausted.",
        "proof_gate": "A readable fallback passes parity and the custom path is benchmarked over representative shapes and devices.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Do not port. CUDA Graphs capture/replay is NVIDIA-specific; the portable MLX analog is mx.compile of the single decode step, which this skill already recommends.",
        "rollback_conditions": [
          "Always: this method is rejected for MLX and must not be adopted."
        ],
        "status": "rejected-or-superseded",
        "technique_id": "cuda-graphs-decode-capture",
        "title": "CUDA Graphs decode-loop capture",
        "tradeoffs": [
          "Effort spent replicating a CUDA-only launch-capture mechanism instead of using mx.compile.",
          "Microbenchmark folklore does not establish a Metal/MLX win."
        ],
        "validation_gates": [
          "Not applicable on Metal/MLX; use the mx.compile stable-region gate instead."
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "autoregressive-decoder",
          "dense-decoder-transformer",
          "moe-decoder-transformer"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation"
          ],
          "id": "mlx-lm-speculative",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "official_mlx_project",
          "title": "MLX-LM speculative generation implementation",
          "url": "https://github.com/ml-explore/mlx-lm/blob/2c008fd0252b2c569227d12568356ab88ab0560a/mlx_lm/generate.py"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "mlx-lm-speculative",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "official_mlx_project",
            "title": "MLX-LM speculative generation implementation",
            "url": "https://github.com/ml-explore/mlx-lm/blob/2c008fd0252b2c569227d12568356ab88ab0560a/mlx_lm/generate.py"
          },
          {
            "claim_types": [],
            "id": "paper-2211-17192",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Fast Inference from Transformers via Speculative Decoding",
            "url": "https://arxiv.org/abs/2211.17192"
          },
          {
            "claim_types": [],
            "id": "paper-2302-01318",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Accelerating Large Language Model Decoding with Speculative Sampling",
            "url": "https://arxiv.org/abs/2302.01318"
          }
        ],
        "expected_effect": "Targets decode latency, throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "inference-algorithms",
        "id": "draft-model-speculation",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "decode-latency",
          "throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Autoregressive target-model calls dominate after the underlying decoder is correct.",
        "proof_gate": "Acceptance semantics, token distribution or exact mode, state, and task quality match the declared policy.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Try only when a compatible smaller draft model shares tokenizer/vocabulary and acceptance is high enough to offset draft cost.",
        "rollback_conditions": [
          "acceptance too low",
          "memory budget exceeded",
          "quality/distribution mismatch"
        ],
        "status": "official-mlx-project",
        "technique_id": "classic-speculative",
        "title": "Independent draft-model speculative decoding",
        "tradeoffs": [
          "Extra model memory",
          "low acceptance can slow generation",
          "sampling distribution must remain correct."
        ],
        "validation_gates": [
          "lossless distribution or quality-preserving test",
          "acceptance length",
          "target verification cost",
          "TTFT/decode benchmark"
        ]
      },
      {
        "advisor": {
          "description": "Promising contributor, blog, repository, paper, package, or research-loop learning that is not promotion-ready for supported guidance.",
          "id": "experimental-approach",
          "label": "Experimental approach",
          "requires_user_opt_in": true
        },
        "applies_to": [
          "decoder-transformer",
          "vision-language-omni"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation",
            "serving_semantics"
          ],
          "id": "mlx-vlm-readme",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "third_party_pinned",
          "title": "MLX-VLM serving, caching, and speculative decoding documentation",
          "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/README.md"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation",
              "serving_semantics"
            ],
            "id": "mlx-vlm-readme",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "MLX-VLM serving, caching, and speculative decoding documentation",
            "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/README.md"
          },
          {
            "claim_types": [],
            "id": "paper-2401-10774",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads",
            "url": "https://arxiv.org/abs/2401.10774"
          },
          {
            "claim_types": [],
            "id": "paper-2401-15077",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty",
            "url": "https://arxiv.org/abs/2401.15077"
          },
          {
            "claim_types": [],
            "id": "paper-2406-16858",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees",
            "url": "https://arxiv.org/abs/2406.16858"
          },
          {
            "claim_types": [],
            "id": "paper-2412-19437",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "DeepSeek-V3 Technical Report",
            "url": "https://arxiv.org/abs/2412.19437"
          },
          {
            "claim_types": [],
            "id": "vllm-mlx-repo",
            "review_depth": "screened",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "vLLM-MLX serving repository",
            "url": "https://github.com/waybarrios/vllm-mlx"
          }
        ],
        "expected_effect": "Targets decode latency, throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "inference-algorithms",
        "id": "eagle-medusa-mtp-drafters",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "decode-latency",
          "throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Autoregressive target-model calls dominate after the underlying decoder is correct.",
        "proof_gate": "Acceptance semantics, token distribution or exact mode, state, and task quality match the declared policy.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Use only when compatible trained drafter heads/artifacts and verification code exist for the target architecture.",
        "rollback_conditions": [
          "incompatible features",
          "low acceptance",
          "verification overhead dominates"
        ],
        "status": "research-candidate",
        "technique_id": "eagle-family",
        "title": "EAGLE/EAGLE3 feature-level speculation",
        "tradeoffs": [
          "Extra weights/training",
          "tree verification complexity",
          "quantized targets can change cost balance."
        ],
        "validation_gates": [
          "drafter compatibility",
          "acceptance/tree stats",
          "distribution or task quality",
          "end-to-end benchmark"
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "dense-decoder-transformer",
          "moe-decoder-transformer",
          "encoder-transformer",
          "encoder-decoder-transformer",
          "vision-language-omni"
        ],
        "canonical_source": {
          "claim_types": [
            "api_support"
          ],
          "id": "mlx-doc-fast-sdpa",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "official_mlx",
          "title": "MLX fast scaled dot product attention API",
          "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-doc-fast-sdpa",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX fast scaled dot product attention API",
            "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html"
          },
          {
            "claim_types": [],
            "id": "mlx-repo",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX array framework repository",
            "url": "https://github.com/ml-explore/mlx/tree/96296e9c3075a2389bc5c0f078bf01b5aa377cd9"
          },
          {
            "claim_types": [],
            "id": "paper-2205-14135",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
            "url": "https://arxiv.org/abs/2205.14135"
          },
          {
            "claim_types": [],
            "id": "paper-2307-08691",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
            "url": "https://arxiv.org/abs/2307.08691"
          },
          {
            "claim_types": [],
            "id": "paper-2407-08608",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision",
            "url": "https://arxiv.org/abs/2407.08608"
          }
        ],
        "expected_effect": "Targets prefill latency, decode latency, throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "native-operators-compilation",
        "id": "fast-sdpa",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "prefill-latency",
          "decode-latency",
          "throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: A compatible repeated operator or stable graph region dominates runtime.",
        "proof_gate": "Eager parity exists and the native or compiled path matches it across representative shapes and state.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Try MLX fast SDPA when Q/K/V layout, GQA/MQA grouping, mask, scale, causal alignment, sinks, and dtype semantics match the source.",
        "rollback_conditions": [
          "Any semantic mismatch",
          "no end-to-end latency or throughput win"
        ],
        "status": "native-mlx",
        "technique_id": "fast-sdpa",
        "title": "MLX fast scaled-dot-product attention",
        "tradeoffs": [
          "May be invalid for custom bias, position, mask, or attention-sink semantics.",
          "GQA/MQA K/V must not be pre-tiled unless the source semantics require it."
        ],
        "validation_gates": [
          "Prefill tensor parity",
          "single-token decode parity",
          "mask broadcast check against [B, N, T_q, T_kv]",
          "end-to-end benchmark across target lengths"
        ]
      },
      {
        "advisor": {
          "description": "Promising contributor, blog, repository, paper, package, or research-loop learning that is not promotion-ready for supported guidance.",
          "id": "experimental-approach",
          "label": "Experimental approach",
          "requires_user_opt_in": true
        },
        "applies_to": [
          "audio-lm",
          "tts",
          "stt",
          "speech-to-speech"
        ],
        "canonical_source": {
          "claim_types": [],
          "id": "mlx-audio-streaming-guide",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "unspecified",
          "title": "MLX-Audio streaming audio guide",
          "url": "https://blaizzy.github.io/mlx-audio/guides/streaming/"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [],
            "id": "mlx-audio-streaming-guide",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "unspecified",
            "title": "MLX-Audio streaming audio guide",
            "url": "https://blaizzy.github.io/mlx-audio/guides/streaming/"
          },
          {
            "claim_types": [],
            "id": "paper-2601-15621",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Qwen3-TTS Technical Report",
            "url": "https://arxiv.org/abs/2601.15621"
          }
        ],
        "expected_effect": "Targets first audio latency, rtf, peak memory. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "generic-audio-prefix-cache",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "first-audio-latency",
          "rtf",
          "peak-memory"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Keep generic audio prefix cache as a research experiment until a model-specific MLX-Audio cache path exists.",
        "rollback_conditions": [
          "state mismatch",
          "quality regression",
          "no audio-specific cache hit workload"
        ],
        "status": "research-candidate",
        "technique_id": "generic-audio-prefix-cache",
        "title": "Generic audio prefix cache",
        "tradeoffs": [
          "Audio state includes codec, speaker, chunk, and flush semantics that text caches do not cover.",
          "Incorrect reuse can corrupt audio continuity or transcripts."
        ],
        "validation_gates": [
          "model-specific cache path",
          "offline-vs-streaming parity",
          "WER/CER or speaker similarity",
          "first-audio/RTF/memory benchmark"
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "all"
        ],
        "canonical_source": {
          "claim_types": [
            "api_support"
          ],
          "id": "mlx-doc-lazy",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "official_mlx",
          "title": "MLX lazy evaluation documentation",
          "url": "https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-doc-lazy",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX lazy evaluation documentation",
            "url": "https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html"
          },
          {
            "claim_types": [],
            "id": "mlx-repo",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX array framework repository",
            "url": "https://github.com/ml-explore/mlx/tree/96296e9c3075a2389bc5c0f078bf01b5aa377cd9"
          },
          {
            "claim_types": [],
            "id": "apple-mlx-m5-blog",
            "review_depth": "screened",
            "role": "technical_blogs",
            "support_scope": "unspecified",
            "title": "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU",
            "url": "https://machinelearning.apple.com/research/exploring-llms-mlx-m5"
          },
          {
            "claim_types": [],
            "id": "apple-wwdc25-mlx",
            "review_depth": "screened",
            "role": "technical_blogs",
            "support_scope": "unspecified",
            "title": "Get started with MLX for Apple silicon",
            "url": "https://developer.apple.com/videos/play/wwdc2025/315/"
          }
        ],
        "expected_effect": "Targets latency, throughput, peak memory. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "evaluation-scheduling",
        "id": "lazy-eval-boundaries",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "latency",
          "throughput",
          "peak-memory"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Lazy graph lifetime, synchronization, or misplaced evaluation dominates the measured phase.",
        "proof_gate": "Parity passes with identical state and the measured region realizes the intended work exactly once.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Move host synchronization to intentional boundaries and avoid accidental NumPy conversion, printing, or scalar item extraction in hot loops.",
        "rollback_conditions": [
          "memory blow-up",
          "no profile improvement"
        ],
        "status": "native-mlx",
        "technique_id": "lazy-eval-boundaries",
        "title": "Deliberate lazy evaluation boundaries",
        "tradeoffs": [
          "Too few evaluations can increase live memory.",
          "Too many evaluations serialize the pipeline."
        ],
        "validation_gates": [
          "same outputs",
          "bounded graph growth",
          "dispatch/synchronization count or timing improvement",
          "peak memory recorded"
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "moe-decoder-transformer"
        ],
        "canonical_source": {
          "claim_types": [
            "api_support"
          ],
          "id": "mlx-doc-gather-mm",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "official_mlx",
          "title": "MLX gather matmul API",
          "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-doc-gather-mm",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX gather matmul API",
            "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html"
          },
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-doc-gather-qmm",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX gather quantized matmul API",
            "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html"
          },
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "mlx-lm-switch-layers",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "official_mlx_project",
            "title": "MLX-LM Switch/MoE layers implementation",
            "url": "https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/models/switch_layers.py"
          },
          {
            "claim_types": [],
            "id": "paper-2405-04434",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
            "url": "https://arxiv.org/abs/2405.04434"
          },
          {
            "claim_types": [],
            "id": "paper-2401-04088",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Mixtral of Experts",
            "url": "https://arxiv.org/abs/2401.04088"
          }
        ],
        "expected_effect": "Targets decode throughput, peak memory. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "compression",
        "id": "moe-expert-dispatch-and-quantization",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "decode-throughput",
          "peak-memory"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Weight, state, or token representation dominates memory or bandwidth after baseline parity.",
        "proof_gate": "The compressed artifact passes task-specific quality and memory gates on the representative workload.",
        "quality_gate": [
          "quality against the unquantized expert baseline"
        ],
        "quality_gated": true,
        "recommendation": "After a loop oracle passes, move to MLX gather_mm/gather_qmm expert dispatch and quantize experts only after sensitivity tests.",
        "rollback_conditions": [
          "rare-expert quality regression",
          "small-group latency worsens",
          "host extraction of routing IDs appears in hot path"
        ],
        "status": "native-mlx",
        "technique_id": "gather-segmented-mm",
        "title": "Gather matmul for indexed expert projections",
        "tradeoffs": [
          "Small expert groups can underutilize kernels.",
          "Router/gate/norm quantization is risky and should be excluded first."
        ],
        "validation_gates": [
          "loop oracle parity",
          "sorted/unsorted duplicate tests",
          "tokens_per_expert histogram",
          "quality and peak-memory benchmark"
        ]
      },
      {
        "advisor": {
          "description": "Promising contributor, blog, repository, paper, package, or research-loop learning that is not promotion-ready for supported guidance.",
          "id": "experimental-approach",
          "label": "Experimental approach",
          "requires_user_opt_in": true
        },
        "applies_to": [
          "moe-decoder-transformer"
        ],
        "canonical_source": {
          "claim_types": [],
          "id": "mlx-lm-issue-956",
          "review_depth": "synthesized",
          "role": "issues",
          "support_scope": "unspecified",
          "title": "MLX-LM MoE gate/up fusion proposal",
          "url": "https://github.com/ml-explore/mlx-lm/issues/956"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [],
            "id": "mlx-lm-issue-956",
            "review_depth": "synthesized",
            "role": "issues",
            "support_scope": "unspecified",
            "title": "MLX-LM MoE gate/up fusion proposal",
            "url": "https://github.com/ml-explore/mlx-lm/issues/956"
          },
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "mlx-lm-switch-layers",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "official_mlx_project",
            "title": "MLX-LM Switch/MoE layers implementation",
            "url": "https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/models/switch_layers.py"
          }
        ],
        "expected_effect": "Targets decode throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "custom-backend",
        "id": "moe-gate-up-fusion",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "decode-throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: A proven hot operation has no adequate native MLX path after safer options are exhausted.",
        "proof_gate": "A readable fallback passes parity and the custom path is benchmarked over representative shapes and devices.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Treat fused MoE gate/up projection as an experimental patch, not baseline MLX support.",
        "rollback_conditions": [
          "proposal not available in target runtime",
          "conversion not reversible",
          "speedup within noise"
        ],
        "status": "research-candidate",
        "technique_id": "moe-gate-up-fusion",
        "title": "MoE gate/up projection fusion",
        "tradeoffs": [
          "Checkpoint layout compatibility risk.",
          "May add complexity for small or skewed expert groups."
        ],
        "validation_gates": [
          "token-exact parity",
          "checkpoint roundtrip",
          "end-to-end A/B/A benchmark",
          "sanitizer or shape coverage tests"
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "moe-decoder-transformer"
        ],
        "canonical_source": {
          "claim_types": [
            "api_support"
          ],
          "id": "mlx-doc-gather-mm",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "official_mlx",
          "title": "MLX gather matmul API",
          "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-doc-gather-mm",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX gather matmul API",
            "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html"
          },
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-doc-gather-qmm",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX gather quantized matmul API",
            "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html"
          },
          {
            "claim_types": [],
            "id": "mlx-repo",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX array framework repository",
            "url": "https://github.com/ml-explore/mlx/tree/96296e9c3075a2389bc5c0f078bf01b5aa377cd9"
          },
          {
            "claim_types": [],
            "id": "paper-2401-04088",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Mixtral of Experts",
            "url": "https://arxiv.org/abs/2401.04088"
          }
        ],
        "expected_effect": "Targets throughput, latency. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "layout-numerics",
        "id": "moe-gather-and-expert-batching",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "throughput",
          "latency"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Dispatch layout, gather structure, dtype, or accumulation behavior limits the target path.",
        "proof_gate": "Named axes and numerical roles are explicit and parity plus quality remain within policy.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Use gather or grouped matmul primitives when expert routing creates large enough grouped matrix multiplies and duplicate/scatter semantics match.",
        "rollback_conditions": [
          "small group overhead dominates",
          "routing parity failure"
        ],
        "status": "native-mlx",
        "technique_id": "gather-segmented-mm",
        "title": "Gather matmul for indexed expert projections",
        "tradeoffs": [
          "Small or skewed expert groups can lose.",
          "Routing, capacity, and duplicate-token semantics are easy to break."
        ],
        "validation_gates": [
          "loop oracle parity including duplicates",
          "expert skew benchmark",
          "quality and throughput at target batch/concurrency"
        ]
      },
      {
        "advisor": {
          "description": "Safe to try after parity, but no speedup or memory number may be claimed until measured on target hardware and workload.",
          "id": "benchmark-required",
          "label": "Benchmark required",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "vision-language-omni",
          "long-context-vlm",
          "multimodal-serving"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation",
            "serving_semantics"
          ],
          "id": "vllm-mlx-mllm-batch-source",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "third_party_pinned",
          "title": "vLLM-MLX multimodal batch generator implementation",
          "url": "https://github.com/waybarrios/vllm-mlx/blob/a48c86c1a41900f7d26658471b5f67e5fdd35445/vllm_mlx/mllm_batch_generator.py"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation",
              "serving_semantics"
            ],
            "id": "vllm-mlx-mllm-batch-source",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "vLLM-MLX multimodal batch generator implementation",
            "url": "https://github.com/waybarrios/vllm-mlx/blob/a48c86c1a41900f7d26658471b5f67e5fdd35445/vllm_mlx/mllm_batch_generator.py"
          },
          {
            "claim_types": [],
            "id": "mlx-vlm-apc-source",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX-VLM automatic prefix cache implementation",
            "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/mlx_vlm/apc.py"
          },
          {
            "claim_types": [],
            "id": "mlx-vlm-ar-cache-guards",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX-VLM generation cache media guards",
            "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/mlx_vlm/generate/ar.py"
          },
          {
            "claim_types": [
              "performance",
              "serving_semantics"
            ],
            "id": "paper-2601-19139",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "paper_only",
            "title": "Native LLM and MLLM Inference at Scale on Apple Silicon",
            "url": "https://arxiv.org/abs/2601.19139"
          },
          {
            "claim_types": [],
            "id": "vllm-doc-prefix-caching",
            "review_depth": "screened",
            "role": "official_docs",
            "support_scope": "unspecified",
            "title": "vLLM automatic prefix caching documentation",
            "url": "https://docs.vllm.ai/en/stable/design/prefix_caching/"
          }
        ],
        "expected_effect": "Targets ttft, prefill throughput, concurrent throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "multimodal-content-prefix-cache",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "ttft",
          "prefill-throughput",
          "concurrent-throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Use a media-aware block prefix cache only when restored cache coverage and media placeholder safety are proven.",
        "rollback_conditions": [
          "tenant isolation fails",
          "suffix media placeholders can reuse an unsafe cache",
          "no repeated-prefix workload exists"
        ],
        "status": "proven-mlx-port",
        "technique_id": "mllm-content-prefix-cache",
        "title": "Content-hashed multimodal prefix cache",
        "tradeoffs": [
          "Raises privacy, memory, eviction, and batch-noninvariance risks.",
          "May not compose with KV quantization, rotating caches, or media suffixes."
        ],
        "validation_gates": [
          "media extra hashes include content and processing knobs",
          "tenant/salt namespace test",
          "cold/warm parity",
          "mixed media suffix rejection",
          "eviction and model reload tests"
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "linear-heavy-models",
          "dense-decoder-transformer",
          "moe-decoder-transformer",
          "vision-language-omni",
          "audio-lm"
        ],
        "canonical_source": {
          "claim_types": [
            "api_support"
          ],
          "id": "mlx-doc-core-quantize",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "official_mlx",
          "title": "MLX core quantize API",
          "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "api_support"
            ],
            "id": "mlx-doc-core-quantize",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "official_mlx",
            "title": "MLX core quantize API",
            "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html"
          },
          {
            "claim_types": [],
            "id": "mlx-audio-convert",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "MLX-Audio conversion implementation",
            "url": "https://github.com/Blaizzy/mlx-audio/blob/412cf7cd381c2a3f6a8189af04a95af24cb415b6/mlx_audio/convert.py"
          },
          {
            "claim_types": [],
            "id": "mlx-doc-nn-quantize",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "unspecified",
            "title": "MLX neural network quantize API",
            "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html"
          },
          {
            "claim_types": [],
            "id": "mlx-doc-quantized-matmul",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "unspecified",
            "title": "MLX quantized matmul API",
            "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html"
          },
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "mlx-lm-convert",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "official_mlx_project",
            "title": "MLX-LM model conversion and quantization",
            "url": "https://github.com/ml-explore/mlx-lm/blob/2c008fd0252b2c569227d12568356ab88ab0560a/mlx_lm/convert.py"
          },
          {
            "claim_types": [],
            "id": "nvidia-nvfp4-blog",
            "review_depth": "screened",
            "role": "technical_blogs",
            "support_scope": "unspecified",
            "title": "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference",
            "url": "https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/"
          },
          {
            "claim_types": [],
            "id": "paper-2310-16836",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "LLM-FP4: 4-Bit Floating-Point Quantized Transformers",
            "url": "https://arxiv.org/abs/2310.16836"
          },
          {
            "claim_types": [],
            "id": "spheron-nvfp4-mxfp4-guide",
            "review_depth": "screened",
            "role": "technical_blogs",
            "support_scope": "unspecified",
            "title": "NVFP4 vs MXFP4: 4-Bit Quantization Format Decision Guide",
            "url": "https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/"
          }
        ],
        "expected_effect": "Targets peak memory, model size, throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "compression",
        "id": "native-low-bit-weight-quantization",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "peak-memory",
          "model-size",
          "throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Weight, state, or token representation dominates memory or bandwidth after baseline parity.",
        "proof_gate": "The compressed artifact passes task-specific quality and memory gates on the representative workload.",
        "quality_gate": [
          "task quality against unquantized baseline"
        ],
        "quality_gated": true,
        "recommendation": "Use MLX affine, mxfp4, mxfp8, or nvfp4 only where the target Linear path and conversion tooling support the mode.",
        "rollback_conditions": [
          "quality drift exceeds budget",
          "latency worsens",
          "unsupported layer coverage"
        ],
        "status": "native-mlx",
        "technique_id": "mxfp4-nvfp4-mxfp8",
        "title": "Native MLX affine/MX/NV low-bit quantization modes",
        "tradeoffs": [
          "Low-bit formats can harm quality, especially MoE or outlier-heavy models.",
          "quantize_input is limited to supported MLX modes and layer paths."
        ],
        "validation_gates": [
          "task quality against unquantized baseline",
          "load/save roundtrip",
          "latency and peak memory benchmark",
          "fallback precision recorded"
        ]
      },
      {
        "advisor": {
          "description": "Promising contributor, blog, repository, paper, package, or research-loop learning that is not promotion-ready for supported guidance.",
          "id": "experimental-approach",
          "label": "Experimental approach",
          "requires_user_opt_in": true
        },
        "applies_to": [
          "repetitive-prompts",
          "code",
          "rag"
        ],
        "canonical_source": {
          "claim_types": [],
          "id": "paper-2311-08252",
          "review_depth": "screened",
          "role": "papers",
          "support_scope": "unspecified",
          "title": "REST: Retrieval-Based Speculative Decoding",
          "url": "https://arxiv.org/abs/2311.08252"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [],
            "id": "paper-2311-08252",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "REST: Retrieval-Based Speculative Decoding",
            "url": "https://arxiv.org/abs/2311.08252"
          },
          {
            "claim_types": [],
            "id": "prompt-lookup-decoding-repo",
            "review_depth": "screened",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "Prompt Lookup Decoding repository",
            "url": "https://github.com/apoorvumang/prompt-lookup-decoding"
          }
        ],
        "expected_effect": "Targets decode latency. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "inference-algorithms",
        "id": "prompt-lookup-ngram-speculation",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "decode-latency"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Autoregressive target-model calls dominate after the underlying decoder is correct.",
        "proof_gate": "Acceptance semantics, token distribution or exact mode, state, and task quality match the declared policy.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Consider as a low-memory experiment for copy-heavy workloads only after implementing a target MLX drafter.",
        "rollback_conditions": [
          "low match rate",
          "no decode win"
        ],
        "status": "research-candidate",
        "technique_id": "prompt-lookup",
        "title": "Prompt lookup / n-gram speculation",
        "tradeoffs": [
          "Little help for creative generations",
          "lookup overhead can dominate",
          "needs exact verification."
        ],
        "validation_gates": [
          "hit rate",
          "accepted length",
          "exactness",
          "end-to-end latency"
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "autoregressive-transformer",
          "vision-language-omni",
          "agents",
          "rag"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation"
          ],
          "id": "mlx-lm-cache",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "official_mlx_project",
          "title": "MLX-LM cache implementations",
          "url": "https://github.com/ml-explore/mlx-lm/blob/2c008fd0252b2c569227d12568356ab88ab0560a/mlx_lm/models/cache.py"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "mlx-lm-cache",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "official_mlx_project",
            "title": "MLX-LM cache implementations",
            "url": "https://github.com/ml-explore/mlx-lm/blob/2c008fd0252b2c569227d12568356ab88ab0560a/mlx_lm/models/cache.py"
          },
          {
            "claim_types": [
              "mlx_implementation",
              "serving_semantics"
            ],
            "id": "mlx-vlm-readme",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "MLX-VLM serving, caching, and speculative decoding documentation",
            "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/README.md"
          },
          {
            "claim_types": [],
            "id": "paper-2402-05099",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Hydragen: High-Throughput LLM Inference with Shared Prefixes",
            "url": "https://arxiv.org/abs/2402.05099"
          },
          {
            "claim_types": [],
            "id": "paper-2412-03594",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing",
            "url": "https://arxiv.org/abs/2412.03594"
          },
          {
            "claim_types": [],
            "id": "paper-2606-21842",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG",
            "url": "https://arxiv.org/abs/2606.21842"
          },
          {
            "claim_types": [],
            "id": "vllm-doc-prefix-caching",
            "review_depth": "screened",
            "role": "official_docs",
            "support_scope": "unspecified",
            "title": "vLLM automatic prefix caching documentation",
            "url": "https://docs.vllm.ai/en/stable/design/prefix_caching/"
          },
          {
            "claim_types": [],
            "id": "vllm-mlx-repo",
            "review_depth": "screened",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "vLLM-MLX serving repository",
            "url": "https://github.com/waybarrios/vllm-mlx"
          }
        ],
        "expected_effect": "Targets ttft, throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "prompt-prefix-cache",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "ttft",
          "throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Reuse immutable prefix state only when model, tokenizer/template, processor, adapter, quantization, positions, and cache format version match.",
        "rollback_conditions": [
          "incorrect reuse",
          "privacy leak risk",
          "low hit rate"
        ],
        "status": "official-mlx-project",
        "technique_id": "prompt-cache",
        "title": "Prompt cache",
        "tradeoffs": [
          "Cache invalidation and namespace mistakes can corrupt outputs.",
          "Cross-tenant reuse creates privacy risk without isolation."
        ],
        "validation_gates": [
          "exact-hit and miss tests",
          "namespace/version tests",
          "save/load roundtrip",
          "tenant isolation review",
          "TTFT distribution"
        ]
      },
      {
        "advisor": {
          "description": "Safe to try after parity, but no speedup or memory number may be claimed until measured on target hardware and workload.",
          "id": "benchmark-required",
          "label": "Benchmark required",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "autoregressive-audio-lm"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation",
            "performance",
            "audio_quality"
          ],
          "id": "mlx-audio-qwen3-tts-docs",
          "review_depth": "synthesized",
          "role": "official_docs",
          "support_scope": "third_party_pinned",
          "title": "MLX-Audio Qwen3-TTS model guide",
          "url": "https://github.com/Blaizzy/mlx-audio/blob/a7ef98604cfd752e9e5c9011bcee8ec8c67228be/docs/models/tts/qwen3-tts.md"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation",
              "performance",
              "audio_quality"
            ],
            "id": "mlx-audio-qwen3-tts-docs",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "third_party_pinned",
            "title": "MLX-Audio Qwen3-TTS model guide",
            "url": "https://github.com/Blaizzy/mlx-audio/blob/a7ef98604cfd752e9e5c9011bcee8ec8c67228be/docs/models/tts/qwen3-tts.md"
          },
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "mlx-audio-release-044",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "MLX-Audio v0.4.4 release",
            "url": "https://github.com/Blaizzy/mlx-audio/releases/tag/v0.4.4"
          },
          {
            "claim_types": [],
            "id": "paper-2601-15621",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Qwen3-TTS Technical Report",
            "url": "https://arxiv.org/abs/2601.15621"
          }
        ],
        "expected_effect": "Targets concurrent throughput, rtf, first audio latency. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "qwen3-tts-batch-generation",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "concurrent-throughput",
          "rtf",
          "first-audio-latency"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Use Qwen3-TTS batch_generate for compatible concurrent TTS jobs after single-request quality is locked.",
        "rollback_conditions": [
          "TTFB or P95 tail violates budget",
          "batched audio differs from single output beyond tolerance",
          "voice/reference state crosses requests"
        ],
        "status": "proven-mlx-port",
        "technique_id": "qwen3-tts-batch-generation",
        "title": "Qwen3-TTS batch generation",
        "tradeoffs": [
          "Batching can increase TTFB and tail latency.",
          "Voice/reference/sampling isolation must be per request."
        ],
        "validation_gates": [
          "single-vs-batched parity",
          "mixed voices and lengths",
          "cancellation and fairness",
          "TTFB/RTF/memory/audio quality"
        ]
      },
      {
        "advisor": {
          "description": "Safe to try after parity, but no speedup or memory number may be claimed until measured on target hardware and workload.",
          "id": "benchmark-required",
          "label": "Benchmark required",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "diffusion-flow",
          "vision-language-omni"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation"
          ],
          "id": "katlun-grid-sample-source",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "third_party_pinned",
          "title": "mlx-grid-sample custom Metal implementation",
          "url": "https://github.com/katlun-lgtm/mlx-grid-sample/blob/467385fa2b84864659bf3076eca32c4c4d4ddbec/mlx_grid_sample.py"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "katlun-grid-sample-source",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "mlx-grid-sample custom Metal implementation",
            "url": "https://github.com/katlun-lgtm/mlx-grid-sample/blob/467385fa2b84864659bf3076eca32c4c4d4ddbec/mlx_grid_sample.py"
          },
          {
            "claim_types": [],
            "id": "katlun-grid-sample-benchmark",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "mlx-grid-sample benchmark harness",
            "url": "https://github.com/katlun-lgtm/mlx-grid-sample/blob/467385fa2b84864659bf3076eca32c4c4d4ddbec/bench_grid_sample.py"
          },
          {
            "claim_types": [],
            "id": "katlun-grid-sample-tests",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "unspecified",
            "title": "mlx-grid-sample correctness tests",
            "url": "https://github.com/katlun-lgtm/mlx-grid-sample/blob/467385fa2b84864659bf3076eca32c4c4d4ddbec/test_grid_sample.py"
          },
          {
            "claim_types": [],
            "id": "mlx-doc-custom-extensions",
            "review_depth": "synthesized",
            "role": "official_docs",
            "support_scope": "unspecified",
            "title": "MLX custom extensions documentation",
            "url": "https://ml-explore.github.io/mlx/build/html/dev/extensions.html"
          }
        ],
        "expected_effect": "Targets latency, throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "custom-backend",
        "id": "spatial-grid-sample-kernel",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "latency",
          "throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: A proven hot operation has no adequate native MLX path after safer options are exhausted.",
        "proof_gate": "A readable fallback passes parity and the custom path is benchmarked over representative shapes and devices.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Use only when the source model actually depends on PyTorch grid_sample-style spatial or volume warping; first lock a readable MLX/NumPy reference, then consider a custom Metal kernel for the profiled warp.",
        "rollback_conditions": [
          "unsupported source grid_sample mode is required",
          "oracle parity drifts",
          "layout transposes dominate",
          "kernel speedup does not improve the end-to-end model"
        ],
        "status": "proven-mlx-port",
        "technique_id": "grid-sample-metal-kernel",
        "title": "PyTorch-compatible grid_sample Metal kernel",
        "tradeoffs": [
          "Pinned implementation covers forward bilinear/trilinear sampling with zeros or border padding, not reflection, nearest mode, or training gradients.",
          "Channels-last layout avoids transposes, but PyTorch-style NCHW wrappers may copy.",
          "Custom Metal code adds maintenance and MLX-version compatibility risk."
        ],
        "validation_gates": [
          "coordinate order and align_corners parity",
          "zeros/border padding and out-of-range boundary tests",
          "NumPy or Torch oracle parity for 2D and 3D cases",
          "pure-MLX fallback path",
          "shape/dtype/layout coverage",
          "end-to-end model benchmark"
        ]
      },
      {
        "advisor": {
          "description": "Backed by official MLX/API docs, a pinned implementation, or primary paper, but still requiring local confirmation for the chosen model.",
          "id": "validated-source-theory",
          "label": "Validated by source or theory",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "dense-decoder-transformer",
          "moe-decoder-transformer",
          "autoregressive-transformer",
          "long-context"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation"
          ],
          "id": "mlx-lm-cache",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "official_mlx_project",
          "title": "MLX-LM cache implementations",
          "url": "https://github.com/ml-explore/mlx-lm/blob/2c008fd0252b2c569227d12568356ab88ab0560a/mlx_lm/models/cache.py"
        },
        "claim_eligibility": "withheld",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation"
            ],
            "id": "mlx-lm-cache",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "official_mlx_project",
            "title": "MLX-LM cache implementations",
            "url": "https://github.com/ml-explore/mlx-lm/blob/2c008fd0252b2c569227d12568356ab88ab0560a/mlx_lm/models/cache.py"
          },
          {
            "claim_types": [
              "mlx_implementation",
              "serving_semantics"
            ],
            "id": "mlx-vlm-readme",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "MLX-VLM serving, caching, and speculative decoding documentation",
            "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/README.md"
          },
          {
            "claim_types": [],
            "id": "paper-2402-02750",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
            "url": "https://arxiv.org/abs/2402.02750"
          }
        ],
        "expected_effect": "Targets peak memory, long context, concurrency. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "state-memory",
        "id": "uniform-kv-quantization",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "peak-memory",
          "long-context",
          "concurrency"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Weights or KV state exceed the useful memory budget or create avoidable transfer and cache cost.",
        "proof_gate": "State update, position, reset, isolation, and representative memory accounting are validated.",
        "quality_gate": [
          "logit/perplexity drift over context lengths",
          "retrieval/reasoning quality"
        ],
        "quality_gated": true,
        "recommendation": "Start with official/proven uniform KV quantization before adaptive or rotated KV schemes.",
        "rollback_conditions": [
          "quality regression",
          "dequant overhead erases memory/concurrency benefit",
          "cache composition failure"
        ],
        "status": "official-mlx-project",
        "technique_id": "uniform-kv-quant",
        "title": "Uniform KV cache quantization",
        "tradeoffs": [
          "Long-context quality can degrade.",
          "May not compose with every rotating, prompt-cache, trimming, batching, or save/load path."
        ],
        "validation_gates": [
          "logit/perplexity drift over context lengths",
          "retrieval/reasoning quality",
          "bytes/token/layer",
          "decode latency",
          "cache reset/reuse tests"
        ]
      },
      {
        "advisor": {
          "description": "Safe to try after parity, but no speedup or memory number may be claimed until measured on target hardware and workload.",
          "id": "benchmark-required",
          "label": "Benchmark required",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "vision-language-omni",
          "video-vlm"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation",
            "serving_semantics"
          ],
          "id": "vllm-mlx-multimodal-guide",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "third_party_pinned",
          "title": "vLLM-MLX multimodal serving guide",
          "url": "https://github.com/waybarrios/vllm-mlx/blob/a48c86c1a41900f7d26658471b5f67e5fdd35445/docs/guides/multimodal.md"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation",
              "serving_semantics"
            ],
            "id": "vllm-mlx-multimodal-guide",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "vLLM-MLX multimodal serving guide",
            "url": "https://github.com/waybarrios/vllm-mlx/blob/a48c86c1a41900f7d26658471b5f67e5fdd35445/docs/guides/multimodal.md"
          },
          {
            "claim_types": [],
            "id": "paper-2502-13923",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Qwen2.5-VL Technical Report",
            "url": "https://arxiv.org/abs/2502.13923"
          },
          {
            "claim_types": [],
            "id": "paper-2409-12191",
            "review_depth": "screened",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution",
            "url": "https://arxiv.org/abs/2409.12191"
          }
        ],
        "expected_effect": "Targets ttft, prefill throughput, peak memory. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "video-input-budgeting",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "ttft",
          "prefill-throughput",
          "peak-memory"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [
          "video QA/OCR/diagram probes"
        ],
        "quality_gated": true,
        "recommendation": "Treat FPS/frame/pixel budgets as an explicit quality mode for video VLMs.",
        "rollback_conditions": [
          "quality drops beyond budget",
          "token reduction does not improve end-to-end metric"
        ],
        "status": "proven-mlx-port",
        "technique_id": "video-input-budgeting",
        "title": "Video FPS/frame/pixel input budgeting",
        "tradeoffs": [
          "Can harm OCR, diagrams, small objects, temporal reasoning, and M-RoPE alignment.",
          "The output is intentionally a different quality-latency mode."
        ],
        "validation_gates": [
          "visual token count recorded",
          "video QA/OCR/diagram probes",
          "TTFT/decode/memory benchmark",
          "rollback threshold written before tuning"
        ]
      },
      {
        "advisor": {
          "description": "Safe to try after parity, but no speedup or memory number may be claimed until measured on target hardware and workload.",
          "id": "benchmark-required",
          "label": "Benchmark required",
          "requires_user_opt_in": false
        },
        "applies_to": [
          "vision-language-omni",
          "multimodal-serving"
        ],
        "canonical_source": {
          "claim_types": [
            "mlx_implementation",
            "serving_semantics"
          ],
          "id": "mlx-vlm-vision-cache-source",
          "review_depth": "synthesized",
          "role": "repositories",
          "support_scope": "third_party_pinned",
          "title": "MLX-VLM vision feature cache implementation",
          "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/mlx_vlm/vision_cache.py"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [
              "mlx_implementation",
              "serving_semantics"
            ],
            "id": "mlx-vlm-vision-cache-source",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "MLX-VLM vision feature cache implementation",
            "url": "https://github.com/Blaizzy/mlx-vlm/blob/6a8cdff6a1f53f46a15d4adb997c3b2d5f621263/mlx_vlm/vision_cache.py"
          },
          {
            "claim_types": [
              "mlx_implementation",
              "serving_semantics"
            ],
            "id": "vllm-mlx-vision-cache-source",
            "review_depth": "synthesized",
            "role": "repositories",
            "support_scope": "third_party_pinned",
            "title": "vLLM-MLX vision embedding cache implementation",
            "url": "https://github.com/waybarrios/vllm-mlx/blob/a48c86c1a41900f7d26658471b5f67e5fdd35445/vllm_mlx/vision_embedding_cache.py"
          },
          {
            "claim_types": [
              "performance",
              "serving_semantics"
            ],
            "id": "paper-2601-19139",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "paper_only",
            "title": "Native LLM and MLLM Inference at Scale on Apple Silicon",
            "url": "https://arxiv.org/abs/2601.19139"
          }
        ],
        "expected_effect": "Targets ttft, prefill throughput. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "serving-pipeline",
        "id": "vision-feature-cache",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "ttft",
          "prefill-throughput"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "quality_gate": [],
        "quality_gated": false,
        "recommendation": "Use for repeated identical image/video inputs after keying by content hash and processor/projector revision.",
        "rollback_conditions": [
          "warm outputs drift from cold outputs",
          "cache key lacks model/processor/projector/media fields",
          "reuse is rare enough that memory overhead dominates"
        ],
        "status": "proven-mlx-port",
        "technique_id": "vision-feature-cache",
        "title": "Projected vision feature cache",
        "tradeoffs": [
          "Cache keys are more complex than text prefixes.",
          "Stale media or changed preprocessing can silently corrupt results."
        ],
        "validation_gates": [
          "cold/warm logits parity",
          "content-hash key mutation tests",
          "eviction/reset/mixed-batch tests",
          "TTFT and prompt throughput benchmark"
        ]
      },
      {
        "advisor": {
          "description": "Promising contributor, blog, repository, paper, package, or research-loop learning that is not promotion-ready for supported guidance.",
          "id": "experimental-approach",
          "label": "Experimental approach",
          "requires_user_opt_in": true
        },
        "applies_to": [
          "vision-language-omni"
        ],
        "canonical_source": {
          "claim_types": [],
          "id": "llava-onevision-blog",
          "review_depth": "synthesized",
          "role": "technical_blogs",
          "support_scope": "unspecified",
          "title": "LLaVA-OneVision release blog",
          "url": "https://llava-vl.github.io/blog/2024-08-05-llava-onevision/"
        },
        "claim_eligibility": "not-catalogued",
        "evidence_links": [
          {
            "claim_types": [],
            "id": "llava-onevision-blog",
            "review_depth": "synthesized",
            "role": "technical_blogs",
            "support_scope": "unspecified",
            "title": "LLaVA-OneVision release blog",
            "url": "https://llava-vl.github.io/blog/2024-08-05-llava-onevision/"
          },
          {
            "claim_types": [],
            "id": "paper-2403-15388",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models",
            "url": "https://arxiv.org/abs/2403.15388"
          },
          {
            "claim_types": [],
            "id": "paper-2412-04467",
            "review_depth": "synthesized",
            "role": "papers",
            "support_scope": "unspecified",
            "title": "VisionZip: Longer is Better but Not Necessary in Vision Language Models",
            "url": "https://arxiv.org/abs/2412.04467"
          }
        ],
        "expected_effect": "Targets ttft, prefill throughput, peak memory. No numeric effect is claimed; profile the declared target workload and consult the effective-claim catalog before publishing a number.",
        "family_id": "compression",
        "id": "visual-token-pruning-or-merge",
        "numeric_authority": "effective_claims",
        "numeric_claim": null,
        "objectives": [
          "ttft",
          "prefill-throughput",
          "peak-memory"
        ],
        "prerequisite": "A parity-passing readable baseline and a profile showing this bottleneck: Weight, state, or token representation dominates memory or bandwidth after baseline parity.",
        "proof_gate": "The compressed artifact passes task-specific quality and memory gates on the representative workload.",
        "quality_gate": [
          "task quality suite"
        ],
        "quality_gated": true,
        "recommendation": "Do not recommend generic visual-token pruning unless the model-specific MLX implementation and quality probes exist.",
        "rollback_conditions": [
          "quality regression",
          "method not implemented in MLX",
          "no end-to-end speedup"
        ],
        "status": "research-candidate",
        "technique_id": "visual-token-pruning-or-merge",
        "title": "Generic visual-token pruning or merging",
        "tradeoffs": [
          "Can destroy text, diagram, OCR, spatial, or temporal reasoning.",
          "Often requires model-specific feature semantics."
        ],
        "validation_gates": [
          "MLX implementation path",
          "task quality suite",
          "token-count and latency benchmark",
          "explicit rollback threshold"
        ]
      }
    ],
    "journey_statuses": {
      "proven": "A pinned checkpoint has a checked-in, reproducible MLX proof packet in this repository.",
      "simulation": "A teaching route assembled from canonical runbooks and evidence; it is not a completed checkpoint port."
    },
    "journeys": [
      {
        "architecture_ids": [
          "dense-decoder-transformer"
        ],
        "checkpoint_notes": {
          "implement": "Build a readable standalone MLX decoder with explicit RoPE, GQA, RMSNorm, masks, and cache.",
          "inspect": "Pin the exact Qwen2.5-0.5B-Instruct revision and identify the dense decoder route.",
          "map": "Apply the checked-in schema-2 rename and transform contract with full source-key coverage.",
          "optimize": "Only after parity, test attention, compile, dtype, quantization, cache, or decode-algorithm branches independently.",
          "oracle": "Capture tokenizer IDs, primitive tensors, repeated blocks, logits, KV state, and greedy outputs.",
          "parity": "Run the ordered 29-rung ladder under its declared floating-point policy.",
          "profile": "Measure load, prefill, and cached decode separately; keep observations tied to the local receipt.",
          "publish": "Publish the exact checkpoint boundary, parity packet, receipts, and limitations without generalizing to all Qwen models."
        },
        "component_path": [
          {
            "checkpoint": "oracle",
            "concept": "Prompt formatting and tokenization define the first model input.",
            "evidence_state": "proven",
            "id": "qwen-tokenizer",
            "inspect": "Tokenizer revision, special tokens, template, padding, and input IDs.",
            "outcome": "Exact input IDs",
            "prerequisite": "Pinned model and tokenizer artifacts.",
            "proof": "The source and target consume the same token IDs.",
            "title": "Tokenizer contract",
            "why_mlx_differs": "The MLX graph can be correct while a different chat template changes every later value."
          },
          {
            "checkpoint": "implement",
            "concept": "The embedding table maps token IDs into the model width.",
            "evidence_state": "proven",
            "id": "qwen-embedding",
            "inspect": "Vocabulary size, hidden size, tied weights, dtype, and gather output.",
            "outcome": "Tokens become hidden states",
            "prerequisite": "Exact token IDs and mapped embedding weight.",
            "proof": "Embedding outputs pass the declared tensor comparison.",
            "title": "Embedding",
            "why_mlx_differs": "Weight naming and gather semantics must match before repeated blocks are meaningful."
          },
          {
            "checkpoint": "parity",
            "concept": "Repeated RMSNorm, RoPE, grouped-query attention, residual, and MLP blocks transform the sequence.",
            "evidence_state": "proven",
            "id": "qwen-decoder-block",
            "inspect": "Norm epsilon, projection layout, head counts, RoPE base, mask, activation, and residual order.",
            "outcome": "Attention and MLP preserve source order",
            "prerequisite": "Primitive-level parity for norm, RoPE, attention, and MLP.",
            "proof": "Every ordered block checkpoint passes before the final head is compared.",
            "title": "Decoder block loop",
            "why_mlx_differs": "Mask shape, RoPE convention, GQA grouping, residual order, and lazy state must stay explicit."
          },
          {
            "checkpoint": "parity",
            "concept": "Key and value state grows across autoregressive generation.",
            "evidence_state": "proven",
            "id": "qwen-kv-state",
            "inspect": "Cache axes, length, grouped heads, positions, reset, and next-token logits.",
            "outcome": "Cached decode matches full-context semantics",
            "prerequisite": "Full prefill parity and explicit cache state.",
            "proof": "Single-token cached decode matches the source at each checked step.",
            "title": "KV cache state",
            "why_mlx_differs": "Cache layout, position offsets, update order, and evaluation boundaries are target-runtime choices."
          },
          {
            "checkpoint": "publish",
            "concept": "Final normalization and the language head produce logits used by the generation loop.",
            "evidence_state": "proven",
            "id": "qwen-logits-generation",
            "inspect": "Final norm, head weight, logits, greedy selection, stop tokens, and emitted IDs.",
            "outcome": "The pinned task result matches",
            "prerequisite": "All earlier parity rungs and cache checks pass.",
            "proof": "The checked-in packet records matching inputs and the exact greedy output sequence.",
            "title": "Norm, head, and generation",
            "why_mlx_differs": "Tied weights, sampling order, cache reuse, and stop rules can diverge after tensor parity."
          }
        ],
        "id": "qwen25-dense-decoder",
        "modality": "text",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_outcome_id": "qwen25-05b-instruct-local-worked-port",
        "optimization_method_ids": [
          "fast-sdpa",
          "compile-stable-region",
          "lazy-eval-boundaries",
          "bf16-weight-cast",
          "native-low-bit-weight-quantization",
          "uniform-kv-quantization",
          "adaptive-kv-quantization",
          "prompt-prefix-cache",
          "continuous-batching-serving",
          "draft-model-speculation",
          "prompt-lookup-ngram-speculation",
          "eagle-medusa-mtp-drafters",
          "cache-privacy-and-isolation",
          "cuda-graphs-decode-capture"
        ],
        "proof_boundary": "Claim one pinned local real-model port, exact eight-token greedy agreement for the F32 artifact, and one held F32-versus-BF16 wall-time observation. Do not generalize the tolerance, BF16 quality window, performance ratio, longer-context behavior, batching, or publication readiness.",
        "proof_ladder_rungs": 29,
        "runbooks": [
          {
            "id": "dense-decoder-transformer",
            "label": "Dense decoder Transformer",
            "path": "references/runbook-decoder-transformer.md"
          }
        ],
        "source_format": "safetensors",
        "source_ids": [
          "mlx-lm-models",
          "mlx-lm-convert",
          "paper-2407-10671"
        ],
        "status": "proven",
        "summary": "The only guided lab backed by a checked-in checkpoint port and ordered parity packet.",
        "title": "Qwen2.5-0.5B-Instruct"
      },
      {
        "architecture_ids": [
          "automatic-speech-recognition",
          "encoder-decoder-transformer"
        ],
        "checkpoint_notes": {
          "implement": "Rebuild the frontend, audio encoder, text decoder, masks, positions, and distinct cache lifetimes.",
          "inspect": "Classify speech modality separately from the encoder-decoder architecture and frontend.",
          "map": "Map convolution, encoder, decoder, cross-attention, embedding, and head tensors explicitly.",
          "optimize": "For offline Whisper, start with attention, compilation, and evaluation boundaries only when their prerequisites hold.",
          "oracle": "Capture waveform, mel features, encoder outputs, cross-attention decoder checkpoints, and timestamps.",
          "parity": "Prove features, encoder, decoder, state, logits, transcript, and timestamp behavior in order.",
          "profile": "Separate frontend, encoder, autoregressive decoder, and task post-processing costs.",
          "publish": "Keep this route labeled simulation until a pinned checkpoint and task-quality packet are reproduced."
        },
        "component_path": [
          {
            "checkpoint": "inspect",
            "concept": "Sample rate, channel policy, normalization, clipping, and duration define the raw input.",
            "evidence_state": "simulation",
            "id": "whisper-waveform",
            "inspect": "Sample rate, channel fold, dtype, amplitude, padding, and truncation.",
            "outcome": "Deterministic audio input",
            "prerequisite": "Pinned audio fixture and preprocessing configuration.",
            "proof": "The same normalized waveform reaches both frontends.",
            "title": "Waveform contract",
            "why_mlx_differs": "A model graph cannot correct a frontend mismatch introduced before MLX sees the samples."
          },
          {
            "checkpoint": "oracle",
            "concept": "Windowing, FFT, mel filters, log scaling, and padding create encoder features.",
            "evidence_state": "simulation",
            "id": "whisper-mel-frontend",
            "inspect": "Window, hop, FFT size, mel basis, log floor, frame count, layout, and dtype.",
            "outcome": "Feature parity before the network",
            "prerequisite": "Exact waveform contract and source frontend oracle.",
            "proof": "Mel features pass shape and value gates under an explicit tolerance.",
            "title": "Mel frontend",
            "why_mlx_differs": "FFT and layout details can diverge before any learned layer runs."
          },
          {
            "checkpoint": "implement",
            "concept": "The encoder turns mel frames into contextual audio representations.",
            "evidence_state": "simulation",
            "id": "whisper-encoder",
            "inspect": "Frontend convolutions, positions, encoder masks, attention, MLP, and norms.",
            "outcome": "Context features match",
            "prerequisite": "Feature parity and mapped encoder weights.",
            "proof": "Selected encoder blocks and the final encoder output pass parity.",
            "title": "Audio encoder",
            "why_mlx_differs": "Convolution layout, positions, attention masks, and normalization need explicit translation."
          },
          {
            "checkpoint": "parity",
            "concept": "The decoder mixes text self-attention with cross-attention over encoder features.",
            "evidence_state": "simulation",
            "id": "whisper-cross-attention-decoder",
            "inspect": "Decoder prompt tokens, self mask, cross mask, cache split, forced tokens, and language/task controls.",
            "outcome": "Text state attends to audio correctly",
            "prerequisite": "Encoder output parity and explicit cache structures.",
            "proof": "Decoder checkpoints and next-token logits match for the same state.",
            "title": "Cross-attention decoder",
            "why_mlx_differs": "Self and cross caches have different lifetimes, masks, positions, and update rules."
          },
          {
            "checkpoint": "publish",
            "concept": "Token decoding, suppression rules, timestamps, and text normalization create the user-visible output.",
            "evidence_state": "simulation",
            "id": "whisper-transcript-timestamps",
            "inspect": "Decoding options, suppressed tokens, timestamp rules, normalization, WER or CER, and segment output.",
            "outcome": "Task behavior is measured",
            "prerequisite": "Encoder-decoder parity on a representative corpus slice.",
            "proof": "Task metrics and timestamp behavior meet a declared quality gate.",
            "title": "Transcript and timestamps",
            "why_mlx_differs": "Matching logits is necessary but not sufficient for identical transcript and timestamp behavior."
          }
        ],
        "hybrid_profile_id": "whisper-asr-seq2seq",
        "id": "whisper-style-asr",
        "modality": "audio-to-text",
        "optimization_method_ids": [
          "fast-sdpa",
          "compile-stable-region",
          "lazy-eval-boundaries"
        ],
        "proof_boundary": "This is a runbook simulation, not a completed Whisper checkpoint port. The official MLX example proves an implementation path exists; it does not provide this repository's pinned parity and task-quality packet.",
        "runbooks": [
          {
            "id": "automatic-speech-recognition",
            "label": "Automatic speech recognition",
            "path": "references/runbook-asr.md"
          },
          {
            "id": "encoder-decoder-transformer",
            "label": "Encoder-decoder Transformer",
            "path": "references/runbook-encoder-decoder.md"
          }
        ],
        "source_format": "safetensors",
        "source_ids": [
          "repo-whisper",
          "paper-2311-00430",
          "mlx-examples-repo"
        ],
        "status": "simulation",
        "summary": "A runbook simulation for audio preprocessing, encoder-decoder structure, cross-attention, cache, timestamps, and task quality.",
        "title": "Whisper-style ASR"
      },
      {
        "architecture_ids": [
          "diffusion-flow"
        ],
        "checkpoint_notes": {
          "implement": "Rebuild components independently, then compose a readable single-step pipeline before iteration.",
          "inspect": "Inventory every pipeline component, scheduler, source artifact, and conditioning path.",
          "map": "Keep encoder, denoiser, and VAE weight namespaces and transforms separately auditable.",
          "optimize": "Current registry candidates are stable compilation, evaluation boundaries, block streaming, and the spatial grid-sample kernel.",
          "oracle": "Capture conditioning, initial latent, schedule, selected denoiser blocks, latent steps, and final decode.",
          "parity": "Compare scheduler identity, one-step updates, fixed latent checkpoints, VAE output, and quality.",
          "profile": "Separate prompt encoding, repeated denoiser work, scheduler overhead, transfers, and VAE decode.",
          "publish": "Label the route FLUX-style simulation until an exact model source and reproducible MLX proof are checked in."
        },
        "component_path": [
          {
            "checkpoint": "inspect",
            "concept": "Tokenizer and text encoder outputs condition the image-generation process.",
            "evidence_state": "simulation",
            "id": "flux-prompt-encoder",
            "inspect": "Tokenizer, encoder variants, hidden-state selection, pooling, masks, and guidance inputs.",
            "outcome": "Conditioning is reproducible",
            "prerequisite": "Pinned pipeline configuration and prompt fixture.",
            "proof": "Conditioning tensors match before entering the denoiser.",
            "title": "Prompt encoder",
            "why_mlx_differs": "A pipeline may contain separately versioned encoders with their own layouts and dtype policies."
          },
          {
            "checkpoint": "oracle",
            "concept": "Initial latent, timestep or sigma schedule, guidance, and RNG define the iterative path.",
            "evidence_state": "simulation",
            "id": "flux-latent-scheduler",
            "inspect": "Seed, latent shape, scheduler configuration, timestep sequence, scaling, and guidance.",
            "outcome": "The same trajectory starts",
            "prerequisite": "Pinned pipeline config and deterministic source execution.",
            "proof": "Initial latent and every scheduled input agree under the declared policy.",
            "title": "Latent and scheduler",
            "why_mlx_differs": "Different RNG, scheduler math, or timestep dtype changes every later latent."
          },
          {
            "checkpoint": "implement",
            "concept": "A large transformer or denoiser predicts the update applied at each scheduled step.",
            "evidence_state": "simulation",
            "id": "flux-denoiser-step",
            "inspect": "Block order, conditioning joins, attention, timestep embedding, residuals, and output parameterization.",
            "outcome": "One iterative update matches",
            "prerequisite": "Matching latent, schedule, and conditioning tensors.",
            "proof": "Selected internal blocks and one full update pass parity before iteration.",
            "title": "Repeated flow or denoiser step",
            "why_mlx_differs": "Layout, multimodal conditioning, timestep embedding, attention, and lazy graph lifetime shape the hot loop."
          },
          {
            "checkpoint": "parity",
            "concept": "The scheduler repeatedly combines predictions and latents.",
            "evidence_state": "simulation",
            "id": "flux-latent-loop",
            "inspect": "Latent checkpoints at fixed steps, evaluation boundaries, dtype, and scheduler state.",
            "outcome": "Divergence stays bounded across steps",
            "prerequisite": "One-step parity and identical schedule.",
            "proof": "Fixed-step latent checkpoints satisfy the declared comparison policy.",
            "title": "Latent trajectory",
            "why_mlx_differs": "Small numerical or scheduler differences accumulate over the entire trajectory."
          },
          {
            "checkpoint": "publish",
            "concept": "The final latent is decoded and post-processed into an image.",
            "evidence_state": "simulation",
            "id": "flux-vae-image",
            "inspect": "Latent scaling, decoder, output range, channel order, post-processing, and quality metric.",
            "outcome": "The task output is comparable",
            "prerequisite": "Latent trajectory parity and mapped VAE weights.",
            "proof": "Fixed-seed output and declared image-quality checks pass without claiming pixel identity by default.",
            "title": "VAE decode and image",
            "why_mlx_differs": "VAE layouts, scaling, clipping, color conversion, and quality metrics sit outside the denoiser."
          }
        ],
        "id": "flux-style-diffusion",
        "modality": "text-to-image",
        "optimization_method_ids": [
          "compile-stable-region",
          "lazy-eval-boundaries",
          "block-weight-streaming",
          "spatial-grid-sample-kernel"
        ],
        "proof_boundary": "This is a generic FLUX-style teaching route, not an exact FLUX checkpoint claim. Fixed-seed output still requires declared image-quality checks; pixel identity is not promised by default.",
        "runbooks": [
          {
            "id": "diffusion-flow",
            "label": "Diffusion and flow models",
            "path": "references/runbook-diffusion-flow.md"
          }
        ],
        "source_format": "safetensors",
        "source_ids": [
          "paper-2212-09748",
          "paper-2112-10752",
          "paper-2209-03003",
          "paper-2210-02747",
          "paper-2403-03206",
          "mlx-examples-repo"
        ],
        "status": "simulation",
        "summary": "A generic FLUX-style simulation for multi-component pipelines and repeated flow or denoising steps; it is not an exact FLUX port claim.",
        "title": "FLUX-style diffusion/flow"
      },
      {
        "architecture_ids": [
          "vision-language-omni",
          "encoder-transformer",
          "dense-decoder-transformer"
        ],
        "checkpoint_notes": {
          "implement": "Rebuild each component through its native runbook before composing the multimodal path.",
          "inspect": "Route image processing, the encoder tower, projector, dense decoder, and KV state as separate components.",
          "map": "Keep vision, projector, and language tensor transforms distinct with complete source-key coverage.",
          "optimize": "Select attention, compile, memory, cache, batching, spatial, or research branches only after their own gates.",
          "oracle": "Capture pixels, vision features, projected embeddings, token assembly, logits, cache, and final output.",
          "parity": "Prove every modality boundary before evaluating the final image-conditioned task.",
          "profile": "Separate image processing, vision encoding, projection, language prefill, cached decode, and serving reuse.",
          "publish": "Keep Apple MLX, third-party MLX-VLM, paper-only research, and local reproduction states visibly distinct."
        },
        "component_path": [
          {
            "checkpoint": "inspect",
            "concept": "Resize, crop, normalize, tile, and patch policies create the vision input.",
            "evidence_state": "simulation",
            "id": "llava-image-processor",
            "inspect": "Image size, aspect policy, interpolation, normalization, tiling, channel order, and batch shape.",
            "outcome": "Pixels have an exact contract",
            "prerequisite": "Pinned processor configuration and image fixture.",
            "proof": "Processed pixels and metadata match the source contract.",
            "title": "Image processor",
            "why_mlx_differs": "A correct vision encoder cannot repair different pixels or patch layout."
          },
          {
            "checkpoint": "oracle",
            "concept": "An encoder transformer turns image patches into visual features.",
            "evidence_state": "simulation",
            "id": "llava-vision-encoder",
            "inspect": "Patch embedding, positions, masks, encoder blocks, selected layers, and output layout.",
            "outcome": "Image features match",
            "prerequisite": "Pixel parity and mapped vision weights.",
            "proof": "Selected vision blocks and final feature tensors pass parity.",
            "title": "Vision encoder",
            "why_mlx_differs": "Patch layout, position encoding, attention, feature selection, and dtype must be routed through the encoder family."
          },
          {
            "checkpoint": "implement",
            "concept": "A projector maps vision width and token structure into the language model embedding space.",
            "evidence_state": "simulation",
            "id": "llava-projector",
            "inspect": "Input layer selection, projector architecture, activation, output width, and token count.",
            "outcome": "Vision features enter language space",
            "prerequisite": "Vision feature parity and mapped projector weights.",
            "proof": "Projected embeddings match before they are inserted into the text sequence.",
            "title": "Multimodal projector",
            "why_mlx_differs": "Projection layout, activation, feature selection, and image-token count determine language positions."
          },
          {
            "checkpoint": "parity",
            "concept": "Projected image tokens replace or join placeholders in the language sequence.",
            "evidence_state": "simulation",
            "id": "llava-token-assembly",
            "inspect": "Prompt template, image placeholders, insertion index, attention mask, positions, and labels.",
            "outcome": "Image and text positions agree",
            "prerequisite": "Projected embedding and tokenizer parity.",
            "proof": "Assembled embeddings, positions, and masks match exactly where required.",
            "title": "Multimodal token assembly",
            "why_mlx_differs": "Template, placeholder count, padding, positions, and masks can shift the entire decoder state."
          },
          {
            "checkpoint": "publish",
            "concept": "A dense decoder with KV state generates text conditioned on visual tokens.",
            "evidence_state": "simulation",
            "id": "llava-decoder-output",
            "inspect": "Prefill logits, cached decode, positions, stop rules, text output, and image-conditioned task quality.",
            "outcome": "Multimodal behavior is measured",
            "prerequisite": "All modality-boundary and decoder checkpoints pass.",
            "proof": "Logits, cache behavior, and a declared multimodal task gate pass for the pinned scenario.",
            "title": "Language decoder and output",
            "why_mlx_differs": "Decoder attention, cache, image-prefix reuse, sampling, and serving state are separate contracts."
          }
        ],
        "id": "llava-style-vlm",
        "modality": "image-to-text",
        "optimization_method_ids": [
          "fast-sdpa",
          "compile-stable-region",
          "lazy-eval-boundaries",
          "block-weight-streaming",
          "native-low-bit-weight-quantization",
          "prompt-prefix-cache",
          "continuous-batching-serving",
          "content-prefix-cache-vlm",
          "eagle-medusa-mtp-drafters",
          "cache-privacy-and-isolation",
          "vision-feature-cache",
          "multimodal-content-prefix-cache",
          "video-input-budgeting",
          "visual-token-pruning-or-merge",
          "spatial-grid-sample-kernel"
        ],
        "proof_boundary": "This is a composed-route simulation with no pinned model outcome or local proof ladder. MLX-VLM evidence is third-party and must not be presented as Apple-official or locally reproduced.",
        "runbooks": [
          {
            "id": "vision-language-omni",
            "label": "Vision-language, audio-language, and omni models",
            "path": "references/runbook-multimodal-omni.md"
          },
          {
            "id": "encoder-transformer",
            "label": "Encoder Transformer",
            "path": "references/runbook-encoder-transformer.md"
          },
          {
            "id": "dense-decoder-transformer",
            "label": "Dense decoder Transformer",
            "path": "references/runbook-decoder-transformer.md"
          }
        ],
        "source_format": "safetensors",
        "source_ids": [
          "paper-2304-08485",
          "mlx-vlm-repo",
          "mlx-vlm-readme",
          "llava-onevision-blog",
          "paper-2403-15388"
        ],
        "status": "simulation",
        "summary": "A composed-route simulation that keeps image processing, vision encoding, projection, language decoding, and KV state visible.",
        "title": "LLaVA-style vision-language model"
      }
    ],
    "method_quality_gates": {
      "adaptive-kv-quantization": [
        "matched bit budget comparison",
        "long-context quality"
      ],
      "audio-streaming-and-cache": [
        "chunk-boundary continuity",
        "speaker similarity/intelligibility"
      ],
      "bf16-weight-cast": [
        "task-specific quality against F32"
      ],
      "moe-expert-dispatch-and-quantization": [
        "quality against the unquantized expert baseline"
      ],
      "native-low-bit-weight-quantization": [
        "task quality against unquantized baseline"
      ],
      "uniform-kv-quantization": [
        "logit/perplexity drift over context lengths",
        "retrieval/reasoning quality"
      ],
      "video-input-budgeting": [
        "video QA/OCR/diagram probes"
      ],
      "visual-token-pruning-or-merge": [
        "task quality suite"
      ]
    },
    "official_learning_source_ids": [
      "mlx-docs",
      "mlx-doc-quick-start",
      "mlx-doc-unified-memory",
      "mlx-doc-lazy",
      "mlx-doc-compile",
      "mlx-doc-framework-conversion",
      "mlx-doc-function-transforms",
      "mlx-doc-neural-networks",
      "mlx-doc-streams",
      "mlx-doc-fast-sdpa",
      "mlx-repo",
      "mlx-examples-repo",
      "mlx-lm-repo"
    ],
    "official_learning_sources": [
      {
        "id": "mlx-docs",
        "title": "MLX documentation",
        "url": "https://ml-explore.github.io/mlx/build/html/index.html"
      },
      {
        "id": "mlx-doc-quick-start",
        "title": "MLX quick start",
        "url": "https://ml-explore.github.io/mlx/build/html/usage/quick_start.html"
      },
      {
        "id": "mlx-doc-unified-memory",
        "title": "MLX unified memory documentation",
        "url": "https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html"
      },
      {
        "id": "mlx-doc-lazy",
        "title": "MLX lazy evaluation documentation",
        "url": "https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html"
      },
      {
        "id": "mlx-doc-compile",
        "title": "MLX compilation documentation",
        "url": "https://ml-explore.github.io/mlx/build/html/usage/compile.html"
      },
      {
        "id": "mlx-doc-framework-conversion",
        "title": "MLX framework conversion and DLPack documentation",
        "url": "https://ml-explore.github.io/mlx/build/html/usage/numpy.html"
      },
      {
        "id": "mlx-doc-function-transforms",
        "title": "MLX function transforms documentation",
        "url": "https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html"
      },
      {
        "id": "mlx-doc-neural-networks",
        "title": "MLX neural networks documentation",
        "url": "https://ml-explore.github.io/mlx/build/html/python/nn.html"
      },
      {
        "id": "mlx-doc-streams",
        "title": "MLX using streams documentation",
        "url": "https://ml-explore.github.io/mlx/build/html/usage/using_streams.html"
      },
      {
        "id": "mlx-doc-fast-sdpa",
        "title": "MLX fast scaled dot product attention API",
        "url": "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html"
      },
      {
        "id": "mlx-repo",
        "title": "MLX array framework repository",
        "url": "https://github.com/ml-explore/mlx/tree/96296e9c3075a2389bc5c0f078bf01b5aa377cd9"
      },
      {
        "id": "mlx-examples-repo",
        "title": "MLX examples repository",
        "url": "https://github.com/ml-explore/mlx-examples"
      },
      {
        "id": "mlx-lm-repo",
        "title": "MLX-LM repository",
        "url": "https://github.com/ml-explore/mlx-lm/tree/2c008fd0252b2c569227d12568356ab88ab0560a"
      }
    ],
    "optimization_families": [
      {
        "bottleneck": "Lazy graph lifetime, synchronization, or misplaced evaluation dominates the measured phase.",
        "id": "evaluation-scheduling",
        "method_ids": [
          "lazy-eval-boundaries"
        ],
        "proof_gate": "Parity passes with identical state and the measured region realizes the intended work exactly once.",
        "rollback": "Remove the boundary change when it increases memory, synchronization, or end-to-end latency.",
        "title": "Evaluation and scheduling"
      },
      {
        "bottleneck": "A compatible repeated operator or stable graph region dominates runtime.",
        "id": "native-operators-compilation",
        "method_ids": [
          "fast-sdpa",
          "compile-stable-region"
        ],
        "proof_gate": "Eager parity exists and the native or compiled path matches it across representative shapes and state.",
        "rollback": "Use the readable eager fallback on semantic mismatch, retracing instability, or no end-to-end benefit.",
        "title": "Native operators and compilation"
      },
      {
        "bottleneck": "Dispatch layout, gather structure, dtype, or accumulation behavior limits the target path.",
        "id": "layout-numerics",
        "method_ids": [
          "moe-gather-and-expert-batching"
        ],
        "proof_gate": "Named axes and numerical roles are explicit and parity plus quality remain within policy.",
        "rollback": "Restore the baseline layout or precision when conversion overhead, instability, or quality loss appears.",
        "title": "Layout and numerical representation"
      },
      {
        "bottleneck": "Weights or KV state exceed the useful memory budget or create avoidable transfer and cache cost.",
        "id": "state-memory",
        "method_ids": [
          "block-weight-streaming",
          "uniform-kv-quantization",
          "adaptive-kv-quantization"
        ],
        "proof_gate": "State update, position, reset, isolation, and representative memory accounting are validated.",
        "rollback": "Return to full state or resident weights when parity, quality, privacy, or end-to-end latency regresses.",
        "title": "State and memory"
      },
      {
        "bottleneck": "Weight, state, or token representation dominates memory or bandwidth after baseline parity.",
        "id": "compression",
        "method_ids": [
          "bf16-weight-cast",
          "native-low-bit-weight-quantization",
          "visual-token-pruning-or-merge",
          "moe-expert-dispatch-and-quantization"
        ],
        "proof_gate": "The compressed artifact passes task-specific quality and memory gates on the representative workload.",
        "rollback": "Restore the last parity-passing representation when quality, stability, or total latency fails its gate.",
        "title": "Compression"
      },
      {
        "bottleneck": "Autoregressive target-model calls dominate after the underlying decoder is correct.",
        "id": "inference-algorithms",
        "method_ids": [
          "draft-model-speculation",
          "prompt-lookup-ngram-speculation",
          "eagle-medusa-mtp-drafters"
        ],
        "proof_gate": "Acceptance semantics, token distribution or exact mode, state, and task quality match the declared policy.",
        "rollback": "Use baseline generation when acceptance overhead, incompatibility, or output-quality drift removes the benefit.",
        "title": "Inference algorithms"
      },
      {
        "bottleneck": "Batching, preprocessing, cache reuse, streaming, or multi-request scheduling limits the real workload.",
        "id": "serving-pipeline",
        "method_ids": [
          "prompt-prefix-cache",
          "continuous-batching-serving",
          "content-prefix-cache-vlm",
          "audio-streaming-and-cache",
          "cache-privacy-and-isolation",
          "vision-feature-cache",
          "multimodal-content-prefix-cache",
          "video-input-budgeting",
          "qwen3-tts-batch-generation",
          "audio-reference-conditioning-cache",
          "generic-audio-prefix-cache"
        ],
        "proof_gate": "Isolation, cache keys, fairness, task quality, workload mix, and end-to-end behavior are validated.",
        "rollback": "Disable reuse or scheduling changes on privacy, correctness, starvation, memory, or tail-latency regression.",
        "title": "Serving and pipeline"
      },
      {
        "bottleneck": "A proven hot operation has no adequate native MLX path after safer options are exhausted.",
        "id": "custom-backend",
        "method_ids": [
          "moe-gate-up-fusion",
          "spatial-grid-sample-kernel",
          "cuda-graphs-decode-capture"
        ],
        "proof_gate": "A readable fallback passes parity and the custom path is benchmarked over representative shapes and devices.",
        "rollback": "Keep or restore the fallback on semantic mismatch, maintenance risk, rejected platform assumptions, or no measured win.",
        "title": "Custom backend work"
      }
    ],
    "reviewed": "2026-07-12",
    "schema_version": 1,
    "translation_lens": [
      {
        "common_trap": "Assuming every external buffer shares storage without a copy.",
        "example": "Keep the array identity stable and choose the GPU stream for the operation that consumes it.",
        "id": "array-and-device",
        "mlx_translation": "Use mx.array and operation or stream placement; treat framework interop as a separate boundary.",
        "next_step": "Account for lazy evaluation before comparing timing.",
        "plain_language": "MLX arrays live in unified memory while operations select a device and stream.",
        "proof_check": "The same input produces the expected shape, dtype, and values on the selected MLX device.",
        "pytorch_cuda": "torch.Tensor plus .to('cuda') or explicit host/device copies.",
        "title": "Arrays and device placement",
        "why_it_matters": "Ported code should not copy PyTorch device-transfer choreography blindly."
      },
      {
        "common_trap": "Printing, scalar extraction, or NumPy conversion creates an accidental synchronization point.",
        "example": "Evaluate model outputs inside a timed region before stopping the timer.",
        "id": "lazy-evaluation",
        "mlx_translation": "Use deliberate mx.eval boundaries at comparisons, dependencies, and measurements.",
        "next_step": "Make checkpoint transforms explicit.",
        "plain_language": "MLX builds computation graphs lazily and realizes them when results are needed.",
        "proof_check": "Timing includes the intended work once and parity reads realized outputs.",
        "pytorch_cuda": "Most eager PyTorch operations execute as they are issued.",
        "title": "Eager assumptions to lazy evaluation",
        "why_it_matters": "Evaluation placement affects memory, timing, and when state becomes observable."
      },
      {
        "common_trap": "Guessing a transpose from a coincidentally compatible shape.",
        "example": "Split a fused QKV tensor only when the source layout and target attention module require it.",
        "id": "weight-map",
        "mlx_translation": "Declare rename, transpose, reshape, split, merge, and dtype transforms before assignment.",
        "next_step": "Verify operation-specific layouts.",
        "plain_language": "Weights need a deterministic name, shape, layout, and dtype transformation contract.",
        "proof_check": "Source-key coverage is complete and every transformed tensor has a shape and checksum record.",
        "pytorch_cuda": "Load a state_dict into a source class with framework-native naming and layouts.",
        "title": "state_dict to weight map",
        "why_it_matters": "Matching parameter counts does not prove tensors land in semantically equivalent operations."
      },
      {
        "common_trap": "Applying one repository-wide layout rule to every modality.",
        "example": "Record whether attention uses batch, heads, query, and key dimensions before calling fast SDPA.",
        "id": "tensor-layout",
        "mlx_translation": "Make every required transpose and reshape visible at the boundary that needs it.",
        "next_step": "Choose numerical roles explicitly.",
        "plain_language": "Layout is an operation contract, not a global framework preference.",
        "proof_check": "The primitive comparison passes with named axes and no unexplained reshape.",
        "pytorch_cuda": "Source code may rely on NCHW conventions or implicit contiguous tensors.",
        "title": "Framework layout to operation layout",
        "why_it_matters": "Image, attention, convolution, and linear operations may expect different axis orders."
      },
      {
        "common_trap": "Calling a full-model dtype cast a lossless optimization.",
        "example": "Keep a sensitive reduction in higher precision while storing linear weights more compactly.",
        "id": "numerical-dtype",
        "mlx_translation": "Set weight, activation, accumulation, and sensitive-operation dtypes deliberately.",
        "next_step": "Expose mutable state.",
        "plain_language": "Choose dtype and accumulation by the numerical role of each operation.",
        "proof_check": "Parity tolerances and task quality remain within the declared numerical policy.",
        "pytorch_cuda": "Global AMP or autocast may select mixed precision implicitly.",
        "title": "Autocast habits to numerical roles",
        "why_it_matters": "Norms, softmax, recurrence, FFTs, and quantizer distances can be more sensitive than linear layers."
      },
      {
        "common_trap": "Capturing mutable state inside a compiled function and reusing stale values.",
        "example": "Compare one-token KV growth and the next-token logits from the same prior state.",
        "id": "explicit-state",
        "mlx_translation": "Thread cache, recurrent, and compiled state explicitly and validate update order.",
        "next_step": "Compile only stable regions.",
        "plain_language": "Caches and recurrent values should be visible inputs and outputs of the ported computation.",
        "proof_check": "State shapes, positions, contents, and reset semantics pass staged comparisons.",
        "pytorch_cuda": "Modules may mutate buffers, KV caches, or recurrent fields internally.",
        "title": "Hidden mutation to explicit state",
        "why_it_matters": "Hidden mutation can make parity, compilation, and reuse incorrect or stale."
      },
      {
        "common_trap": "Compiling an entire dynamic pipeline before locating the measured bottleneck.",
        "example": "Compile a repeated denoiser step while leaving scheduler bookkeeping explicit.",
        "id": "stable-compilation",
        "mlx_translation": "Use mx.compile for a function with explicit inputs, outputs, and stable structure.",
        "next_step": "Escalate to custom code only when profiling justifies it.",
        "plain_language": "Compile pure, stable, reused MLX regions after eager parity.",
        "proof_check": "Compiled and eager outputs match and the target workload shows an end-to-end benefit.",
        "pytorch_cuda": "torch.compile and CUDA graphs optimize or capture framework execution under their own contracts.",
        "title": "torch.compile and CUDA graphs to mx.compile",
        "why_it_matters": "Changing shapes, dtypes, Python structure, or captured state can retrace or invalidate assumptions."
      },
      {
        "common_trap": "Porting a CUDA optimization before reproducing the source semantics it accelerates.",
        "example": "Replace compatible attention with MLX fast SDPA before designing an attention kernel.",
        "id": "custom-extensions",
        "mlx_translation": "Try native MLX operations, mx.fast, and compilation first; use custom Metal only after profiling.",
        "next_step": "Measure with explicit synchronization.",
        "plain_language": "Prefer readable MLX operations before custom backend work.",
        "proof_check": "The custom path matches the readable fallback and names a no-benefit rollback condition.",
        "pytorch_cuda": "A source project may depend on a CUDA extension or fused kernel.",
        "title": "CUDA extensions to native MLX or Metal",
        "why_it_matters": "A custom kernel adds correctness, portability, maintenance, and fallback obligations."
      },
      {
        "common_trap": "Comparing an eager source measurement with an unevaluated MLX expression.",
        "example": "Realize prefill logits before ending the prefill timer, then time cached decode separately.",
        "id": "timing-and-sync",
        "mlx_translation": "Evaluate outputs at the intended dependency or measurement boundary and avoid unrelated sync points.",
        "next_step": "Use framework bridges only as controlled oracle tools.",
        "plain_language": "Lazy scheduling means timing must force the work being measured.",
        "proof_check": "Repeated cold and warm runs account for the same work and state.",
        "pytorch_cuda": "CUDA measurements commonly use explicit synchronization or framework events.",
        "title": "CUDA timing to MLX evaluation boundaries",
        "why_it_matters": "A timer around graph construction can omit execution or charge it to the next phase."
      },
      {
        "common_trap": "Including conversion time in one runtime but not the other.",
        "example": "Export one source checkpoint tensor for a deterministic primitive comparison.",
        "id": "framework-bridges",
        "mlx_translation": "Use documented conversion paths and keep the bridge outside measured model regions.",
        "next_step": "Return to native MLX arrays for the actual port.",
        "plain_language": "NumPy and DLPack can move comparable values across framework boundaries.",
        "proof_check": "The receipt records copy, dtype, ownership, and synchronization behavior at the boundary.",
        "pytorch_cuda": "Convert source tensors to a neutral or exchange representation for comparison.",
        "title": "Framework bridges for oracle comparison",
        "why_it_matters": "They are useful for proof but can force evaluation, copy memory, or change ownership semantics."
      }
    ]
  },
  "local_docs": {
    "references": 35,
    "runbooks": 17
  },
  "schema_version": 1,
  "sources": {
    "by_classification": {
      "official_api_doc": 14,
      "primary_paper": 1,
      "primary_source_code": 13,
      "release_note": 2,
      "unclassified": 326
    },
    "by_kind": {
      "issue-report": 1,
      "official-doc": 38,
      "paper": 227,
      "release": 8,
      "repository": 49,
      "source-code": 26,
      "technical-blog": 7
    },
    "by_review_depth": {
      "indexed": 24,
      "screened": 190,
      "synthesized": 142
    },
    "by_support_scope": {
      "context_only": 1,
      "official_mlx": 14,
      "official_mlx_project": 4,
      "paper_only": 1,
      "third_party_pinned": 10,
      "unspecified": 326
    },
    "total": 356
  },
  "techniques": {
    "by_status": {
      "native-mlx": 15,
      "official-mlx-project": 9,
      "proven-mlx-port": 22,
      "rejected-or-superseded": 1,
      "research-candidate": 19
    },
    "total": 66
  },
  "version": "0.6.0"
};
