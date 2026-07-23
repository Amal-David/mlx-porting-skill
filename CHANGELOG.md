# Changelog

## 0.7.0 — 2026-07-23

- Completed 7/7 architecture-class coverage: every stress-tested family now has
  a checked-in proof, anchored by the verified Granite-MoE-1B MLX port that
  closes the sparse-MoE class. The README, validation status, site, and
  `llms.txt` narrative were rewritten around this coverage and the four
  `optimize-receipts/` observations, which record why the structured 8-bit pick
  held quality where the naive 4-bit default failed the gate. These remain local
  single-Mac, single-workload observations, not promoted benchmark claims.
- Landed the 2026-07-18 and 2026-07-20 nightly knowledge-curator runs as new
  dated research-run directories and retained bounded prior candidates instead
  of dropping them on refresh; the reviewed graph now holds 712 nodes and 505
  edges with the reconciled backlog and drift check still green.
- Added `.github/workflows/deploy-site.yml`: on a push to `main` touching
  `site/` or `VERSION`, it deploys the static site to Cloudflare Pages with a
  pinned Wrangler and then polls the live `data.js` until it serves the released
  version, failing loudly if the deploy never goes live.
- Added `.github/workflows/release.yml`: when `VERSION` changes on `main`, it
  cuts the annotated `vX.Y.Z` tag and publishes a GitHub Release whose body is
  the matching `CHANGELOG.md` section, and no-ops when the tag already exists.
- Added `.github/dependabot.yml` for GitHub Actions and the `.github/` CI pip
  requirements, both grouped, PR-limited, and gated by a 7-day supply-chain
  cooldown before a freshly published release can be proposed.
- Extended CI: a Python 3.13 lane joins 3.10 and 3.14 in the supported-Python
  matrix (with its hash-locked NumPy 2.5.1 wheel), and the release job now
  smoke-installs `mlx-model-porting/requirements-tools.lock` into a throwaway
  venv under `--require-hashes --only-binary=:all:` so the runtime lockfile is
  proven installable.
- Taught `install_skill.py` an `antigravity` client preset (the shared
  `.agents/skills` symlink root) and a `--check` mode that reports whether an
  installed skill's SKILL.md version matches `VERSION` without modifying
  anything, returning non-zero on a stale or missing install.
- Fixed a manifest time-of-check/time-of-use race: `manifest.py` now excludes
  the transient `.converted-*.safetensors` atomic-write staging files that
  `convert_checkpoint.py` creates and unlinks in place, so a `check` that races
  an in-progress conversion no longer tries to hash a file being written.
  Because these files are transient by construction and never distributed,
  skipping them cannot hide drift in a shipped file.
- Expanded the SKILL.md trigger map to route high-demand families directly:
  vision-language/multimodal, diffusion/flow, text-to-speech, ASR/streaming
  speech, sparse-MoE, and selective-SSM/hybrid inputs now each point at their
  runbooks, keeping the body under the audited size ceiling.
- Added `tests/test_corpus_count_contract.py`, which discovers the real suite
  size in-process and fails loud when README.md or VALIDATION.md advertise a
  stale offline-test count, and a repository `conftest.py`. Corrected the
  published counts across README, VALIDATION, and the research report to the
  discovered 551 offline tests and the current 712-node / 505-edge graph.

### Deferred follow-ups

These deep-research gaps were identified during the release audit. Admitting
them to `research_backlog.json` requires new `backlog_item` graph nodes and a
graph regeneration (a new dated run), so they are tracked here for a future
release rather than hand-written into the reconciled backlog:

- an iOS/Swift on-device MLX deployment reference;
- a distributed / multi-Mac inference reference (`mx.distributed`);
- a runnable serving harness covering the `mlx-lm` server workflow;
- promotion of one stress-tested class (VLM or MoE) into the executable scaffold
  generator with an in-payload worked example; and
- a training/LoRA parity fixture, which today exists only as a contract.

## 0.6.1 — 2026-07-14

- Added four evidence-gated research candidates from the Ideogram V4 serving
  write-ups and primary papers: DSpark confidence-scheduled speculation,
  diffusion quantization-aware distillation, classifier-free-guidance branch
  distillation, and few-step diffusion/flow distillation.
- Added a diffusion cost model that separates denoising steps, branches per
  step, and cost per branch, with explicit quality gates and a prohibition on
  multiplying unmeasured gains.
- Added MLX-specific low-bit follow-through: re-profile norms, activations,
  casts, layouts, and materialized intermediates after accelerating a matmul;
  prefer native fast operations and stable-region compilation; and never
  translate CUTLASS epilogue or Blackwell tile schedules directly to Metal.
- Preserved every fal B200/NVFP4/CUTLASS performance number as context-only
  evidence. No new numeric claim, supported-MLX status, optimization method, or
  compound stack was promoted.
- Expanded the corpus to 361 sources and 70 techniques and raised the offline
  suite to 473 tests with focused provenance, status, gate, rollback, and
  hardware-boundary contracts.

## 0.6.0 — 2026-07-12

- Rebuilt the public documentation as a learning-first study guide covering
  MLX fundamentals, what a port preserves, architecture/modality recognition,
  PyTorch and CUDA translation lenses, the ordered parity rail, honest
  benchmarking, publication boundaries, and an expanded glossary before the
  executable field manual.
- Added four guided model journeys over one canonical eight-checkpoint spine:
  the pinned Qwen2.5-0.5B-Instruct proof remains `proven`, while Whisper-style
  ASR, FLUX-style diffusion, and LLaVA-style VLM remain explicitly labelled
  simulations with model-specific proof boundaries and runbooks.
- Added a dependency-free interactive porting atlas with URL/history state,
  keyboard traversal, comparison mode, readable text export, responsive
  vertical stepping, complete no-JavaScript fallback, and fail-closed malformed
  data handling.
- Replaced the landing-page status graphic with a pausable mathematical proof
  loop—classify, translate, prove parity, profile—and equivalent static mobile
  and reduced-motion frames.
- Added a parity- and profile-gated optimization atlas covering all 28 methods
  across eight measured-bottleneck families. Research requires explicit
  opt-in, rejected paths stay unavailable, lossy methods carry canonical
  quality gates, and current numeric outcomes remain withheld or absent.
- Connected optimization hypotheses to the selected model journey, preserving
  journey applicability, proof boundaries, canonical sources, real runbooks,
  tested commands, rollback conditions, focus, and browser Back/Forward state
  in the expert handoff.
- Hardened generated learning data against indexed nightly evidence, unsafe or
  missing links, advisor/status drift, malformed family and journey metadata,
  unscoped future local-promotion ranges, and status-incompatible canonical
  source selection.
- Added the official PyTorch CUDA Graphs semantics source to document why CUDA
  capture/replay is NVIDIA-specific and why stable MLX compilation is the
  portable branch; the corpus now contains 356 sources and a 703-node,
  501-edge review graph.
- Documented the curriculum and optimization-atlas contributor flow and raised
  the deterministic offline suite to 468 tests.
- Reconciled the durable nightly-curator content already present on `main` and
  regenerated the current graph, backlog, evidence index, site data, and
  release manifest without importing the stale conflicting snapshot from the
  closed nightly PR.
- Expanded the executable scaffolder from one to six architecture families:
  dense decoder plus BERT encoder, T5 encoder-decoder, HuBERT/Wav2Vec2 ASR
  acoustic encoder, sparse MoE decoder, and selective SSM.
- Added three checked-in real checkpoint ports beyond Qwen2.5:
  BAAI/bge-base-en, t5-small, and facebook/hubert-base-ls960. Each includes a
  pinned inspection/conversion/parity packet without model weights.
- Added synthetic correctness gates for sparse MoE routing/experts and the
  minimal selective-SSM recurrence. These do not claim real-checkpoint support.
- Unified source capture, MLX capture, and parity under explicit
  `dense-decoder` (default), `encoder`, `encoder-decoder`, `ssm`, and `asr`
  modes while preserving dense-decoder behavior.
- Reconciled fail-closed config identity, numeric-zero handling, decoder flags,
  inspected tensor topology, conversion attestations, and the Whisper identity
  guard across all generated families.
- Kept the evidence boundary sealed: exact-output parity is the only built-in
  task metric, and all 10 numeric claims remain withheld with 0
  promotion-ready receipts.

## 0.5.0 — 2026-07-11

- Added an executable dense-decoder porting chain: `capture_oracle.py` records
  a pinned Torch oracle; `scaffold_port.py` generates a fail-closed eager MLX
  package; `convert_checkpoint.py` applies explicit schema-2 weight maps;
  `capture_mlx.py` records matching MLX tensors; and `run_parity.py` runs both
  captures plus the ordered first-divergence parity ladder in one command.
- Hardened the dense-decoder scaffold for inactive sliding-window metadata and
  independently detected Q/K/V/O projection biases without weakening its
  fail-closed architecture boundary.
- Added the first checked-in real end-to-end port packet for pinned
  Qwen2.5-0.5B-Instruct: 290 tensors converted, all 29 parity rungs passed,
  exact eight-token greedy output matched Torch, standalone MLX, and offline
  MLX-LM, and no model weights or NPZ captures were committed.
- Added the repository-owned `attested-mlx-port-wall-time` runner lane with
  per-run challenges, checked-in runner and dependency evidence, exact-output
  quality recomputation, and reproducibility-on-request evidence. The lane is
  resealed because unkeyed digests establish internal consistency, not
  authenticity; promotion now requires an external signer and trust root. The
  13 receipts classify as 12 `performance_observation`, 0 `promotion_ready`,
  and 1 `rejected`.
- Withheld all 10 effective claims, including `bf16-weight-cast`. The raw
  `1.8122003933x` inverse-wall-time ratio is one reproducible local-run observation, not a claim of reliable speedup. It remains evidence for the captured Qwen
  load-plus-six-token workload, not an effective range, portable guarantee, or
  pure-decode claim.
- Wired `knowledge_graph.json` into the advisor as a bounded, fail-closed,
  separate unreviewed-research-signals queue; reconciled
  `research_backlog.json` deterministically with a drift check; and removed the
  unused top-model popularity collector. The reviewed graph contains 697 nodes
  and 499 edges.
- Expanded the canonical corpus to 66 techniques and 28 guidance methods while
  preserving 17 architecture routes, 350 sources, and 4 planning stacks. The
  tool directory now contains 29 inspectable Python scripts.
- Added asset-consumer coverage and raised the offline suite to 425 tests,
  including dense-decoder scaffold, conversion, capture, parity, attestation,
  catalogue, backlog, security, and generated-artifact contracts.
- Clarified the capability boundary: dense-decoder Transformers are tooled end
  to end and proven by one real model port; the other 16 families have tooled
  intake, routing, planning, generic validation, and evidence gates, but their
  architecture-module implementations remain runbook-guided. Domain quality
  evaluators beyond exact-output parity remain future work.
- Documented the deterministic backlog reconcile/check step, regenerated site
  data from canonical inputs, and refreshed public documentation, adapters,
  validation truth, and the final manifest for version 0.5.0.
- Preserved the earlier DNS-rebinding, benchmark-ratio integrity,
  evidence-index sanitization, crash-safe installation, retired-surface,
  timeout, and generated-drift guards.

## 0.4.0 — 2026-07-10

- Retired the public model-advisor Worker and removed the runtime/deployment
  package; the documentation site is now static, offline-loadable, and derived
  from repository data.
- Expanded architecture routing to 17 synthetic golden families, added
  confidence and margin gates for ambiguous inputs, and made known hybrid
  models emit every required component runbook.
- Reworked recommendation matching around exact controlled family, model-type,
  capability, workload, objective, and TargetProfile identifiers.
- Added five advisor buckets (`validated-locally`,
  `validated-source-theory`, `benchmark-required`, `experimental-approach`, and
  `rejected-do-not-use`) with explicit opt-in for experimental execution.
- Made the generated effective-claim catalogue the sole numeric authority and
  blocked custom assessment sidecars, duplicate lineage, regressions, and
  unvalidated stack multiplication from manufacturing a claim.
- Preserved every source-reported benchmark range as a held observation only;
  source numbers cannot enter advice until an equivalent local target-workload
  receipt set passes every promotion gate.
- Added schema-2 benchmark contracts with controlled MLX-LM runner semantics
  and a family-neutral external-command adapter that accepts only a safe exact
  argv template bound to a digest-pinned runner, model, workload, and variant.
  Parent-measured wall time, bounded raw reports, pinned target/source lineage,
  built-in exact-output quality recomputation, compatible baselines,
  stability/noise gates, and rollback are required; child-reported metrics,
  arbitrary external JSON scores, and contributor-selected thresholds remain
  observation-only.
- Isolated external receipt Python with `-I -B`, statically rejected symlink components,
  froze quality contracts before execution, and moved revision/baseline failures
  ahead of measured work.
- Added a fail-closed `execution_attested` gate after proving that arbitrary
  generic runners and imported MLX-LM modules could fabricate otherwise valid
  receipts; both lanes remain observation-only pending reviewed built-in adapters.
- Bound every future local numeric promotion to one canonical experiment
  fingerprint from verified receipt through assessment, claim, stack, and
  TargetProfile matching; heterogeneous candidates cannot be pooled and
  compatible repetitions require one stable baseline/protocol/quality identity
  and use the conservative minimum receipt's complete fingerprint.
- Reclassified the 11 checked-in receipts as 10
  `performance_observation`, 0 `promotion_ready`, and 1 `rejected`; removed
  public wording that presented historical ratios as reliable wins.
- Added deterministic benchmark assessments, receipt index, benchmark report,
  effective-claim catalogue, evidence index, and static-site data with drift
  checks and documented ownership.
- Expanded the canonical corpus to 350 live sources, 65 techniques, 27 guidance
  methods, and 4 planning stacks with explicit review depth, claim boundaries,
  and controlled `support_scope` values including `third_party_pinned`.
- Hardened research campaigns, static inspection, archive handling, bounded
  traversal and structured input, subprocess cleanup and output capture,
  authenticated network origin/redirect handling, atomic writes, installation,
  manifest generation, and daily review-only automation.
- Made model intake reject partial/corrupt shards and bound ONNX, GGUF, Hub
  metadata, artifact count, and byte budgets; local paths are portable by
  default and require explicit opt-in to expose.
- Made local intake hash the complete inventoried tree, bind license evidence to
  that identity, protect model files from report-output overwrite, and clear all
  recommendation routes when identity or license evidence is incomplete.
- Made recommendation and port-plan inputs fail closed on incomplete or forged
  inspection reports; actionable plans require the original artifact root,
  rerun static inspection, and recompute the complete recommendation report.
  Blocked plans contain remediation only, and hybrid overrides cannot drop or
  invent routed families.
- Restricted default GitHub credentials to the canonical API origin, enforced
  public-HTTPS live collector bases, and redacted reflected HTTP error bodies.
- Made the distribution manifest descriptor-relative and mode-aware, rejected
  special filesystem nodes, and changed copy installation to an exact
  manifest-attested allowlist with an in-payload Apache-2.0 license.
- Added digest-pinned secret scanning, dependency-free repository scanning,
  live-source SSRF guards, full Python 3.10/3.12/3.14 suites, and a separate
  read-only source-health workflow.
- Added distribution portability, worker-retirement, site, evidence,
  recommendation, benchmark-promotion, claim-catalogue, routing, workflow, and
  adversarial hardening contracts.
- Rebuilt the README, validation status, research report, contribution guide,
  and public site around the actual architecture and extension flows and made
  the absence of a generic arbitrary-model converter explicit.
- Reconciled the conflicting July 8 nightly-curator PR by preserving its
  immutable scaffold-only run and refreshed graph snapshots while retaining the
  newer site, manifest, lessons, and stricter portability implementation. The
  final review-only candidate snapshot was refreshed on July 10 (including
  MLX-Audio v0.4.5 as unpromoted evidence), and superseded methods can no longer
  reappear as fresh curator leads.
- Applied a path-only privacy redaction to 11 older June 27/July 7 research
  artifacts, replacing machine-specific checkout prefixes with `<repo-root>`;
  findings, sources, timestamps, command meaning, claims, and outcomes are
  unchanged. All 28 July 8 nightly files remain byte-identical to their PR.

## 0.3.0 — 2026-07-08

- Added inspector mode for existing local MLX projects and already-running MLX ports.
- Added a static project inspector that reports MLX runtime surface, proof gaps, improvement opportunities, and contribution candidates without executing target code.
- Added docs and site diagrams that explain the porting pipeline, inspector loop, parity ladder, proof packet, and optimization decision path.

## 0.2.0 — 2026-07-06

- Corrected misidentified Orca, HQQ, and Prompt Lookup evidence sources.
- Replaced the broken broad MLX core API citation with concrete MLX operator docs.
- Added provenance validation for source IDs, snapshots, URL checks, and supported-technique evidence.
- Scoped third-party MLX optimization claims and indexed June 2026 KV-cache research as research candidates.
- Improved skill activation with trigger-rich frontmatter, a When-to-use section, a Trigger map, checked-in Claude/Codex discovery symlinks, installer client presets, audit guard rails, and AGENTS/CLAUDE routing pointers.
- Added improvement observations and four planning stacks; 0.4.0 later replaced
  prose authority and derived compound output with the generated claim
  catalogue and strict composition gates.
- Added the benchmark receipt harness and local Apple M4 Pro receipts for 4-bit weights, prompt cache, KV quantization, draft-model speculation, and the measured-together dense-decoder stack.
- Hardened model inspection, tensor comparison, weight-map validation, template warnings, and benchmark timeout handling.

## 0.1.0 — 2026-06-23

- Initial portable Agent Skill for architecture-aware MLX and MLX-Audio porting
  guidance.
- Fourteen architecture-family runbooks and a 49-technique decision registry.
- Evidence registry with explicit `synthesized`, `screened`, and `indexed` review depth.
- Static model inspection, architecture routing, deterministic weight-map validation, tensor parity, benchmark, installation, audit, and daily candidate tools.
- Review-only GitHub Actions update workflow and synthetic offline integration tests.
