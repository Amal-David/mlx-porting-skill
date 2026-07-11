# Changelog

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
