#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptPath = fileURLToPath(import.meta.url);
const appRoot = path.resolve(path.dirname(scriptPath), "..");
const repoRoot = path.resolve(appRoot, "..", "..");
const skillRoot = path.join(repoRoot, "mlx-model-porting");
const assetsRoot = path.join(skillRoot, "assets");
const outputPath = path.join(appRoot, "src", "skill-data.generated.ts");
const args = new Set(process.argv.slice(2));

async function readJson(relativePath) {
  const absolutePath = path.join(skillRoot, relativePath);
  const text = await readFile(absolutePath, "utf8");
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${relativePath} is not JSON-compatible: ${error.message}`);
  }
}

function required(value, label) {
  if (value === undefined || value === null || value === "" || (Array.isArray(value) && value.length === 0)) {
    throw new Error(`Missing required field: ${label}`);
  }
  return value;
}

function flattenEvidenceRefs(refs) {
  if (!refs || typeof refs !== "object") {
    return [];
  }
  return Object.values(refs).flatMap((value) => (Array.isArray(value) ? value : []));
}

function sourceSummary(source) {
  return {
    id: required(source.id, "source.id"),
    title: required(source.title, `source.${source.id}.title`),
    url: required(source.url, `source.${source.id}.url`),
    kind: source.kind ?? "unknown",
    owner: source.owner ?? "",
    reviewDepth: source.review_depth ?? "",
    snapshot: source.snapshot ?? "",
    note: source.note ?? ""
  };
}

function advisorBucketForLearning(status) {
  if (status === "adopted") {
    return "benchmark-required";
  }
  if (status?.includes("insufficient") || status?.includes("research") || status?.includes("adjacent")) {
    return "experimental-approach";
  }
  return "validated-source-theory";
}

function advisorBucketForBacklog(status) {
  if (status === "validated") {
    return "validated-locally";
  }
  if (status === "rejected" || status === "closed") {
    return "rejected-do-not-use";
  }
  return "experimental-approach";
}

function speedupSlot(value) {
  const slot = value && typeof value === "object" ? value : {};
  return {
    range: slot.range ?? "",
    confidence: slot.confidence ?? "",
    basis: slot.basis ?? "",
    appliesWhen: Array.isArray(slot.applies_when) ? slot.applies_when : [],
    measure: Array.isArray(slot.measure) ? slot.measure : []
  };
}

function requiredSource(sourceById, sourceId, label) {
  const source = sourceById.get(sourceId);
  if (!source) {
    throw new Error(`Unknown source '${sourceId}' referenced by ${label}`);
  }
  return source;
}

async function buildData() {
  const [guidance, taxonomy, architectures, sources, contributorLearnings, researchBacklog, modelOutcomes, topModelsSnapshot] = await Promise.all([
    readJson("assets/optimization_guidance.yaml"),
    readJson("assets/recommendation-taxonomy.yaml"),
    readJson("assets/architectures.yaml"),
    readJson("assets/sources.yaml"),
    readJson("assets/contributor_learnings.json"),
    readJson("assets/research_backlog.json"),
    readJson("assets/model_outcomes.json"),
    readJson("assets/top_models_snapshot.json")
  ]);

  const sourceById = new Map((sources.sources ?? []).map((source) => [source.id, sourceSummary(source)]));
  const statusToBucket = taxonomy.status_to_advisor_bucket ?? {};

  const localAssetSources = [
    {
      id: "asset-optimization-guidance",
      title: "MLX optimization guidance registry",
      url: "mlx-model-porting/assets/optimization_guidance.yaml",
      kind: "local-file",
      owner: "mlx-porting-skill",
      reviewDepth: "synthesized",
      snapshot: guidance.reviewed ?? "",
      note: "Structured method, status, validation, rollback, and evidence references."
    },
    {
      id: "asset-contributor-learnings",
      title: "Top contributor implementation learnings",
      url: "mlx-model-porting/assets/contributor_learnings.json",
      kind: "local-file",
      owner: "mlx-porting-skill",
      reviewDepth: "synthesized",
      snapshot: contributorLearnings.reviewed ?? "",
      note: "Adopted and held findings from the MLX contributor research loop."
    },
    {
      id: "asset-research-backlog",
      title: "MLX porting research backlog",
      url: "mlx-model-porting/assets/research_backlog.json",
      kind: "local-file",
      owner: "mlx-porting-skill",
      reviewDepth: "synthesized",
      snapshot: researchBacklog.reviewed ?? "",
      note: "Validated and needs-validation gaps that should remain visible to the advisor."
    },
    {
      id: "asset-architectures",
      title: "Architecture family registry",
      url: "mlx-model-porting/assets/architectures.yaml",
      kind: "local-file",
      owner: "mlx-porting-skill",
      reviewDepth: "synthesized",
      snapshot: architectures.reviewed ?? "",
      note: "Machine-readable family aliases, runbooks, and route targets."
    },
    {
      id: "asset-model-outcomes",
      title: "Model outcome evidence registry",
      url: "mlx-model-porting/assets/model_outcomes.json",
      kind: "local-file",
      owner: "mlx-porting-skill",
      reviewDepth: "synthesized",
      snapshot: modelOutcomes.reviewed ?? "",
      note: "Source-backed records of what worked, what did not, and which claims remain benchmark-bound."
    },
    {
      id: "asset-top-models-snapshot",
      title: "Top Hugging Face model coverage snapshot",
      url: "mlx-model-porting/assets/top_models_snapshot.json",
      kind: "local-file",
      owner: "mlx-porting-skill",
      reviewDepth: "screened",
      snapshot: topModelsSnapshot.generated_at ?? "",
      note: "Generated top-model demand snapshot annotated with reviewed outcome ids."
    }
  ];

  for (const source of localAssetSources) {
    sourceById.set(source.id, source);
  }

  const methodSourceIds = new Set();
  const methods = (guidance.methods ?? []).map((method) => {
    const id = required(method.id, "method.id");
    const status = required(method.status, `method.${id}.status`);
    required(method.applies_to, `method.${id}.applies_to`);
    required(method.recommendation, `method.${id}.recommendation`);
    required(method.expected_effect, `method.${id}.expected_effect`);
    required(method.validation_gates, `method.${id}.validation_gates`);
    required(method.rollback_conditions, `method.${id}.rollback_conditions`);
    required(method.evidence_refs, `method.${id}.evidence_refs`);
    const evidenceSourceIds = flattenEvidenceRefs(method.evidence_refs);
    for (const sourceId of evidenceSourceIds) {
      if (!sourceById.has(sourceId)) {
        throw new Error(`Unknown evidence source '${sourceId}' referenced by method '${id}'`);
      }
      methodSourceIds.add(sourceId);
    }
    methodSourceIds.add("asset-optimization-guidance");
    return {
      id,
      techniqueId: method.technique_id ?? id,
      category: method.category ?? "uncategorized",
      status,
      advisorBucket: statusToBucket[status] ?? "experimental-approach",
      objectives: method.objectives ?? [],
      appliesTo: method.applies_to ?? [],
      recommendation: method.recommendation,
      expectedEffect: method.expected_effect,
      tradeoffs: method.tradeoffs ?? [],
      validationGates: method.validation_gates ?? [],
      rollbackConditions: method.rollback_conditions ?? [],
      evidenceRefs: method.evidence_refs,
      evidenceSourceIds
    };
  });

  const families = (architectures.families ?? []).map((family) => {
    required(family.id, "family.id");
    required(family.runbook, `family.${family.id}.runbook`);
    return {
      id: family.id,
      runbook: family.runbook,
      targets: family.targets ?? [],
      aliases: family.model_type_aliases ?? [],
      classPatterns: family.class_patterns ?? [],
      configSignals: family.config_signals ?? [],
      weightSignals: family.weight_signals ?? [],
      state: family.state ?? "",
      notes: family.notes ?? ""
    };
  });

  const learnings = (contributorLearnings.learnings ?? []).map((learning) => {
    required(learning.id, "learning.id");
    methodSourceIds.add("asset-contributor-learnings");
    return {
      id: learning.id,
      status: learning.status ?? "unknown",
      advisorBucket: advisorBucketForLearning(learning.status ?? ""),
      evidence: learning.evidence ?? [],
      summary: learning.porting_skill_change ?? learning.reason_held ?? "",
      validationGate: learning.validation_gate ?? "",
      rollbackCondition: learning.rollback_condition ?? "",
      reasonHeld: learning.reason_held ?? "",
      evidenceSourceIds: ["asset-contributor-learnings"]
    };
  });

  const backlogItems = (researchBacklog.items ?? []).map((item) => {
    required(item.id, "backlog.id");
    methodSourceIds.add("asset-research-backlog");
    return {
      id: item.id,
      priority: item.priority ?? "",
      status: item.status ?? "unknown",
      advisorBucket: advisorBucketForBacklog(item.status ?? ""),
      summary: item.summary ?? "",
      requiredGate: item.required_gate ?? "",
      affected: item.affected ?? [],
      source: item.source ?? "",
      evidenceSourceIds: ["asset-research-backlog"]
    };
  });

  for (const family of families) {
    methodSourceIds.add("asset-architectures");
  }

  const outcomeRecords = (modelOutcomes.records ?? []).map((record) => {
    const id = required(record.id, "outcome.id");
    required(record.status, `outcome.${id}.status`);
    required(record.summary, `outcome.${id}.summary`);
    required(record.potential_speedup, `outcome.${id}.potential_speedup`);
    required(record.source_ids, `outcome.${id}.source_ids`);
    for (const sourceId of record.source_ids ?? []) {
      if (!sourceById.has(sourceId)) {
        throw new Error(`Unknown source '${sourceId}' referenced by outcome '${id}'`);
      }
      methodSourceIds.add(sourceId);
    }
    methodSourceIds.add("asset-model-outcomes");
    return {
      id,
      label: record.label ?? id,
      status: record.status,
      summary: record.summary,
      worked: record.worked ?? [],
      didNotWork: record.did_not_work ?? [],
      claimBoundary: record.claim_boundary ?? "",
      potentialSpeedup: {
        overall: speedupSlot(record.potential_speedup?.overall),
        speculativeDecoding: speedupSlot(record.potential_speedup?.speculative_decoding)
      },
      match: record.match ?? {},
      sourceIds: record.source_ids ?? [],
      nextValidation: record.next_validation ?? ""
    };
  });
  methodSourceIds.add("asset-top-models-snapshot");
  methodSourceIds.add("hf-top-models-api-2026-06-29");

  const topModels = (topModelsSnapshot.models ?? []).map((model) => ({
    rank: model.rank ?? 0,
    id: model.id ?? "",
    downloads: model.downloads ?? 0,
    pipelineTag: model.pipeline_tag ?? "",
    libraryName: model.library_name ?? "",
    license: model.license ?? "",
    licenseClass: model.license_class ?? "",
    gated: Boolean(model.gated),
    matchedOutcomeIds: model.matched_outcome_ids ?? [],
    coverageState: model.coverage_state ?? "unknown"
  }));

  const data = {
    schemaVersion: 1,
    generatedFrom: {
      optimizationGuidanceReviewed: guidance.reviewed ?? "",
      taxonomyReviewed: taxonomy.reviewed ?? "",
      architecturesReviewed: architectures.reviewed ?? "",
      sourcesReviewed: sources.reviewed ?? "",
      contributorLearningsReviewed: contributorLearnings.reviewed ?? "",
      researchBacklogReviewed: researchBacklog.reviewed ?? "",
      modelOutcomesReviewed: modelOutcomes.reviewed ?? "",
      topModelsSnapshotGeneratedAt: topModelsSnapshot.generated_at ?? ""
    },
    taxonomy: {
      advisorBuckets: taxonomy.advisor_buckets ?? [],
      statusToAdvisorBucket: statusToBucket,
      defaultKeepGate: taxonomy.default_keep_gate ?? []
    },
    families,
    methods,
    learnings,
    backlogItems,
    modelOutcomes: {
      statusDefinitions: modelOutcomes.status_definitions ?? {},
      claimPolicy: modelOutcomes.claim_policy ?? [],
      coverageTarget: modelOutcomes.coverage_target ?? {},
      records: outcomeRecords
    },
    topModelsSnapshot: {
      source: topModelsSnapshot.source ?? "",
      generatedAt: topModelsSnapshot.generated_at ?? "",
      modelCount: topModelsSnapshot.model_count ?? topModels.length,
      coveredCount: topModelsSnapshot.covered_count ?? 0,
      unknownCount: topModelsSnapshot.unknown_count ?? 0,
      models: topModels
    },
    sources: [...methodSourceIds].sort().map((sourceId) => requiredSource(sourceById, sourceId, "generated advisor data"))
  };

  return data;
}

function renderModule(data) {
  return [
    "/* Generated by scripts/build-skill-data.mjs. Do not edit by hand. */",
    `export const ADVISOR_DATA = ${JSON.stringify(data, null, 2)} as const;`,
    ""
  ].join("\n");
}

const data = await buildData();
const rendered = renderModule(data);

if (args.has("--check")) {
  const current = await readFile(outputPath, "utf8");
  if (current !== rendered) {
    throw new Error(`${path.relative(repoRoot, outputPath)} is out of date. Run npm run build:data from ${path.relative(repoRoot, appRoot)}.`);
  }
  console.log(`ok ${path.relative(repoRoot, outputPath)} is current`);
} else {
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, rendered, "utf8");
  console.log(`wrote ${path.relative(repoRoot, outputPath)}`);
}
