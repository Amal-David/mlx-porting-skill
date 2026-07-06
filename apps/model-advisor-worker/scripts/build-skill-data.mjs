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
const bandRangePattern = /^(\d+(?:\.\d+)?)x-(\d+(?:\.\d+)?)x$/;
const stackLossiness = new Set(["lossless", "conditionally-lossy"]);
const stackPairValidity = new Set(["validated-composable", "unknown", "known-conflicting"]);

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

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
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

function parseBandRange(value, label) {
  const range = required(value, label);
  if (typeof range !== "string") {
    throw new Error(`${label} must be a string`);
  }
  const match = bandRangePattern.exec(range);
  if (!match) {
    throw new Error(`${label} must look like '1.0x-4.3x'`);
  }
  const floor = Number.parseFloat(match[1]);
  const ceiling = Number.parseFloat(match[2]);
  if (!Number.isFinite(floor) || !Number.isFinite(ceiling) || floor <= 0 || ceiling <= 0 || floor > ceiling) {
    throw new Error(`${label} has invalid multiplier bounds`);
  }
  return { range, floorText: match[1] };
}

function normalizeImprovementBand(value, methodId, provenanceValues) {
  if (value === undefined || value === null) {
    return null;
  }
  if (!isRecord(value)) {
    throw new Error(`method.${methodId}.improvement_band must be an object`);
  }
  const provenance = required(value.provenance, `method.${methodId}.improvement_band.provenance`);
  if (!provenanceValues.has(provenance)) {
    throw new Error(`method.${methodId}.improvement_band.provenance has unknown value '${provenance}'`);
  }
  const { range, floorText } = parseBandRange(value.range, `method.${methodId}.improvement_band.range`);
  if (provenance !== "local_reproduced" && floorText !== "1.0") {
    throw new Error(`method.${methodId}.improvement_band.range must start at 1.0x for ${provenance}`);
  }
  required(value.metric, `method.${methodId}.improvement_band.metric`);
  required(value.basis, `method.${methodId}.improvement_band.basis`);
  required(value.applies_when, `method.${methodId}.improvement_band.applies_when`);
  if (value.measured_on !== undefined && !isRecord(value.measured_on)) {
    throw new Error(`method.${methodId}.improvement_band.measured_on must be an object when present`);
  }
  if (provenance === "local_reproduced" && (!isRecord(value.measured_on) || Object.keys(value.measured_on).length === 0)) {
    throw new Error(`method.${methodId}.improvement_band.measured_on is required for local_reproduced bands`);
  }
  if (value.receipts !== undefined && !Array.isArray(value.receipts)) {
    throw new Error(`method.${methodId}.improvement_band.receipts must be an array when present`);
  }
  if (provenance === "local_reproduced" && !Array.isArray(value.receipts)) {
    throw new Error(`method.${methodId}.improvement_band.receipts is required for local_reproduced bands`);
  }
  return {
    provenance,
    range,
    metric: value.metric,
    basis: value.basis,
    appliesWhen: value.applies_when,
    measuredOn: value.measured_on ?? null,
    receipts: value.receipts ?? []
  };
}

function compoundHasNumericField(value) {
  if (Array.isArray(value)) {
    return value.some((item) => compoundHasNumericField(item));
  }
  if (isRecord(value)) {
    return Object.entries(value).some(([key, item]) => key.includes("range") || compoundHasNumericField(item));
  }
  return typeof value === "number" && Number.isFinite(value);
}

function normalizeStack(stack, methodIds, familyIds) {
  const id = required(stack.id, "stack.id");
  const familyValues = required(stack.families, `stack.${id}.families`);
  if (!Array.isArray(familyValues)) {
    throw new Error(`stack.${id}.families must be an array`);
  }
  for (const familyId of familyValues) {
    if (!familyIds.has(familyId)) {
      throw new Error(`stack.${id}.families references unknown family '${familyId}'`);
    }
  }

  const rawSteps = required(stack.steps, `stack.${id}.steps`);
  if (!Array.isArray(rawSteps)) {
    throw new Error(`stack.${id}.steps must be an array`);
  }
  const stepMethodIds = new Set();
  const steps = rawSteps.map((step, index) => {
    if (!isRecord(step)) {
      throw new Error(`stack.${id}.steps[${index}] must be an object`);
    }
    const methodId = required(step.method, `stack.${id}.steps[${index}].method`);
    if (!methodIds.has(methodId)) {
      throw new Error(`stack.${id}.steps[${index}].method references unknown method '${methodId}'`);
    }
    if (!stackLossiness.has(step.lossiness)) {
      throw new Error(`stack.${id}.steps[${index}].lossiness has invalid value '${step.lossiness}'`);
    }
    required(step.gate, `stack.${id}.steps[${index}].gate`);
    required(step.rollback, `stack.${id}.steps[${index}].rollback`);
    stepMethodIds.add(methodId);
    return {
      method: methodId,
      lossiness: step.lossiness,
      gate: step.gate,
      rollback: step.rollback
    };
  });

  const rawNotes = stack.composition_notes ?? [];
  if (!Array.isArray(rawNotes)) {
    throw new Error(`stack.${id}.composition_notes must be an array`);
  }
  const compositionNotes = rawNotes.map((note, index) => {
    if (!isRecord(note)) {
      throw new Error(`stack.${id}.composition_notes[${index}] must be an object`);
    }
    const pair = required(note.pair, `stack.${id}.composition_notes[${index}].pair`);
    if (!Array.isArray(pair) || pair.length !== 2) {
      throw new Error(`stack.${id}.composition_notes[${index}].pair must contain two method ids`);
    }
    for (const methodId of pair) {
      if (!stepMethodIds.has(methodId)) {
        throw new Error(`stack.${id}.composition_notes[${index}].pair references method outside stack: '${methodId}'`);
      }
    }
    if (!stackPairValidity.has(note.validity)) {
      throw new Error(`stack.${id}.composition_notes[${index}].validity has invalid value '${note.validity}'`);
    }
    required(note.why, `stack.${id}.composition_notes[${index}].why`);
    return {
      pair,
      validity: note.validity,
      why: note.why
    };
  });

  const compound = required(stack.compound, `stack.${id}.compound`);
  if (!isRecord(compound)) {
    throw new Error(`stack.${id}.compound must be an object`);
  }
  const compoundKeys = Object.keys(compound).sort();
  if (compoundKeys.join(",") !== "measured_together,receipts") {
    throw new Error(`stack.${id}.compound may only contain measured_together and receipts`);
  }
  if (typeof compound.measured_together !== "boolean") {
    throw new Error(`stack.${id}.compound.measured_together must be a boolean`);
  }
  if (!Array.isArray(compound.receipts)) {
    throw new Error(`stack.${id}.compound.receipts must be an array`);
  }
  if (compoundHasNumericField(compound)) {
    throw new Error(`stack.${id}.compound must not store numeric ranges`);
  }

  return {
    id,
    label: stack.label ?? id,
    families: familyValues,
    steps,
    compositionNotes,
    compound: {
      measured_together: compound.measured_together,
      receipts: compound.receipts
    },
    evidenceSourceIds: ["asset-optimization-stacks"]
  };
}

async function buildData() {
  const [guidance, optimizationStacks, taxonomy, architectures, sources, contributorLearnings, researchBacklog, modelOutcomes, topModelsSnapshot] = await Promise.all([
    readJson("assets/optimization_guidance.yaml"),
    readJson("assets/optimization_stacks.yaml"),
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
      id: "asset-optimization-stacks",
      title: "MLX optimization stack registry",
      url: "mlx-model-porting/assets/optimization_stacks.yaml",
      kind: "local-file",
      owner: "mlx-porting-skill",
      reviewDepth: "synthesized",
      snapshot: optimizationStacks.reviewed ?? "",
      note: "Structured stack recipes, composition notes, and compound-measurement receipts."
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
  const improvementBandProvenance = new Set(Object.keys(taxonomy.improvement_band_policy ?? {}));
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
    const improvementBand = normalizeImprovementBand(method.improvement_band, id, improvementBandProvenance);
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
      evidenceSourceIds,
      improvementBand
    };
  });
  const methodIds = new Set(methods.map((method) => method.id));

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
  const familyIds = new Set(families.map((family) => family.id));

  const stacks = required(optimizationStacks.stacks, "optimization_stacks.stacks").map((stack) => normalizeStack(stack, methodIds, familyIds));
  methodSourceIds.add("asset-optimization-stacks");

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
      optimizationStacksReviewed: optimizationStacks.reviewed ?? "",
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
    stacks,
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
    sources: [...methodSourceIds].sort().map((sourceId) => sourceById.get(sourceId))
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
