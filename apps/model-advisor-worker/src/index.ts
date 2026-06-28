import { ADVISOR_DATA } from "./skill-data.generated";

export interface Env {
  HF_API_BASE?: string;
  OPENAI_API_KEY?: string;
  OPENAI_MODEL?: string;
}

type AdvisorData = typeof ADVISOR_DATA;
type Family = AdvisorData["families"][number];
type Method = AdvisorData["methods"][number];
type Source = AdvisorData["sources"][number];
type Bucket = AdvisorData["taxonomy"]["advisorBuckets"][number];

type HuggingFaceModel = {
  id?: string;
  modelId?: string;
  pipeline_tag?: string;
  library_name?: string;
  tags?: string[];
  downloads?: number;
  likes?: number;
  gated?: boolean | string;
  private?: boolean;
  createdAt?: string;
  lastModified?: string;
  cardData?: Record<string, unknown>;
  siblings?: Array<{ rfilename?: string; size?: number }>;
  config?: Record<string, unknown>;
  transformersInfo?: Record<string, unknown>;
};

const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store"
};

const HTML_HEADERS = {
  "content-type": "text/html; charset=utf-8",
  "cache-control": "no-store"
};

const BUCKET_ORDER = [
  "validated-locally",
  "validated-source-theory",
  "benchmark-required",
  "experimental-approach",
  "rejected-do-not-use"
];

const MAX_SEARCH_QUERY_LENGTH = 96;
const MAX_MODEL_ID_LENGTH = 220;
const MAX_JSON_BODY_LENGTH = 4096;
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMITS: Record<string, number> = {
  search: 60,
  model: 40,
  advice: 40,
  "ai-summary": 8
};
const rateCounters = new Map<string, { count: number; resetAt: number }>();

const TASK_FAMILY_HINTS: Array<{ tasks: string[]; family: string; reason: string }> = [
  { tasks: ["text-generation", "conversational"], family: "dense-decoder-transformer", reason: "text generation task" },
  { tasks: ["text2text-generation", "summarization", "translation"], family: "encoder-decoder-transformer", reason: "sequence-to-sequence task" },
  { tasks: ["fill-mask", "text-classification", "token-classification", "sentence-similarity", "feature-extraction"], family: "encoder-transformer", reason: "encoder-style NLP task" },
  { tasks: ["image-classification", "zero-shot-image-classification"], family: "non-generative-cv-backbone", reason: "vision classification task" },
  { tasks: ["image-to-text", "image-text-to-text", "visual-question-answering", "document-question-answering"], family: "vision-language-omni", reason: "vision-language task" },
  { tasks: ["text-to-image", "image-to-image", "unconditional-image-generation", "text-to-video", "image-to-video"], family: "diffusion-flow", reason: "diffusion or generative vision task" },
  { tasks: ["automatic-speech-recognition"], family: "automatic-speech-recognition", reason: "speech recognition task" },
  { tasks: ["text-to-speech", "text-to-audio"], family: "flow-diffusion-tts", reason: "text-to-speech task" },
  { tasks: ["audio-to-audio"], family: "separation-enhancement", reason: "audio transformation task" },
  { tasks: ["time-series-forecasting"], family: "time-series-forecasting", reason: "time-series forecasting task" },
  { tasks: ["graph-ml"], family: "graph-message-passing", reason: "graph model task" }
];

const FAMILY_OVERRIDES: Array<{ tokens: string[]; family: string; reason: string }> = [
  { tokens: ["mixtral", "qwen3_moe", "qwen2_moe", "deepseek_v2", "deepseek_v3", "dbrx", "grok", "moe"], family: "moe-decoder-transformer", reason: "MoE alias or tag" },
  { tokens: ["mamba", "rwkv", "jamba", "zamba", "recurrentgemma", "state-space"], family: "ssm-recurrent-hybrid", reason: "SSM or recurrent alias" },
  { tokens: ["llava", "qwen2_vl", "qwen3_vl", "idefics", "paligemma", "pixtral", "vision-language", "multimodal"], family: "vision-language-omni", reason: "vision-language alias or tag" },
  { tokens: ["stable_diffusion", "stable-diffusion", "flux", "sd3", "dit", "unet", "diffusers"], family: "diffusion-flow", reason: "diffusion alias or library" },
  { tokens: ["encodec", "snac", "dac", "mimi", "wavtokenizer"], family: "neural-audio-codec", reason: "audio codec alias" },
  { tokens: ["bark", "fish-speech", "fish_speech", "moshi", "zonos", "chatterbox"], family: "autoregressive-audio-lm", reason: "autoregressive audio alias" },
  { tokens: ["f5-tts", "f5_tts", "matcha", "cosyvoice", "naturalspeech"], family: "flow-diffusion-tts", reason: "flow or diffusion TTS alias" },
  { tokens: ["vocos", "hifigan", "bigvgan", "vocoder"], family: "vocoder-waveform-decoder", reason: "vocoder alias" },
  { tokens: ["whisper", "wav2vec2", "parakeet", "canary", "voxtral", "asr"], family: "automatic-speech-recognition", reason: "ASR alias" },
  { tokens: ["streaming", "realtime", "rnnt", "transducer"], family: "streaming-speech", reason: "streaming speech alias" },
  { tokens: ["demucs", "roformer", "audio-separator", "enhancement"], family: "separation-enhancement", reason: "separation or enhancement alias" },
  { tokens: ["chronos", "timesfm", "patchtst", "time-series"], family: "time-series-forecasting", reason: "time-series alias" },
  { tokens: ["graphsage", "graph-sage", "gcn", "gat", "mpnn"], family: "graph-message-passing", reason: "graph message-passing alias" }
];

const APPLIES_TO_FAMILY_ALIASES: Record<string, string[]> = {
  "linear-heavy-models": [
    "dense-decoder-transformer",
    "moe-decoder-transformer",
    "encoder-transformer",
    "encoder-decoder-transformer",
    "vision-language-omni",
    "diffusion-flow",
    "time-series-forecasting"
  ],
  "audio-lm": ["autoregressive-audio-lm"],
  tts: ["autoregressive-audio-lm", "flow-diffusion-tts", "vocoder-waveform-decoder"],
  stt: ["automatic-speech-recognition", "streaming-speech"],
  "speech-to-speech": ["streaming-speech", "separation-enhancement"]
};

class HttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      const url = new URL(request.url);
      if (request.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: corsHeaders(request) });
      }
      if (url.pathname === "/") {
        return new Response(renderAppHtml(), { headers: HTML_HEADERS });
      }
      enforceRateLimit(request, url);
      if (url.pathname === "/api/search") {
        return json(await handleSearch(url, env), 200, request);
      }
      if (url.pathname === "/api/model") {
        return json(await handleModel(url, env), 200, request);
      }
      if (url.pathname === "/api/advice") {
        return json(await handleAdvice(request, url, env), 200, request);
      }
      return json({ error: "not_found" }, 404, request);
    } catch (error) {
      const status = error instanceof HttpError ? error.status : 500;
      return json({ error: error instanceof Error ? error.message : "unknown_error" }, status, request);
    }
  }
};

function corsHeaders(request: Request): Record<string, string> {
  const origin = request.headers.get("origin");
  const headers: Record<string, string> = {
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "content-type",
    "vary": "origin"
  };
  if (origin === new URL(request.url).origin) {
    headers["access-control-allow-origin"] = origin;
  }
  return headers;
}

function json(value: unknown, status = 200, request?: Request): Response {
  return new Response(JSON.stringify(value, null, 2), {
    status,
    headers: request ? { ...JSON_HEADERS, ...corsHeaders(request) } : JSON_HEADERS
  });
}

async function handleSearch(url: URL, env: Env) {
  const query = normalizedSearchQuery(url.searchParams.get("q"));
  const limit = clampNumber(Number(url.searchParams.get("limit") ?? 12), 1, 20);
  if (query.length < 2) {
    return { query, results: [] };
  }
  const hfUrl = new URL("/api/models", hfBase(env));
  hfUrl.searchParams.set("search", query);
  hfUrl.searchParams.set("limit", String(limit));
  hfUrl.searchParams.set("full", "true");
  hfUrl.searchParams.set("sort", "downloads");
  hfUrl.searchParams.set("direction", "-1");
  const models = await hfFetch<HuggingFaceModel[]>(hfUrl);
  return {
    query,
    results: models.map(sanitizeModelSummary)
  };
}

async function handleModel(url: URL, env: Env) {
  const id = requiredModelId(url.searchParams.get("id"));
  const model = await fetchModelDetails(id, env);
  return { model: sanitizeModelDetails(model) };
}

async function handleAdvice(request: Request, url: URL, env: Env) {
  const body = request.method === "POST" ? await readJsonBody(request) : {};
  const id = requiredModelId(url.searchParams.get("id") ?? stringValue(body.id));
  const useAi = url.searchParams.get("ai") === "1" || body.useAi === true;
  if (useAi) {
    enforceRateLimit(request, url, "ai-summary");
  }
  const model = await fetchModelDetails(id, env);
  const advisor = buildAdvisor(model);
  const aiSummary = useAi ? await generateOpenAiSummary(env, model, advisor) : openAiStatus(env);
  return {
    model: sanitizeModelDetails(model),
    advisor,
    aiSummary
  };
}

async function readJsonBody(request: Request): Promise<Record<string, unknown>> {
  const text = await request.text();
  if (text.length > MAX_JSON_BODY_LENGTH) {
    throw new HttpError(413, `JSON body must be ${MAX_JSON_BODY_LENGTH} characters or fewer.`);
  }
  if (!text.trim()) {
    return {};
  }
  const value = JSON.parse(text);
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function hfBase(env: Env): string {
  return env.HF_API_BASE || "https://huggingface.co";
}

async function fetchModelDetails(id: string, env: Env): Promise<HuggingFaceModel> {
  const hfUrl = new URL(`/api/models/${encodeRepoId(id)}`, hfBase(env));
  hfUrl.searchParams.set("full", "true");
  return hfFetch<HuggingFaceModel>(hfUrl);
}

async function hfFetch<T>(url: URL): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "accept": "application/json",
      "user-agent": "mlx-model-advisor-worker/0.1"
    }
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Hugging Face API ${response.status}: ${text.slice(0, 200)}`);
  }
  return response.json() as Promise<T>;
}

function requiredModelId(value: string | null | undefined): string {
  const id = (value ?? "").trim();
  if (!id || !id.includes("/")) {
    throw new HttpError(400, "A Hugging Face model id like owner/name is required.");
  }
  if (id.length > MAX_MODEL_ID_LENGTH) {
    throw new HttpError(400, `Model id must be ${MAX_MODEL_ID_LENGTH} characters or fewer.`);
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(id)) {
    throw new HttpError(400, "Model id must look like owner/name and use Hugging Face repo characters.");
  }
  return id;
}

function normalizedSearchQuery(value: string | null | undefined): string {
  const query = (value ?? "").trim().replace(/\s+/g, " ");
  if (query.length > MAX_SEARCH_QUERY_LENGTH) {
    throw new HttpError(400, `Search query must be ${MAX_SEARCH_QUERY_LENGTH} characters or fewer.`);
  }
  return query;
}

function enforceRateLimit(request: Request, url: URL, bucketOverride?: string) {
  const bucket = bucketOverride ?? rateLimitBucket(url.pathname);
  if (!bucket) {
    return;
  }
  const limit = RATE_LIMITS[bucket] ?? 30;
  const now = Date.now();
  const key = `${clientKey(request)}:${bucket}`;
  const current = rateCounters.get(key);
  if (!current || current.resetAt <= now) {
    rateCounters.set(key, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    pruneRateCounters(now);
    return;
  }
  if (current.count >= limit) {
    throw new HttpError(429, "Rate limit exceeded. Try again shortly.");
  }
  current.count += 1;
}

function rateLimitBucket(pathname: string): string {
  if (pathname === "/api/search") {
    return "search";
  }
  if (pathname === "/api/model") {
    return "model";
  }
  if (pathname === "/api/advice") {
    return "advice";
  }
  return "";
}

function clientKey(request: Request): string {
  return request.headers.get("cf-connecting-ip")
    || request.headers.get("x-forwarded-for")?.split(",")[0]?.trim()
    || "anonymous";
}

function pruneRateCounters(now: number) {
  if (rateCounters.size < 5000) {
    return;
  }
  for (const [key, value] of rateCounters) {
    if (value.resetAt <= now) {
      rateCounters.delete(key);
    }
  }
}

function encodeRepoId(id: string): string {
  return id.split("/").map(encodeURIComponent).join("/");
}

function sanitizeModelSummary(model: HuggingFaceModel) {
  const id = model.id || model.modelId || "";
  return {
    id,
    url: id ? `https://huggingface.co/${id}` : "",
    pipelineTag: model.pipeline_tag ?? "",
    libraryName: model.library_name ?? "",
    tags: (model.tags ?? []).slice(0, 16),
    downloads: model.downloads ?? 0,
    likes: model.likes ?? 0,
    gated: model.gated ?? false,
    private: Boolean(model.private),
    lastModified: model.lastModified ?? ""
  };
}

function sanitizeModelDetails(model: HuggingFaceModel) {
  const summary = sanitizeModelSummary(model);
  return {
    ...summary,
    cardData: pickCardData(model.cardData),
    siblings: (model.siblings ?? [])
      .filter((file) => typeof file.rfilename === "string")
      .slice(0, 80)
      .map((file) => ({ rfilename: file.rfilename, size: file.size ?? null })),
    config: pickConfigSignals(model.config),
    transformersInfo: pickConfigSignals(model.transformersInfo)
  };
}

function pickCardData(cardData: Record<string, unknown> | undefined) {
  if (!cardData) {
    return {};
  }
  const keys = ["pipeline_tag", "library_name", "tags", "base_model", "license", "language", "datasets", "metrics"];
  return Object.fromEntries(keys.filter((key) => key in cardData).map((key) => [key, cardData[key]]));
}

function pickConfigSignals(config: Record<string, unknown> | undefined) {
  if (!config) {
    return {};
  }
  const keys = ["model_type", "architectures", "num_hidden_layers", "num_attention_heads", "hidden_size", "vocab_size"];
  return Object.fromEntries(keys.filter((key) => key in config).map((key) => [key, config[key]]));
}

function buildAdvisor(model: HuggingFaceModel) {
  const classification = classifyModel(model);
  const methods = ADVISOR_DATA.methods
    .filter((method) => appliesToFamily(method.appliesTo, classification.family.id))
    .sort(methodSortKey);
  const sources = new Map<string, Source>();
  addSource(sources, "asset-architectures");

  const bucketMap = new Map<string, Array<ReturnType<typeof serializeMethod>>>();
  for (const bucket of ADVISOR_DATA.taxonomy.advisorBuckets) {
    bucketMap.set(bucket.id, []);
  }
  for (const method of methods) {
    for (const sourceId of method.evidenceSourceIds) {
      addSource(sources, sourceId);
    }
    addSource(sources, "asset-optimization-guidance");
    const item = serializeMethod(method);
    const items = bucketMap.get(item.advisorBucket) ?? [];
    items.push(item);
    bucketMap.set(item.advisorBucket, items);
  }

  const researchNotes = relevantResearchNotes(classification.family, methods);
  for (const note of researchNotes) {
    for (const sourceId of note.evidenceSourceIds) {
      addSource(sources, sourceId);
    }
  }

  const citations = [...sources.values()].sort((a, b) => a.id.localeCompare(b.id));
  const buckets = ADVISOR_DATA.taxonomy.advisorBuckets
    .slice()
    .sort((a, b) => BUCKET_ORDER.indexOf(a.id) - BUCKET_ORDER.indexOf(b.id))
    .map((bucket) => serializeBucket(bucket, bucketMap.get(bucket.id) ?? []));

  return {
    family: classification.family,
    confidence: classification.confidence,
    reasons: classification.reasons,
    buckets,
    researchNotes,
    citations,
    instructions: buildInstructions(model, classification.family),
    defaults: {
      keepGates: ADVISOR_DATA.taxonomy.defaultKeepGate
    }
  };
}

function classifyModel(model: HuggingFaceModel): { family: Family; confidence: "high" | "medium" | "low"; reasons: string[] } {
  const corpus = modelCorpus(model);
  const pipeline = (model.pipeline_tag || stringValue(model.cardData?.pipeline_tag)).toLowerCase();
  const scores = new Map<string, { score: number; reasons: string[] }>();

  for (const family of ADVISOR_DATA.families) {
    const entry = scores.get(family.id) ?? { score: 0, reasons: [] };
    for (const alias of family.aliases) {
      if (matchesToken(corpus, alias)) {
        entry.score += 4;
        entry.reasons.push(`matched alias '${alias}'`);
      }
    }
    for (const pattern of family.classPatterns) {
      if (matchesToken(corpus, pattern)) {
        entry.score += 3;
        entry.reasons.push(`matched class pattern '${pattern}'`);
      }
    }
    scores.set(family.id, entry);
  }

  for (const hint of TASK_FAMILY_HINTS) {
    if (hint.tasks.includes(pipeline)) {
      bump(scores, hint.family, 3, hint.reason);
    }
  }
  for (const override of FAMILY_OVERRIDES) {
    if (override.tokens.some((token) => matchesToken(corpus, token))) {
      bump(scores, override.family, 7, override.reason);
    }
  }

  const best = [...scores.entries()].sort((a, b) => b[1].score - a[1].score)[0];
  const family = ADVISOR_DATA.families.find((item) => item.id === best?.[0]) ?? ADVISOR_DATA.families[0];
  const score = best?.[1].score ?? 0;
  return {
    family,
    confidence: score >= 7 ? "high" : score >= 3 ? "medium" : "low",
    reasons: (best?.[1].reasons.length ? best[1].reasons : ["fallback to closest supported family"]).slice(0, 6)
  };
}

function modelCorpus(model: HuggingFaceModel): string {
  const values = [
    model.id,
    model.modelId,
    model.pipeline_tag,
    model.library_name,
    ...(model.tags ?? []),
    ...(Array.isArray(model.cardData?.tags) ? model.cardData?.tags ?? [] : []),
    model.cardData?.pipeline_tag,
    model.cardData?.library_name,
    model.cardData?.base_model,
    model.config?.model_type,
    model.config?.architectures,
    model.transformersInfo?.auto_model
  ];
  return values.flat(3).filter(Boolean).join(" ").toLowerCase().replace(/_/g, "-");
}

function bump(scores: Map<string, { score: number; reasons: string[] }>, family: string, score: number, reason: string) {
  const entry = scores.get(family) ?? { score: 0, reasons: [] };
  entry.score += score;
  entry.reasons.push(reason);
  scores.set(family, entry);
}

function matchesToken(corpus: string, token: string): boolean {
  const normalized = token.toLowerCase().replace(/_/g, "-");
  if (normalized.length <= 3) {
    return new RegExp(`(^|[^a-z0-9])${escapeRegExp(normalized)}([^a-z0-9]|$)`).test(corpus);
  }
  return corpus.includes(normalized);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function appliesToFamily(appliesTo: readonly string[], family: string): boolean {
  const f = family.toLowerCase();
  const tokens = new Set(f.replace(/-/g, " ").split(" "));
  for (const raw of appliesTo) {
    const value = raw.toLowerCase();
    if (APPLIES_TO_FAMILY_ALIASES[value]?.includes(f)) {
      return true;
    }
    if (value === "all" || value === f || value.includes(f) || f.includes(value)) {
      return true;
    }
    if (!value.includes("-") && tokens.has(value)) {
      return true;
    }
  }
  return false;
}

function methodSortKey(a: Method, b: Method): number {
  const bucketDelta = BUCKET_ORDER.indexOf(a.advisorBucket) - BUCKET_ORDER.indexOf(b.advisorBucket);
  if (bucketDelta !== 0) {
    return bucketDelta;
  }
  return `${a.category}:${a.id}`.localeCompare(`${b.category}:${b.id}`);
}

function serializeMethod(method: Method) {
  return {
    id: method.id,
    category: method.category,
    status: method.status,
    advisorBucket: method.advisorBucket,
    objectives: method.objectives,
    recommendation: method.recommendation,
    expectedEffect: method.expectedEffect,
    impact: summarizeImpact(method),
    certainty: summarizeCertainty(method),
    tradeoffs: method.tradeoffs,
    validationGates: method.validationGates,
    rollbackConditions: method.rollbackConditions,
    evidenceSourceIds: method.evidenceSourceIds
  };
}

function summarizeImpact(method: Method) {
  const text = method.expectedEffect;
  const numericMatches = text.match(/(?:up to\s+|about\s+)?\d+(?:\.\d+)?x|\d+(?:\.\d+)?%\s*(?:-|to)\s*\d+(?:\.\d+)?%|\d+(?:\.\d+)?%/gi) ?? [];
  if (numericMatches.length > 0) {
    return {
      value: numericMatches.slice(0, 3).join(", "),
      label: "Source-reported",
      caveat: text
    };
  }
  if (/profile-required/i.test(text)) {
    return {
      value: "Profile-required",
      label: "Measure locally",
      caveat: text.replace(/^Profile-required\.\s*/i, "")
    };
  }
  if (/no speedup claim|not a speedup|no generic/i.test(text)) {
    return {
      value: "No portable speedup",
      label: "Boundary",
      caveat: text
    };
  }
  if (/memory.*bit width|KV memory/i.test(text)) {
    return {
      value: "Memory-scaled",
      label: "Estimate",
      caveat: text
    };
  }
  return {
    value: "Workload-dependent",
    label: "Estimate",
    caveat: text
  };
}

function summarizeCertainty(method: Method) {
  if (method.status === "native-mlx" || method.status === "official-mlx-project") {
    return "supported path; still validate locally";
  }
  if (method.status === "proven-mlx-port") {
    return "proven port pattern; benchmark required";
  }
  if (method.status === "research-candidate") {
    return "experimental; explicit opt-in required";
  }
  return "do not use for MLX";
}

function serializeBucket(bucket: Bucket, items: Array<ReturnType<typeof serializeMethod>>) {
  return {
    id: bucket.id,
    label: bucket.label,
    description: bucket.description,
    requiresUserOptIn: Boolean(bucket.requires_user_opt_in),
    prompt: "prompt" in bucket ? stringValue(bucket.prompt) : "",
    items
  };
}

function relevantResearchNotes(family: Family, methods: readonly Method[]) {
  const terms = new Set([
    ...family.id.split("-"),
    ...family.aliases.slice(0, 12).map((value) => value.replace(/_/g, "-")),
    ...methods.flatMap((method) => [method.id, method.category, ...method.appliesTo])
  ].map((value) => value.toLowerCase()));

  const notes = [
    ...ADVISOR_DATA.learnings.map((item) => ({
      id: item.id,
      source: "contributor-learning",
      status: item.status,
      advisorBucket: item.advisorBucket,
      summary: item.summary,
      validationGate: item.validationGate,
      rollbackCondition: item.rollbackCondition,
      reasonHeld: item.reasonHeld,
      evidence: item.evidence,
      evidenceSourceIds: item.evidenceSourceIds
    })),
    ...ADVISOR_DATA.backlogItems.map((item) => ({
      id: item.id,
      source: "research-backlog",
      status: item.status,
      advisorBucket: item.advisorBucket,
      summary: item.summary,
      validationGate: item.requiredGate,
      rollbackCondition: "",
      reasonHeld: item.priority,
      evidence: item.affected,
      evidenceSourceIds: item.evidenceSourceIds
    }))
  ];

  return notes
    .map((note) => ({ note, score: noteScore(note, terms) }))
    .filter(({ score, note }) => score > 0 || note.status === "validated")
    .sort((a, b) => b.score - a.score || a.note.id.localeCompare(b.note.id))
    .slice(0, 8)
    .map(({ note }) => note);
}

function noteScore(note: { id: string; summary: string; evidence: readonly string[] }, terms: Set<string>): number {
  const text = `${note.id} ${note.summary} ${note.evidence.join(" ")}`.toLowerCase().replace(/_/g, "-");
  let score = 0;
  for (const term of terms) {
    if (term.length > 3 && text.includes(term)) {
      score += 1;
    }
  }
  return score;
}

function buildInstructions(model: HuggingFaceModel, family: Family) {
  const id = model.id || model.modelId || "owner/model";
  return {
    installSkill: "codex skill install https://github.com/Amal-David/mlx-porting-skill",
    inspect: `python3 mlx-model-porting/scripts/inspect_model.py ${id} --output inspection.json`,
    plan: `python3 mlx-model-porting/scripts/make_port_plan.py inspection.json --family ${family.id} --output port-plan.json`,
    optimize: `python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json --family ${family.id} --markdown optimization-shortlist.md`,
    validate: "Run parity first, then benchmark with workload metadata. Keep rollback conditions attached to every optimization.",
    experimentalPrompt: "This is an experimental approach. Do you want to try it?"
  };
}

function addSource(sources: Map<string, Source>, sourceId: string) {
  const source = ADVISOR_DATA.sources.find((item) => item.id === sourceId);
  if (source) {
    sources.set(source.id, source);
  }
}

function openAiStatus(env: Env) {
  return {
    status: env.OPENAI_API_KEY ? "available" : "not_configured",
    model: env.OPENAI_API_KEY ? env.OPENAI_MODEL || "gpt-5.4-mini" : "",
    text: ""
  };
}

async function generateOpenAiSummary(env: Env, model: HuggingFaceModel, advisor: ReturnType<typeof buildAdvisor>) {
  if (!env.OPENAI_API_KEY) {
    return openAiStatus(env);
  }
  const openAiModel = env.OPENAI_MODEL || "gpt-5.4-mini";
  const payload = {
    model: openAiModel,
    instructions: [
      "You are summarizing an MLX model-porting advisor report.",
      "Do not invent speedup numbers. Mention profile-required or benchmark-required when applicable.",
      "Keep experimental approaches explicitly labeled experimental.",
      "Return 4 short bullets for an engineer."
    ].join(" "),
    input: JSON.stringify({
      model: sanitizeModelSummary(model),
      family: advisor.family.id,
      confidence: advisor.confidence,
      validated: advisor.buckets.filter((bucket) => bucket.id !== "experimental-approach" && bucket.id !== "rejected-do-not-use").map((bucket) => ({ bucket: bucket.label, methods: bucket.items.slice(0, 4).map((item) => item.id) })),
      experimental: advisor.buckets.find((bucket) => bucket.id === "experimental-approach")?.items.slice(0, 5).map((item) => item.id) ?? [],
      runbook: advisor.family.runbook
    }),
    max_output_tokens: 500
  };

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "authorization": `Bearer ${env.OPENAI_API_KEY}`,
      "content-type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  const data = await response.json() as Record<string, unknown>;
  if (!response.ok) {
    return {
      status: "error",
      model: openAiModel,
      text: "",
      error: typeof data.error === "object" ? JSON.stringify(data.error) : `OpenAI API ${response.status}`
    };
  }
  return {
    status: "ok",
    model: openAiModel,
    text: extractResponseText(data)
  };
}

function extractResponseText(data: Record<string, unknown>): string {
  if (typeof data.output_text === "string") {
    return data.output_text;
  }
  const output = Array.isArray(data.output) ? data.output : [];
  return output.flatMap((item) => {
    const content = item && typeof item === "object" && "content" in item ? (item as { content?: unknown }).content : [];
    return Array.isArray(content) ? content : [];
  }).map((content) => {
    if (content && typeof content === "object" && "text" in content) {
      return String((content as { text?: unknown }).text ?? "");
    }
    return "";
  }).filter(Boolean).join("\n").trim();
}

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, Math.floor(value)));
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function renderAppHtml() {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLX Model Advisor</title>
  <style>
    :root {
      color-scheme: light;
      --page: #f6f3ec;
      --surface: #fffcf6;
      --surface-strong: #f0ebe1;
      --ink: #202124;
      --muted: #686059;
      --line: #d7d0c5;
      --green: #116247;
      --green-soft: #dcebe2;
      --blue: #315f86;
      --blue-soft: #e1ebf2;
      --amber: #9b6413;
      --amber-soft: #f3e5c8;
      --red: #8c342d;
      --red-soft: #f0dbd8;
      --focus: #1d69a8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--page);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input {
      font: inherit;
    }
    button:focus-visible, input:focus-visible {
      outline: 3px solid color-mix(in srgb, var(--focus) 45%, transparent);
      outline-offset: 2px;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--surface);
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    main {
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 24px; line-height: 1.1; }
    h2 { font-size: 18px; }
    h3 { font-size: 14px; }
    .muted { color: var(--muted); }
    .label {
      display: block;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .search input {
      width: 100%;
      height: 44px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 0 12px;
      border-radius: 8px;
    }
    .results {
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-height: 0;
      overflow: auto;
    }
    .result {
      text-align: left;
      border: 1px solid var(--line);
      background: #fffdf8;
      border-radius: 8px;
      padding: 10px;
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease;
    }
    .result:hover, .result.active {
      border-color: var(--green);
      background: var(--green-soft);
    }
    .result strong {
      display: block;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: rgba(255,255,255,.6);
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, .75fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 8px;
      padding: 16px;
    }
    .section {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .headline {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .status.high, .bucket-validated-locally, .bucket-validated-source-theory { background: var(--green-soft); color: var(--green); }
    .status.medium, .bucket-benchmark-required { background: var(--amber-soft); color: var(--amber); }
    .status.low, .bucket-experimental-approach { background: var(--blue-soft); color: var(--blue); }
    .bucket-rejected-do-not-use { background: var(--red-soft); color: var(--red); }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }
    .metric span {
      display: block;
      font-size: 12px;
      color: var(--muted);
    }
    .metric strong {
      display: block;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }
    .decision-strip {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .decision {
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }
    .decision span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .decision strong {
      display: block;
      font-size: 18px;
      margin-top: 2px;
    }
    .method-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffdf8;
      padding: 12px;
      display: grid;
      gap: 10px;
    }
    .method-title {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: start;
    }
    .method-title strong {
      overflow-wrap: anywhere;
      font-size: 17px;
    }
    .impact-pill {
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--blue-soft);
      color: var(--blue);
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .impact-pill.source { background: var(--green-soft); color: var(--green); }
    .impact-pill.boundary { background: var(--red-soft); color: var(--red); }
    .method-kpis {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .kpi {
      border-top: 1px solid var(--line);
      padding-top: 7px;
      min-width: 0;
    }
    .kpi span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 700;
    }
    .kpi strong {
      display: block;
      margin-top: 2px;
      overflow-wrap: anywhere;
      font-size: 13px;
    }
    .method-more {
      border: 0;
      background: transparent;
      border-radius: 0;
    }
    .method-more summary {
      padding: 0;
      justify-content: flex-start;
      color: var(--blue);
      font-size: 13px;
      font-weight: 700;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffdf8;
      overflow: hidden;
    }
    summary {
      cursor: pointer;
      padding: 12px;
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
    }
    summary::-webkit-details-marker { display: none; }
    .method-body {
      padding: 0 12px 12px;
      display: grid;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    ul {
      margin: 0;
      padding-left: 18px;
    }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #17191d;
      color: #f7f2e8;
      border-radius: 8px;
      padding: 12px;
    }
    .copy-row {
      display: grid;
      gap: 8px;
    }
    .copy-row button, .primary {
      border: 1px solid var(--ink);
      background: var(--ink);
      color: #fff;
      border-radius: 8px;
      min-height: 38px;
      padding: 0 12px;
      cursor: pointer;
    }
    .secondary {
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
      border-radius: 8px;
      min-height: 38px;
      padding: 0 12px;
      cursor: pointer;
    }
    .empty {
      min-height: 260px;
      display: grid;
      place-items: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      text-align: center;
      padding: 24px;
    }
    .citations {
      display: grid;
      gap: 8px;
      font-size: 12px;
    }
    .citation {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      overflow-wrap: anywhere;
    }
    .citation a { color: var(--blue); }
    .note-list {
      display: grid;
      gap: 8px;
    }
    .note {
      border-left: 3px solid var(--blue);
      padding: 8px 0 8px 10px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 920px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .workspace { grid-template-columns: 1fr; }
      .decision-strip, .method-kpis { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      aside, main { padding: 16px; }
      .grid { grid-template-columns: 1fr; }
      .headline { flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <header class="section">
        <h1>MLX Model Advisor</h1>
        <p class="muted">Hugging Face model search with skill-backed MLX routes.</p>
      </header>
      <div class="search">
        <label class="label" for="model-search">Model</label>
        <input id="model-search" autocomplete="off" placeholder="qwen, whisper, flux, demucs">
      </div>
      <div id="search-status" class="muted">Type at least 2 characters.</div>
      <div id="results" class="results"></div>
    </aside>
    <main>
      <section id="workspace" class="empty">Select a model.</section>
    </main>
  </div>
  <script>
    const state = { query: "", selected: "", advice: null, timer: null, searchSeq: 0 };
    const searchInput = document.getElementById("model-search");
    const resultsEl = document.getElementById("results");
    const statusEl = document.getElementById("search-status");
    const workspaceEl = document.getElementById("workspace");

    searchInput.addEventListener("input", () => {
      state.query = searchInput.value.trim();
      window.clearTimeout(state.timer);
      state.timer = window.setTimeout(() => search(state.query), 220);
    });

    searchInput.value = "mlx qwen";
    search("mlx qwen");

    async function search(query) {
      const seq = ++state.searchSeq;
      if (query.length < 2) {
        statusEl.textContent = "Type at least 2 characters.";
        resultsEl.innerHTML = "";
        return;
      }
      statusEl.textContent = "Searching...";
      const response = await fetch("/api/search?q=" + encodeURIComponent(query) + "&limit=12");
      const data = await response.json();
      if (seq !== state.searchSeq || query !== searchInput.value.trim()) {
        return;
      }
      statusEl.textContent = data.results.length + " models";
      resultsEl.innerHTML = data.results.map(renderResult).join("");
      resultsEl.querySelectorAll("[data-model-id]").forEach((button) => {
        button.addEventListener("click", () => selectModel(button.getAttribute("data-model-id")));
      });
    }

    async function selectModel(id, ai = false) {
      state.selected = id;
      workspaceEl.className = "empty";
      workspaceEl.textContent = "Loading " + id + "...";
      resultsEl.querySelectorAll(".result").forEach((button) => {
        button.classList.toggle("active", button.getAttribute("data-model-id") === id);
      });
      const response = await fetch("/api/advice?id=" + encodeURIComponent(id) + (ai ? "&ai=1" : ""));
      const data = await response.json();
      state.advice = data;
      workspaceEl.className = "workspace";
      workspaceEl.innerHTML = renderAdvice(data);
      wireWorkspace();
      if (window.matchMedia("(max-width: 920px)").matches) {
        workspaceEl.scrollIntoView({ block: "start", behavior: "smooth" });
      }
    }

    function wireWorkspace() {
      workspaceEl.querySelectorAll("[data-copy]").forEach((button) => {
        button.addEventListener("click", async () => {
          await navigator.clipboard.writeText(button.getAttribute("data-copy"));
          const original = button.textContent;
          button.textContent = "Copied";
          window.setTimeout(() => { button.textContent = original; }, 1200);
        });
      });
      const aiButton = workspaceEl.querySelector("[data-ai-summary]");
      if (aiButton) {
        aiButton.addEventListener("click", () => selectModel(state.selected, true));
      }
    }

    function renderResult(model) {
      const tags = (model.tags || []).slice(0, 4).map((tag) => '<span class="chip">' + escapeHtml(tag) + '</span>').join("");
      return '<button class="result" data-model-id="' + escapeAttr(model.id) + '">' +
        '<strong>' + escapeHtml(model.id) + '</strong>' +
        '<span class="muted">' + escapeHtml(model.pipelineTag || model.libraryName || "model") + ' · ' + formatNumber(model.downloads) + ' downloads</span>' +
        '<div class="chips">' + tags + '</div>' +
      '</button>';
    }

    function renderAdvice(data) {
      if (data.error) {
        return '<section class="empty">' + escapeHtml(data.error) + '</section>';
      }
      const model = data.model;
      const advisor = data.advisor;
      const ai = data.aiSummary || {};
      return '<section class="section">' +
        renderModelPanel(model, advisor, ai) +
        renderAdvisorSummary(advisor) +
        advisor.buckets.map(renderBucket).join("") +
      '</section>' +
      '<aside class="section">' +
        renderInstructionPanel(advisor.instructions) +
        renderNotes(advisor.researchNotes) +
        renderCitations(advisor.citations) +
      '</aside>';
    }

    function renderModelPanel(model, advisor, ai) {
      const tags = (model.tags || []).slice(0, 8).map((tag) => '<span class="chip">' + escapeHtml(tag) + '</span>').join("");
      const aiBlock = ai.status === "ok"
        ? '<div class="panel section"><h3>AI summary</h3><p class="muted">' + escapeHtml(ai.text) + '</p></div>'
        : '<button class="secondary" data-ai-summary>Generate AI summary</button>' + (ai.status === "error" ? '<p class="muted">' + escapeHtml(ai.error || "AI summary failed") + '</p>' : '');
      return '<div class="panel section">' +
        '<div class="headline"><div><h2>' + escapeHtml(model.id) + '</h2><p class="muted">' + escapeHtml(model.pipelineTag || model.libraryName || "metadata") + '</p></div><span class="status ' + advisor.confidence + '">' + escapeHtml(advisor.confidence) + ' confidence</span></div>' +
        '<div class="grid">' +
          metric("Family", advisor.family.id) +
          metric("Downloads", formatNumber(model.downloads)) +
          metric("Primary route", advisor.family.targets[0] || "standalone-mlx") +
          metric("Runbook", advisor.family.runbook) +
        '</div>' +
        '<div class="chips">' + tags + '</div>' +
        '<p class="muted">Signals: ' + advisor.reasons.map(escapeHtml).join("; ") + '</p>' +
        aiBlock +
      '</div>';
    }

    function renderAdvisorSummary(advisor) {
      const all = advisor.buckets.flatMap((bucket) => (bucket.items || []).map((item) => ({ ...item, bucketLabel: bucket.label })));
      const usable = all.filter((item) => item.advisorBucket !== "rejected-do-not-use");
      const numbered = usable.filter((item) => /^\\d|^up to|^about/i.test(item.impact.value));
      const profile = usable.filter((item) => item.impact.value === "Profile-required");
      const top = (numbered.length ? numbered : usable).slice(0, 4);
      const experimental = advisor.buckets.find((bucket) => bucket.id === "experimental-approach");
      const rejected = advisor.buckets.find((bucket) => bucket.id === "rejected-do-not-use");
      return '<div class="panel section">' +
        '<div class="headline"><div><h2>What this changes</h2><p class="muted">Impact numbers are source-reported only. Profile-required means no portable percentage has been validated for this exact model yet.</p></div></div>' +
        '<div class="decision-strip">' +
          decision("Source numbers", numbered.length ? numbered.length + " methods" : "none yet") +
          decision("Need benchmark", profile.length + " methods") +
          decision("Experimental", (experimental?.items || []).length + " opt-in") +
        '</div>' +
        '<div class="section">' + top.map(renderMethod).join("") + '</div>' +
        (rejected && rejected.items.length ? '<p class="muted">' + rejected.items.length + ' rejected paths are kept visible so users do not chase non-transferable optimizations.</p>' : '') +
      '</div>';
    }

    function renderBucket(bucket) {
      const items = bucket.items || [];
      const prompt = bucket.requiresUserOptIn ? '<p class="muted"><strong>' + escapeHtml(bucket.prompt) + '</strong></p>' : "";
      return '<div class="panel section">' +
        '<div class="headline"><div><h2>' + escapeHtml(bucket.label) + '</h2><p class="muted">' + escapeHtml(bucket.description) + '</p></div><span class="status bucket-' + escapeAttr(bucket.id) + '">' + items.length + '</span></div>' +
        prompt +
        (items.length ? items.map(renderMethod).join("") : '<p class="muted">No matching methods for this model family at this evidence level.</p>') +
      '</div>';
    }

    function renderMethod(method) {
      const gate = first(method.validationGates);
      const rollback = first(method.rollbackConditions);
      const impactClass = /^\\d|^up to|^about/i.test(method.impact.value) ? "source" : method.impact.value.includes("No portable") ? "boundary" : "";
      return '<article class="method-card">' +
        '<div class="method-title"><strong>' + escapeHtml(method.id) + '</strong><span class="impact-pill ' + impactClass + '">' + escapeHtml(method.impact.value) + '</span></div>' +
        '<p class="muted">' + escapeHtml(method.recommendation) + '</p>' +
        '<div class="method-kpis">' +
          kpi("Evidence", method.certainty) +
          kpi("First gate", gate || "define parity gate") +
          kpi("Rollback", rollback || "write before trying") +
        '</div>' +
        '<p class="muted"><strong>' + escapeHtml(method.impact.label) + ':</strong> ' + escapeHtml(method.impact.caveat) + '</p>' +
        '<details class="method-more"><summary>Show all gates and caveats</summary><div class="method-body">' +
          list("Validation gates", method.validationGates) +
          list("Rollback", method.rollbackConditions) +
          list("Tradeoffs", method.tradeoffs) +
        '</div></details>' +
      '</article>';
    }

    function renderInstructionPanel(instructions) {
      const text = [
        instructions.installSkill,
        instructions.inspect,
        instructions.plan,
        instructions.optimize,
        instructions.validate,
        instructions.experimentalPrompt
      ].join("\\n");
      return '<div class="panel section">' +
        '<h2>CLI instructions</h2>' +
        '<pre>' + escapeHtml(text) + '</pre>' +
        '<div class="copy-row"><button data-copy="' + escapeAttr(text) + '">Copy instructions</button></div>' +
      '</div>';
    }

    function renderNotes(notes) {
      if (!notes || !notes.length) return "";
      return '<div class="panel section"><h2>Research notes</h2><div class="note-list">' +
        notes.map((note) => '<div class="note"><strong>' + escapeHtml(note.id) + '</strong><p>' + escapeHtml(note.summary) + '</p><p>' + escapeHtml(note.validationGate || note.reasonHeld || "") + '</p></div>').join("") +
      '</div></div>';
    }

    function renderCitations(citations) {
      return '<div class="panel section"><h2>Citations</h2><div class="citations">' +
        citations.map((source) => '<div class="citation"><strong>' + escapeHtml(source.title) + '</strong><br><a href="' + escapeAttr(source.url) + '" target="_blank" rel="noreferrer">' + escapeHtml(source.url) + '</a><br><span class="muted">' + escapeHtml(source.kind) + ' · ' + escapeHtml(source.reviewDepth || "reviewed") + '</span></div>').join("") +
      '</div></div>';
    }

    function metric(label, value) {
      return '<div class="metric"><span>' + escapeHtml(label) + '</span><strong>' + escapeHtml(String(value || "")) + '</strong></div>';
    }
    function decision(label, value) {
      return '<div class="decision"><span>' + escapeHtml(label) + '</span><strong>' + escapeHtml(value) + '</strong></div>';
    }
    function kpi(label, value) {
      return '<div class="kpi"><span>' + escapeHtml(label) + '</span><strong>' + escapeHtml(value) + '</strong></div>';
    }
    function first(values) {
      return values && values.length ? values[0] : "";
    }

    function list(label, values) {
      if (!values || !values.length) return "";
      return '<div><strong>' + escapeHtml(label) + '</strong><ul>' + values.map((value) => '<li>' + escapeHtml(value) + '</li>').join("") + '</ul></div>';
    }

    function formatNumber(value) {
      return new Intl.NumberFormat().format(value || 0);
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
    }
    function escapeAttr(value) {
      return escapeHtml(value).replace(/\\n/g, "&#10;");
    }
  </script>
</body>
</html>`;
}
