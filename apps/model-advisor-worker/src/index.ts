import { ADVISOR_DATA } from "./skill-data.generated";

export interface Env {
  HF_API_BASE?: string;
  OPENAI_API_KEY?: string;
  OPENAI_MODEL?: string;
}

type AdvisorData = typeof ADVISOR_DATA;
type Family = AdvisorData["families"][number];
type Method = AdvisorData["methods"][number];
type OutcomeRecord = AdvisorData["modelOutcomes"]["records"][number];
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
  "autoregressive-transformer": ["dense-decoder-transformer", "moe-decoder-transformer", "encoder-decoder-transformer"],
  "decoder-transformer": ["dense-decoder-transformer", "moe-decoder-transformer"],
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

const DERIVATIVE_OWNER_TOKENS = ["mlx-community", "lmstudio-community", "trl-internal-testing", "hf-internal-testing"];
const DERIVATIVE_REPOSITORY_TOKENS = [
  "mlx",
  "coreml",
  "gguf",
  "ggml",
  "onnx",
  "openvino",
  "tflite",
  "ollama",
  "gptq",
  "awq",
  "exl2",
  "bnb",
  "bitsandbytes",
  "quantized",
  "quant",
  "4bit",
  "8bit",
  "int4",
  "int8",
  "q4",
  "q5",
  "q8",
  "lora",
  "adapter",
  "peft",
  "distill"
];

const BASE_SEARCH_REWRITES: Array<[RegExp, string]> = [
  [/\bwhisperkit\b/gi, "whisper"],
  [/\bmlx\b/gi, ""],
  [/\bcoreml\b/gi, ""],
  [/\bgguf\b/gi, ""],
  [/\bggml\b/gi, ""],
  [/\bonnx\b/gi, ""],
  [/\b4bit\b|\b8bit\b|\bquantized\b/gi, ""]
];

const DISCOVERY_CATEGORY_QUERIES: Record<string, { label: string; query: string; pipelineTag?: string; sort?: string; minDownloads?: number }> = {
  popular: { label: "Popular", query: "", sort: "downloads" },
  recent: { label: "Recent", query: "qwen", sort: "lastModified", minDownloads: 1000 },
  text: { label: "Text", query: "instruct", pipelineTag: "text-generation", sort: "downloads" },
  vision: { label: "Vision", query: "vision language", pipelineTag: "image-text-to-text", sort: "downloads" },
  audio: { label: "Audio", query: "whisper", pipelineTag: "automatic-speech-recognition", sort: "downloads" }
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
      if (url.pathname === "/api/discover") {
        return json(await handleDiscover(url, env), 200, request);
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
  const requestedQuery = normalizedSearchQuery(url.searchParams.get("q"));
  const limit = clampNumber(Number(url.searchParams.get("limit") ?? 12), 1, 20);
  const includeDerivatives = url.searchParams.get("include_derivatives") === "1" || url.searchParams.get("includePorts") === "1";
  const category = DISCOVERY_CATEGORY_QUERIES[url.searchParams.get("category") ?? ""] ?? null;
  const exactModelQuery = looksLikeModelId(requestedQuery);
  const upstreamQuery = exactModelQuery ? requestedQuery : baseModelQuery(requestedQuery || category?.query || "");
  if (requestedQuery.length > 0 && requestedQuery.length < 2) {
    return { query: requestedQuery, upstreamQuery, mode: "search", includeDerivatives, hiddenDerivatives: 0, results: [] };
  }
  const hfUrl = new URL("/api/models", hfBase(env));
  if (upstreamQuery) {
    hfUrl.searchParams.set("search", upstreamQuery);
  }
  if (category?.pipelineTag && !requestedQuery) {
    hfUrl.searchParams.set("pipeline_tag", category.pipelineTag);
  }
  hfUrl.searchParams.set("limit", String(includeDerivatives ? limit : Math.min(100, limit * 10)));
  hfUrl.searchParams.set("full", "true");
  hfUrl.searchParams.set("sort", category?.sort || "downloads");
  hfUrl.searchParams.set("direction", "-1");
  const models = await hfFetch<HuggingFaceModel[]>(hfUrl);
  if (exactModelQuery && !models.some((model) => modelId(model).toLowerCase() === requestedQuery.toLowerCase())) {
    try {
      models.unshift(await fetchModelDetails(requestedQuery, env));
    } catch {
      // Search results still render when an exact-model lookup fails.
    }
  }
  const summaries = models.map(sanitizeModelSummary).filter((model) => model.id);
  const results = summaries.filter((model) => (
    includeDerivatives
    || model.repositoryKind !== "derivative"
    || (exactModelQuery && model.id.toLowerCase() === requestedQuery.toLowerCase())
  )).filter((model) => !category?.minDownloads || model.downloads >= category.minDownloads);
  const rankedResults = rankSearchResults(results, requestedQuery, upstreamQuery);
  return {
    query: requestedQuery,
    upstreamQuery,
    mode: requestedQuery ? "search" : category?.label.toLowerCase() || "popular",
    includeDerivatives,
    hiddenDerivatives: summaries.length - results.length,
    results: rankedResults.slice(0, limit)
  };
}

async function handleDiscover(url: URL, env: Env) {
  const limit = clampNumber(Number(url.searchParams.get("limit") ?? 8), 1, 12);
  const includeDerivatives = url.searchParams.get("include_derivatives") === "1" || url.searchParams.get("includePorts") === "1";
  const entries = Object.entries(DISCOVERY_CATEGORY_QUERIES);
  const data = await Promise.all(entries.map(async ([id, category]) => {
    const searchUrl = new URL(url);
    searchUrl.searchParams.set("q", "");
    searchUrl.searchParams.set("category", id);
    searchUrl.searchParams.set("limit", String(limit));
    if (includeDerivatives) {
      searchUrl.searchParams.set("include_derivatives", "1");
    } else {
      searchUrl.searchParams.delete("include_derivatives");
    }
    const result = await handleSearch(searchUrl, env);
    return [id, { label: category.label, ...result }] as const;
  }));
  return Object.fromEntries(data);
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
  if (!looksLikeModelId(id)) {
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

function baseModelQuery(query: string): string {
  let rewritten = query;
  for (const [pattern, replacement] of BASE_SEARCH_REWRITES) {
    rewritten = rewritten.replace(pattern, replacement);
  }
  return rewritten.trim().replace(/\s+/g, " ") || query;
}

function rankSearchResults<T extends ReturnType<typeof sanitizeModelSummary>>(models: T[], requestedQuery: string, upstreamQuery: string): T[] {
  const normalizedQuery = normalizeSearchText(upstreamQuery || requestedQuery);
  if (!normalizedQuery) {
    return models;
  }
  const tokens = searchTokens(normalizedQuery);
  return models.slice().sort((a, b) => {
    const scoreDelta = searchScore(b, requestedQuery, normalizedQuery, tokens) - searchScore(a, requestedQuery, normalizedQuery, tokens);
    if (scoreDelta !== 0) {
      return scoreDelta;
    }
    return (b.downloads ?? 0) - (a.downloads ?? 0);
  });
}

function searchScore(model: ReturnType<typeof sanitizeModelSummary>, requestedQuery: string, normalizedQuery: string, tokens: string[]): number {
  const id = normalizeSearchText(model.id);
  const name = normalizeSearchText(model.id.split("/").pop() ?? model.id);
  const owner = normalizeSearchText(model.id.split("/")[0] ?? "");
  const fields = normalizeSearchText([
    model.id,
    model.pipelineTag,
    model.libraryName,
    model.family,
    model.route,
    ...(model.tags ?? [])
  ].join(" "));
  let score = 0;
  if (requestedQuery && id === normalizeSearchText(requestedQuery)) score += 1000;
  if (id === normalizedQuery || name === normalizedQuery) score += 800;
  if (id.startsWith(normalizedQuery) || name.startsWith(normalizedQuery)) score += 500;
  if (owner === normalizedQuery) score += 200;
  for (const token of tokens) {
    if (name === token) score += 220;
    if (name.startsWith(token)) score += 160;
    if (id.includes(token)) score += 120;
    if (fields.includes(token)) score += 60;
  }
  if (model.repositoryKind === "base-candidate") score += 20;
  return score;
}

function searchTokens(query: string): string[] {
  return [...new Set(normalizeSearchText(query).split(" ").filter((token) => token.length > 1))];
}

function normalizeSearchText(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9/._-]+/g, " ").replace(/[-_./]+/g, " ").trim().replace(/\s+/g, " ");
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
  if (pathname === "/api/discover") {
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
  const id = modelId(model);
  const repository = classifyRepositoryKind(model);
  const classification = classifyModel(model);
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
    lastModified: model.lastModified ?? "",
    family: classification.family.id,
    route: classification.family.targets[0] || "standalone-mlx",
    repositoryKind: repository.kind,
    repositoryReasons: repository.reasons
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

function modelId(model: HuggingFaceModel): string {
  return model.id || model.modelId || "";
}

function looksLikeModelId(value: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value);
}

function classifyRepositoryKind(model: HuggingFaceModel): { kind: "base-candidate" | "derivative"; reasons: string[] } {
  const id = modelId(model).toLowerCase();
  const owner = id.split("/")[0] ?? "";
  const name = id.split("/")[1] ?? "";
  const tags = [
    model.library_name,
    model.pipeline_tag,
    ...(model.tags ?? []),
    ...(Array.isArray(model.cardData?.tags) ? model.cardData.tags as string[] : [])
  ].filter(Boolean).join(" ").toLowerCase().replace(/_/g, "-");
  const haystack = `${owner} ${name.replace(/_/g, "-")} ${tags}`;
  const reasons: string[] = [];

  if (DERIVATIVE_OWNER_TOKENS.includes(owner)) {
    reasons.push("known porting or conversion publisher");
  }
  for (const token of DERIVATIVE_REPOSITORY_TOKENS) {
    const pattern = token.length <= 3
      ? new RegExp(`(^|[^a-z0-9])${escapeRegExp(token)}([^a-z0-9]|$)`)
      : new RegExp(`(^|[^a-z0-9])${escapeRegExp(token)}([^a-z0-9]|$)`);
    if (pattern.test(haystack)) {
      reasons.push(`${token} signal`);
    }
  }

  if (reasons.length) {
    return { kind: "derivative", reasons: [...new Set(reasons)].slice(0, 3) };
  }
  return { kind: "base-candidate", reasons: ["base candidate"] };
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
  const modelOutcomes = relevantModelOutcomes(model, classification.family);
  for (const outcome of modelOutcomes) {
    for (const sourceId of outcome.sourceIds) {
      addSource(sources, sourceId);
    }
    addSource(sources, "asset-model-outcomes");
  }
  addSource(sources, "asset-top-models-snapshot");
  addSource(sources, "hf-top-models-api-2026-06-29");
  const speedupSummary = summarizePotentialSpeedup(modelOutcomes);

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
    modelOutcomes,
    speedupSummary,
    topCoverage: topCoverageForModel(model),
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
    return "supported";
  }
  if (method.status === "proven-mlx-port") {
    return "proven pattern";
  }
  if (method.status === "research-candidate") {
    return "opt-in";
  }
  return "do not use";
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

function relevantModelOutcomes(model: HuggingFaceModel, family: Family) {
  return ADVISOR_DATA.modelOutcomes.records
    .map((outcome) => ({ outcome, score: outcomeScore(outcome, model, family) }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || outcomeStatusRank(a.outcome.status) - outcomeStatusRank(b.outcome.status) || a.outcome.id.localeCompare(b.outcome.id))
    .slice(0, 4)
    .map(({ outcome }) => serializeOutcome(outcome));
}

function outcomeScore(outcome: OutcomeRecord, model: HuggingFaceModel, family: Family): number {
  const match = outcome.match as Record<string, unknown>;
  const topModel = topModelFor(model);
  const topMatchedIds = Array.from(topModel?.matchedOutcomeIds ?? []).map(String);
  let score = topMatchedIds.includes(outcome.id) ? 20 : 0;
  if (stringList(match.families).includes(family.id)) {
    score += 12;
  }
  const pipeline = (model.pipeline_tag || stringValue(model.cardData?.pipeline_tag)).toLowerCase();
  if (pipeline && stringList(match.pipeline_tags).includes(pipeline)) {
    score += 8;
  }
  const library = (model.library_name || stringValue(model.cardData?.library_name)).toLowerCase();
  if (library && stringList(match.library_names).includes(library)) {
    score += 5;
  }
  const corpus = modelCorpus(model);
  for (const pattern of stringList(match.id_patterns)) {
    if (matchesToken(corpus, pattern)) {
      score += 6;
    }
  }
  return score;
}

function serializeOutcome(outcome: OutcomeRecord) {
  return {
    id: outcome.id,
    label: outcome.label,
    status: outcome.status,
    statusLabel: outcomeStatusLabel(outcome.status),
    tone: outcomeTone(outcome.status),
    summary: outcome.summary,
    worked: outcome.worked,
    didNotWork: outcome.didNotWork,
    claimBoundary: outcome.claimBoundary,
    potentialSpeedup: outcome.potentialSpeedup,
    sourceIds: outcome.sourceIds,
    nextValidation: outcome.nextValidation
  };
}

function summarizePotentialSpeedup(outcomes: Array<ReturnType<typeof serializeOutcome>>) {
  const preferred = outcomes.find((outcome) => {
    const range = outcome.potentialSpeedup?.overall?.range ?? "";
    return ["local_reproduced", "source_backed_working", "source_reported_benchmark"].includes(String(outcome.status)) && !isFlatSpeedupRange(range);
  }) ?? outcomes.find((outcome) => outcome.potentialSpeedup) ?? null;
  const fallback = {
    range: "1.0x-1.0x",
    confidence: "unknown",
    basis: "No matched outcome has a reviewed speedup range yet.",
    appliesWhen: ["Add a reviewed outcome or run a local benchmark."],
    measure: ["latency", "throughput", "quality"]
  };
  const overall = preferred?.potentialSpeedup?.overall ?? fallback;
  const speculative = preferred?.potentialSpeedup?.speculativeDecoding ?? fallback;
  return {
    outcomeId: preferred?.id ?? "",
    overallRange: rangeOrFallback(overall.range),
    overallConfidence: overall.confidence || "unknown",
    overallBasis: overall.basis || fallback.basis,
    speculativeRange: rangeOrFallback(speculative.range),
    speculativeConfidence: speculative.confidence || "unknown",
    speculativeBasis: speculative.basis || fallback.basis,
    conditions: uniqueStrings([...(overall.appliesWhen ?? []), ...(speculative.appliesWhen ?? [])]).slice(0, 4),
    measures: uniqueStrings([...(overall.measure ?? []), ...(speculative.measure ?? [])]).slice(0, 5)
  };
}

function rangeOrFallback(value: string): string {
  return value && value.trim() ? value : "1.0x-1.0x";
}

function isFlatSpeedupRange(value: string): boolean {
  return /^1(?:\.0)?x\s*-\s*1(?:\.0)?x/i.test(value || "");
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values.filter(Boolean).map(String))];
}

function outcomeStatusRank(status: string): number {
  const order = ["local_reproduced", "source_backed_working", "source_reported_benchmark", "known_limit_or_gap", "unknown"];
  const index = order.indexOf(status);
  return index === -1 ? order.length : index;
}

function outcomeStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    local_reproduced: "Local proof",
    source_backed_working: "Known route",
    source_reported_benchmark: "Reported result",
    known_limit_or_gap: "Known gap",
    unknown: "Unknown"
  };
  return labels[status] || humanizeId(status);
}

function outcomeTone(status: string): string {
  if (status === "local_reproduced" || status === "source_backed_working") return "source";
  if (status === "source_reported_benchmark") return "benchmark";
  if (status === "known_limit_or_gap") return "boundary";
  return "";
}

function topCoverageForModel(model: HuggingFaceModel) {
  const topModel = topModelFor(model);
  return {
    snapshotGeneratedAt: ADVISOR_DATA.topModelsSnapshot.generatedAt,
    modelCount: ADVISOR_DATA.topModelsSnapshot.modelCount,
    coveredCount: ADVISOR_DATA.topModelsSnapshot.coveredCount,
    unknownCount: ADVISOR_DATA.topModelsSnapshot.unknownCount,
    rank: topModel?.rank ?? 0,
    coverageState: topModel?.coverageState ?? "not-in-top-snapshot",
    licenseClass: topModel?.licenseClass ?? "",
    matchedOutcomeIds: topModel?.matchedOutcomeIds ?? []
  };
}

function topModelFor(model: HuggingFaceModel) {
  const id = modelId(model).toLowerCase();
  return ADVISOR_DATA.topModelsSnapshot.models.find((item) => item.id.toLowerCase() === id);
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item).toLowerCase()) : [];
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

function humanizeId(value: string) {
  return String(value || "").replace(/[-_]/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function buildInstructions(model: HuggingFaceModel, family: Family) {
  const id = model.id || model.modelId || "owner/model";
  return {
    installSkill: "codex skill install https://github.com/Amal-David/mlx-porting-skill",
    inspect: `python3 mlx-model-porting/scripts/inspect_model.py ${id} --output inspection.json`,
    plan: `python3 mlx-model-porting/scripts/make_port_plan.py inspection.json --family ${family.id} --output port-plan.json`,
    optimize: `python3 mlx-model-porting/scripts/recommend_optimizations.py inspection.json --family ${family.id} --markdown optimization-shortlist.md`,
    validate: "Parity first. Benchmark with workload metadata. Keep rollback notes.",
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
      "Use only the provided potential speedup ranges. Do not invent speedup numbers.",
      "Mention profile-required or benchmark-required when applicable.",
      "Keep experimental approaches explicitly labeled experimental.",
      "Return 4 short bullets for an engineer. Do not use Markdown bold. Keep each bullet under 14 words."
    ].join(" "),
    input: JSON.stringify({
      model: sanitizeModelSummary(model),
      family: advisor.family.id,
      confidence: advisor.confidence,
      speedup: advisor.speedupSummary,
      outcomes: advisor.modelOutcomes.slice(0, 4).map((outcome) => ({ id: outcome.id, status: outcome.status, label: outcome.label, potentialSpeedup: outcome.potentialSpeedup })),
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
  <meta name="theme-color" content="#f5f0e7">
  <title>MLX Model Advisor</title>
  <style>
    :root {
      color-scheme: light;
      --page: #f5f0e7;
      --surface: #fffaf1;
      --surface-strong: #ece3d4;
      --surface-cool: #edf4f2;
      --ink: #202124;
      --muted: #635c54;
      --soft: #8a8177;
      --line: #d6cdbf;
      --green: #116247;
      --green-soft: #dcebe2;
      --blue: #2e5f87;
      --blue-soft: #e0ebf2;
      --amber: #9b6413;
      --amber-soft: #f4e4c3;
      --red: #8c342d;
      --red-soft: #f0dbd8;
      --violet: #62518c;
      --violet-soft: #e8e2f0;
      --focus: #1d69a8;
    }
    * { box-sizing: border-box; }
    html {
      overflow-x: hidden;
      scroll-behavior: smooth;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(90deg, rgba(32,33,36,.04) 1px, transparent 1px) 0 0 / 32px 32px,
        linear-gradient(180deg, #faf6ee 0%, var(--page) 46%, #edf4f2 100%);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
      -webkit-tap-highlight-color: rgba(17, 98, 71, .16);
    }
    button, input {
      font: inherit;
      touch-action: manipulation;
    }
    button:focus-visible, input:focus-visible, summary:focus-visible, a:focus-visible {
      outline: 3px solid color-mix(in srgb, var(--focus) 45%, transparent);
      outline-offset: 3px;
    }
    h1, h2, h3, p { margin: 0; }
    h1 {
      font-size: clamp(34px, 7vw, 76px);
      line-height: .94;
      max-width: 820px;
      text-wrap: balance;
    }
    h2 {
      font-size: 22px;
      line-height: 1.15;
      text-wrap: balance;
    }
    h3 {
      font-size: 14px;
      line-height: 1.25;
    }
    a { color: var(--blue); }
    .skip-link {
      position: fixed;
      top: 10px;
      left: -999px;
      z-index: 20;
      background: var(--ink);
      color: #fff;
      padding: 8px 10px;
      border-radius: 6px;
    }
    .skip-link:focus {
      left: 10px;
    }
    .app-shell {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: max(24px, env(safe-area-inset-top)) 0 40px;
    }
    .hero {
      min-height: min(760px, calc(100vh - 36px));
      display: grid;
      align-content: center;
      gap: 28px;
      padding: 38px 0 28px;
    }
    .eyebrow {
      color: var(--green);
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .hero-copy {
      display: grid;
      gap: 14px;
    }
    .hero-copy .muted {
      max-width: 650px;
      font-size: 18px;
      line-height: 1.45;
    }
    .composer {
      position: relative;
      border: 1px solid var(--line);
      background: color-mix(in srgb, var(--surface) 94%, white);
      border-radius: 8px;
      box-shadow: 0 24px 70px rgba(32, 33, 36, .09);
      padding: 14px;
    }
    .prompt-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
    }
    .prompt-row input {
      width: 100%;
      min-height: 62px;
      border: 0;
      background: transparent;
      color: var(--ink);
      font-size: clamp(19px, 3vw, 30px);
      font-weight: 700;
      padding: 4px 8px;
    }
    .prompt-row input::placeholder {
      color: color-mix(in srgb, var(--muted) 78%, transparent);
    }
    .primary, .secondary {
      border-radius: 8px;
      min-height: 42px;
      padding: 0 14px;
      cursor: pointer;
      transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease, transform 140ms ease;
    }
    .primary {
      border: 1px solid var(--ink);
      background: var(--ink);
      color: #fff;
      font-weight: 800;
    }
    .secondary {
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
      font-weight: 700;
    }
    .primary:hover, .secondary:hover {
      transform: translateY(-1px);
    }
    .composer-meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border-top: 1px solid var(--line);
      padding: 12px 4px 0;
      color: var(--muted);
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    .toggle {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      min-height: 32px;
      cursor: pointer;
      color: var(--ink);
      font-weight: 700;
      white-space: nowrap;
    }
    .toggle input {
      width: 18px;
      height: 18px;
      accent-color: var(--green);
    }
    .autocomplete {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .autocomplete.compact {
      grid-template-columns: 1fr;
    }
    .selected-model-row {
      min-height: 62px;
      border: 1px solid var(--green);
      background: var(--green-soft);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .selected-model-row strong {
      display: block;
      overflow-wrap: anywhere;
    }
    .suggestion {
      text-align: left;
      min-height: 142px;
      border: 1px solid var(--line);
      background: #fffdf8;
      border-radius: 8px;
      padding: 12px;
      display: grid;
      align-content: space-between;
      gap: 12px;
      cursor: pointer;
      transition: background-color 140ms ease, border-color 140ms ease, transform 140ms ease;
    }
    .suggestion:hover, .suggestion.active {
      border-color: var(--green);
      background: var(--green-soft);
      transform: translateY(-1px);
    }
    .suggestion strong {
      display: block;
      overflow-wrap: anywhere;
      font-size: 15px;
      line-height: 1.18;
    }
    .suggestion-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .tag, .status {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: rgba(255,255,255,.72);
      white-space: nowrap;
    }
    .status {
      border: 0;
      font-weight: 800;
    }
    .status.base, .status.high, .bucket-validated-locally, .bucket-validated-source-theory {
      background: var(--green-soft);
      color: var(--green);
    }
    .status.derivative, .bucket-benchmark-required {
      background: var(--amber-soft);
      color: var(--amber);
    }
    .status.medium, .bucket-experimental-approach {
      background: var(--blue-soft);
      color: var(--blue);
    }
    .status.low, .bucket-rejected-do-not-use {
      background: var(--red-soft);
      color: var(--red);
    }
    .report-shell {
      scroll-margin-top: 24px;
    }
    .empty-state {
      min-height: 220px;
      display: grid;
      place-items: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      text-align: center;
      padding: 24px;
      background: rgba(255,250,241,.58);
    }
    .report-body {
      display: block;
    }
    .report-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 18px;
      align-items: start;
    }
    .report-main, .report-aside, .section {
      display: grid;
      gap: 14px;
    }
    .report-header, .visual, .aside-panel, .method-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 16px;
    }
    .report-header {
      display: grid;
      gap: 16px;
    }
    .headline {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
    }
    .headline h2, .headline h1 {
      overflow-wrap: anywhere;
    }
    .muted {
      color: var(--muted);
      line-height: 1.4;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .decision-summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffdf8;
      padding: 12px;
    }
    .metric, .decision, .kpi {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      min-width: 0;
    }
    .metric span, .decision span, .kpi span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .metric strong, .decision strong, .kpi strong {
      display: block;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }
    .decision-board {
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 14px;
    }
    .visual {
      display: grid;
      gap: 12px;
      min-width: 0;
    }
    .route-flow {
      display: grid;
      grid-template-columns: 1fr 26px 1fr 26px 1fr;
      align-items: center;
      gap: 8px;
    }
    .route-node {
      min-height: 100px;
      border: 1px solid var(--line);
      background: #fffdf8;
      border-radius: 8px;
      padding: 12px;
      display: grid;
      align-content: center;
      gap: 6px;
    }
    .route-node span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .route-node strong {
      overflow-wrap: anywhere;
    }
    .route-edge {
      height: 2px;
      background: var(--line);
      position: relative;
    }
    .route-edge::after {
      content: "";
      position: absolute;
      right: 0;
      top: -4px;
      width: 0;
      height: 0;
      border-top: 5px solid transparent;
      border-bottom: 5px solid transparent;
      border-left: 7px solid var(--line);
    }
    .bar-list {
      display: grid;
      gap: 9px;
    }
    .bar-row {
      display: grid;
      gap: 5px;
    }
    .bar-label {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 12px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .bar-track {
      height: 10px;
      border-radius: 999px;
      background: var(--surface-strong);
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      width: var(--w);
      border-radius: inherit;
      background: var(--green);
    }
    .bar-fill.benchmark { background: var(--amber); }
    .bar-fill.experimental { background: var(--blue); }
    .bar-fill.rejected { background: var(--red); }
    .boundary-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .boundary {
      border-top: 1px solid var(--line);
      padding-top: 9px;
      min-width: 0;
    }
    .boundary strong {
      display: block;
      font-size: 19px;
      font-variant-numeric: tabular-nums;
      overflow-wrap: anywhere;
    }
    .boundary span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }
    .branch-list {
      display: grid;
      gap: 18px;
    }
    .branch {
      display: grid;
      gap: 10px;
    }
    .branch-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    .method-card {
      background: #fffdf8;
      display: grid;
      gap: 10px;
    }
    .derivative-warning {
      border: 1px solid var(--amber);
      border-radius: 8px;
      background: var(--amber-soft);
      padding: 12px;
      display: grid;
      gap: 8px;
    }
    .outcome-panel {
      display: grid;
      gap: 12px;
    }
    .outcome-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    .outcome-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .outcome-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffdf8;
      padding: 12px;
      display: grid;
      gap: 10px;
      min-width: 0;
    }
    .outcome-card h3 {
      font-size: 17px;
      line-height: 1.2;
    }
    .outcome-card p {
      margin: 0;
    }
    .outcome-facts {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .outcome-speedups .kpi strong {
      font-size: 19px;
    }
    .ai-brief {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffdf8;
      padding: 12px;
      display: grid;
      gap: 8px;
    }
    .ai-brief h3 {
      font-size: 13px;
      line-height: 1.2;
      text-transform: uppercase;
      color: var(--muted);
    }
    .ai-brief ul {
      display: grid;
      gap: 7px;
      padding-left: 18px;
    }
    .ai-brief li {
      color: var(--muted);
      line-height: 1.35;
    }
    .ai-brief strong {
      color: var(--ink);
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
      font-weight: 800;
    }
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
    .note-list, .citations {
      display: grid;
      gap: 8px;
      font-size: 12px;
    }
    .aside-panel.collapsible {
      display: block;
    }
    .aside-panel.collapsible summary {
      padding: 0;
      font-size: 18px;
      font-weight: 800;
    }
    .aside-panel.collapsible .note-list,
    .aside-panel.collapsible .citations {
      margin-top: 12px;
    }
    .note, .citation {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      overflow-wrap: anywhere;
    }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
      }
    }
    @media (max-width: 980px) {
      .hero { min-height: auto; padding-top: 28px; }
      .autocomplete, .decision-board, .report-layout { grid-template-columns: 1fr; }
      .metric-grid, .decision-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
      .app-shell { width: min(100vw - 24px, 1180px); }
      .prompt-row { grid-template-columns: 1fr; }
      .prompt-row input { min-height: 50px; padding: 0; }
      .composer-meta, .headline, .branch-head {
        align-items: flex-start;
        flex-direction: column;
      }
      .route-flow {
        grid-template-columns: 1fr;
      }
      .route-edge {
        width: 2px;
        height: 18px;
        justify-self: center;
      }
      .route-edge::after {
        right: -4px;
        top: auto;
        bottom: -2px;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 7px solid var(--line);
        border-bottom: 0;
      }
      .boundary-grid, .method-kpis, .metric-grid, .decision-summary { grid-template-columns: 1fr; }
      .outcome-grid, .outcome-facts { grid-template-columns: 1fr; }
      .method-title { grid-template-columns: 1fr; }
      .selected-model-row {
        align-items: flex-start;
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#report">Skip to Report</a>
  <div class="app-shell">
    <header class="hero">
      <div class="hero-copy">
        <p class="eyebrow">MLX Model Advisor</p>
        <h1>Pick a base model.</h1>
        <p class="muted">Ports and quantized variants stay hidden by default.</p>
      </div>
      <section class="composer" aria-label="Model Advisor Prompt">
        <form id="search-form" class="prompt-row">
          <label class="sr-only" for="model-search">Model Prompt</label>
          <input id="model-search" name="model" type="search" autocomplete="off" spellcheck="false" placeholder='Try "Llama 3.1 8B", "Whisper large-v3", "Qwen3 14B"...'>
          <button class="primary" type="submit">Analyze</button>
        </form>
        <div class="composer-meta">
          <div id="search-status" aria-live="polite">Loading popular models...</div>
          <label class="toggle" for="include-ports"><input id="include-ports" type="checkbox">Show variants</label>
        </div>
        <div id="results" class="autocomplete" role="listbox" aria-label="Model Suggestions"></div>
      </section>
    </header>
    <main id="report" class="report-shell" tabindex="-1">
      <section id="workspace" class="empty-state">Choose a model to open its MLX plan.</section>
    </main>
  </div>
  <script>
    const state = { query: "", selected: "", advice: null, timer: null, searchSeq: 0, results: [], includeDerivatives: false };
    const searchForm = document.getElementById("search-form");
    const searchInput = document.getElementById("model-search");
    const includePortsInput = document.getElementById("include-ports");
    const resultsEl = document.getElementById("results");
    const statusEl = document.getElementById("search-status");
    const workspaceEl = document.getElementById("workspace");

    searchForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const query = searchInput.value.trim();
      if (looksLikeModelId(query)) {
        selectModel(query);
        return;
      }
      if (state.results.length) {
        selectModel(state.results[0].id);
      }
    });

    searchInput.addEventListener("input", () => {
      state.query = searchInput.value.trim();
      state.selected = "";
      resultsEl.classList.remove("compact");
      window.clearTimeout(state.timer);
      state.timer = window.setTimeout(() => search(state.query), 220);
    });

    includePortsInput.addEventListener("change", () => {
      state.includeDerivatives = includePortsInput.checked;
      search(state.query);
    });

    initFromUrl();

    async function initFromUrl() {
      const params = new URLSearchParams(window.location.search);
      state.query = params.get("q") || "";
      state.selected = params.get("model") || "";
      state.includeDerivatives = params.get("include_ports") === "1";
      searchInput.value = state.query;
      includePortsInput.checked = state.includeDerivatives;
      await search(state.query, { sync: false });
      if (state.selected) {
        await selectModel(state.selected, false, { sync: false });
      }
    }

    async function search(query, options) {
      const seq = ++state.searchSeq;
      const normalized = query.trim();
      if (normalized.length > 0 && normalized.length < 2) {
        statusEl.textContent = "Type 2+ characters.";
        resultsEl.innerHTML = "";
        state.results = [];
        syncUrl(options);
        return;
      }
      statusEl.textContent = normalized ? "Finding models..." : "Loading popular models...";
      try {
        const url = "/api/search?q=" + encodeURIComponent(normalized) + "&limit=12" + (state.includeDerivatives ? "&include_derivatives=1" : "");
        const data = await getJson(url);
        if (seq !== state.searchSeq || normalized !== searchInput.value.trim()) {
          return;
        }
        state.query = normalized;
        state.results = data.results || [];
        const hidden = data.hiddenDerivatives ? " · " + data.hiddenDerivatives + " hidden" : "";
        const label = state.includeDerivatives
          ? (data.mode === "popular" ? "Popular models" : "Matches")
          : (data.mode === "popular" ? "Popular bases" : "Base matches");
        statusEl.textContent = label + " · " + state.results.length + " shown" + hidden;
        resultsEl.classList.remove("compact");
        resultsEl.innerHTML = state.results.map(renderResult).join("");
        resultsEl.querySelectorAll("[data-model-id]").forEach((button) => {
          button.addEventListener("click", () => selectModel(button.getAttribute("data-model-id")));
        });
        syncUrl(options);
      } catch (error) {
        statusEl.textContent = error instanceof Error ? error.message : "Search failed. Try again.";
        resultsEl.innerHTML = "";
        state.results = [];
      }
    }

    async function selectModel(id, ai, options) {
      if (!id) return;
      state.selected = id;
      syncUrl(options);
      workspaceEl.className = "empty-state";
      workspaceEl.textContent = "Loading " + id + "...";
      setSelectedModelRow(id);
      resultsEl.querySelectorAll(".suggestion").forEach((button) => {
        button.classList.toggle("active", button.getAttribute("data-model-id") === id);
        button.setAttribute("aria-selected", button.getAttribute("data-model-id") === id ? "true" : "false");
      });
      try {
        const data = await getJson("/api/advice?id=" + encodeURIComponent(id) + (ai ? "&ai=1" : ""));
        state.advice = data;
        workspaceEl.className = "report-body";
        workspaceEl.innerHTML = renderAdvice(data);
        wireWorkspace();
        document.getElementById("report").focus({ preventScroll: true });
        workspaceEl.scrollIntoView({ block: "start", behavior: "smooth" });
      } catch (error) {
        workspaceEl.className = "empty-state";
        workspaceEl.textContent = error instanceof Error ? error.message : "Could not load that model.";
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
      workspaceEl.querySelectorAll("[data-query]").forEach((button) => {
        button.addEventListener("click", () => {
          searchInput.value = button.getAttribute("data-query") || "";
          state.selected = "";
          resultsEl.classList.remove("compact");
          search(searchInput.value.trim());
          searchInput.focus();
        });
      });
    }

    async function getJson(path) {
      const response = await fetch(path);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || "Request failed with " + response.status + ".");
      }
      return data;
    }

    function syncUrl(options) {
      if (options && options.sync === false) return;
      const params = new URLSearchParams();
      if (state.query) params.set("q", state.query);
      if (state.selected) params.set("model", state.selected);
      if (state.includeDerivatives) params.set("include_ports", "1");
      const next = window.location.pathname + (params.toString() ? "?" + params.toString() : "");
      window.history.replaceState(null, "", next);
    }

    function renderResult(model) {
      const kind = model.repositoryKind === "derivative" ? "derivative" : "base";
      const metadata = compact([model.pipelineTag || model.libraryName || "model", model.family, formatDate(model.lastModified)]).join(" · ");
      const tags = [
        '<span class="status ' + kind + '">' + (kind === "base" ? "Base" : "Variant") + '</span>',
        '<span class="tag">' + formatNumber(model.downloads) + ' downloads</span>',
        '<span class="tag">' + escapeHtml(model.route || "standalone-mlx") + '</span>'
      ].join("");
      return '<button class="suggestion" role="option" aria-selected="' + (state.selected === model.id ? "true" : "false") + '" data-model-id="' + escapeAttr(model.id) + '">' +
        '<span><strong>' + escapeHtml(model.id) + '</strong><span class="muted">' + escapeHtml(metadata) + '</span></span>' +
        '<span class="suggestion-meta">' + tags + '</span>' +
      '</button>';
    }

    function setSelectedModelRow(id) {
      const model = state.results.find((item) => item.id === id);
      const metadata = model ? compact([model.pipelineTag || model.libraryName || "model", model.family, model.route]).join(" · ") : "Selected model";
      resultsEl.classList.add("compact");
      resultsEl.innerHTML = '<div class="selected-model-row">' +
        '<span><strong>' + escapeHtml(id) + '</strong><span class="muted">' + escapeHtml(metadata) + '</span></span>' +
        '<button class="secondary" type="button" data-change-model>Change</button>' +
      '</div>';
      const changeButton = resultsEl.querySelector("[data-change-model]");
      if (changeButton) {
        changeButton.addEventListener("click", () => {
          state.selected = "";
          resultsEl.classList.remove("compact");
          search(state.query);
          searchInput.focus();
        });
      }
      statusEl.textContent = "Selected " + id + ".";
    }

    function renderAdvice(data) {
      if (data.error) {
        return '<section class="empty-state">' + escapeHtml(data.error) + '</section>';
      }
      const model = data.model;
      const advisor = data.advisor;
      const ai = data.aiSummary || {};
      return '<section class="report-layout">' +
        '<div class="report-main">' +
          renderReportHeader(model, advisor, ai) +
          renderOutcomePanel(advisor.modelOutcomes || [], advisor.topCoverage || {}) +
          renderDecisionBoard(model, advisor) +
          renderBranches(advisor) +
        '</div>' +
        '<aside class="report-aside">' +
          renderInstructionPanel(advisor.instructions) +
          renderNotes(advisor.researchNotes) +
          renderCitations(advisor.citations) +
        '</aside>' +
      '</section>';
    }

    function renderReportHeader(model, advisor, ai) {
      const baseModel = readableValue(model.cardData && model.cardData.base_model);
      const tags = compact([
        model.pipelineTag,
        model.libraryName,
        model.repositoryKind === "derivative" ? "Variant" : "Base",
        baseModel ? "Base: " + baseModel : ""
      ]).map((tag) => '<span class="tag">' + escapeHtml(tag) + '</span>').join("");
      const aiBlock = ai.status === "ok"
        ? renderAiBrief(ai.text)
        : '<button class="secondary" data-ai-summary>Generate brief</button>' + (ai.status === "error" ? '<p class="muted">' + escapeHtml(ai.error || "Brief failed.") + '</p>' : '');
      return '<section class="report-header">' +
        renderDerivativeWarning(model) +
        '<div class="headline"><div><h2>' + escapeHtml(answerLine(model, advisor)) + '</h2><p class="muted">' + escapeHtml(model.id + " · " + (advisor.family.label || advisor.family.id)) + '</p></div><span class="status ' + advisor.confidence + '">' + escapeHtml(humanize(advisor.confidence)) + '</span></div>' +
        '<div class="metric-grid">' +
          metric("Route", advisor.family.targets[0] || "standalone-mlx") +
          metric("Family", advisor.family.id) +
          metric("Downloads", formatNumber(model.downloads)) +
          metric("Runbook", advisor.family.runbook) +
        '</div>' +
        renderDecisionSummary(advisor) +
        '<div class="suggestion-meta">' + tags + '</div>' +
        aiBlock +
      '</section>';
    }

    function renderDecisionSummary(advisor) {
      const method = firstActionMethod(advisor);
      const gate = method ? first(method.validationGates) : first(advisor.defaults.keepGates);
      const firstCommand = advisor.instructions.inspect;
      const speedup = advisor.speedupSummary || {};
      return '<div class="decision-summary">' +
        kpi("Route", advisor.family.targets[0] || "standalone-mlx") +
        kpi("Potential", speedup.overallRange || "1.0x-1.0x") +
        kpi("Spec decode", speedup.speculativeRange || "1.0x-1.0x") +
        kpi("First check", gate || "parity + benchmark") +
        '<div class="kpi"><span>CLI</span><button class="secondary" type="button" data-copy="' + escapeAttr(firstCommand) + '">Copy command</button></div>' +
      '</div>';
    }

    function renderAiBrief(text) {
      const items = aiBriefItems(text);
      if (!items.length) {
        return "";
      }
      return '<div class="ai-brief"><h3>Brief</h3><ul>' +
        items.map((item) => '<li>' + renderInlineMarkdown(item) + '</li>').join("") +
      '</ul></div>';
    }

    function aiBriefItems(text) {
      const normalized = String(text || "")
        .replace(/\\r/g, "\\n")
        .replace(/\\n\\s*[-*]\\s+/g, "\\n- ")
        .trim();
      if (!normalized) {
        return [];
      }
      let items = normalized.split(/\\n+/).map(cleanAiBullet).filter(Boolean);
      if (items.length <= 1) {
        items = normalized.split(/\\s+-\\s+(?=\\*\\*|[A-Z0-9])/).map(cleanAiBullet).filter(Boolean);
      }
      return items.slice(0, 5);
    }

    function cleanAiBullet(value) {
      return String(value || "")
        .split(String.fromCharCode(96)).join("")
        .replace(/^[-*]\\s+/, "")
        .replace(/^\\d+[.)]\\s+/, "")
        .replace(/\\s+/g, " ")
        .trim();
    }

    function renderInlineMarkdown(value) {
      return escapeHtml(value)
        .replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
    }

    function renderOutcomePanel(outcomes, coverage) {
      const items = Array.isArray(outcomes) ? outcomes : [];
      if (!items.length) {
        return "";
      }
      const snapshot = coverage && coverage.modelCount
        ? '<span class="tag">' + escapeHtml(String(coverage.coveredCount) + "/" + String(coverage.modelCount) + " top snapshot covered") + '</span>'
        : "";
      const rank = coverage && coverage.rank ? '<span class="tag">HF rank #' + escapeHtml(String(coverage.rank)) + '</span>' : "";
      return '<section class="outcome-panel">' +
        '<div class="outcome-head"><div><h2>Known outcomes</h2><p class="muted">Potential bands, proof boundaries, and gaps.</p></div><div class="suggestion-meta">' + rank + snapshot + '</div></div>' +
        '<div class="outcome-grid">' + items.map(renderOutcomeCard).join("") + '</div>' +
      '</section>';
    }

    function renderOutcomeCard(outcome) {
      const speedup = outcome.potentialSpeedup || {};
      const overall = speedup.overall || {};
      const speculative = speedup.speculativeDecoding || {};
      const needs = first(overall.appliesWhen) || first(speculative.appliesWhen) || outcome.claimBoundary || "run the validation gate";
      const basis = overall.basis || speculative.basis || outcome.claimBoundary || "No reviewed speedup basis.";
      return '<article class="outcome-card">' +
        '<div class="method-title"><h3>' + escapeHtml(outcome.label) + '</h3><span class="impact-pill ' + escapeAttr(outcome.tone || "") + '">' + escapeHtml(outcome.statusLabel || humanize(outcome.status)) + '</span></div>' +
        '<div class="outcome-facts outcome-speedups">' +
          kpi("Potential", overall.range || "1.0x-1.0x") +
          kpi("Speculative", speculative.range || "1.0x-1.0x") +
        '</div>' +
        '<p class="muted">' + escapeHtml(outcome.summary) + '</p>' +
        '<p class="muted"><strong>Needs:</strong> ' + escapeHtml(needs) + '</p>' +
        '<p class="muted"><strong>Basis:</strong> ' + escapeHtml(basis) + '</p>' +
        '<p class="muted"><strong>Next:</strong> ' + escapeHtml(outcome.nextValidation || outcome.claimBoundary || "Run parity and benchmark gates.") + '</p>' +
        '<details class="method-more"><summary>Worked and limits</summary><div class="method-body">' +
          list("Worked", outcome.worked) +
          list("Limit", outcome.didNotWork) +
        '</div></details>' +
      '</article>';
    }

    function renderDecisionBoard(model, advisor) {
      const stats = reportStats(advisor);
      const speedup = advisor.speedupSummary || {};
      return '<section class="decision-board">' +
        '<article class="visual">' +
          '<div><h2>Route</h2><p class="muted">Family -> parity -> benchmark.</p></div>' +
          '<div class="route-flow">' +
            routeNode("Model", model.repositoryKind === "derivative" ? "Find base" : "Base") +
            '<span class="route-edge" aria-hidden="true"></span>' +
            routeNode("MLX", advisor.family.targets[0] || "standalone-mlx") +
            '<span class="route-edge" aria-hidden="true"></span>' +
            routeNode("Gate", advisor.defaults.keepGates[0] || "parity + benchmark") +
          '</div>' +
        '</article>' +
        '<article class="visual">' +
          '<div><h2>Evidence</h2><p class="muted">Ready vs experimental.</p></div>' +
          '<div class="bar-list">' +
            evidenceBar("Validated", stats.validated, stats.total, "") +
            evidenceBar("Benchmark", stats.benchmark, stats.total, "benchmark") +
            evidenceBar("Experimental", stats.experimental, stats.total, "experimental") +
            evidenceBar("Rejected", stats.rejected, stats.total, "rejected") +
          '</div>' +
        '</article>' +
        '<article class="visual">' +
          '<div><h2>Claims</h2><p class="muted">Only measured numbers count.</p></div>' +
          '<div class="boundary-grid">' +
            boundary("Potential", speedup.overallRange || stats.bestImpact, speedup.overallConfidence || stats.bestImpactLabel) +
            boundary("Speculative", speedup.speculativeRange || "1.0x-1.0x", speedup.speculativeConfidence || "potential") +
            boundary("Profiles", String(stats.profileRequired), "local timing") +
          '</div>' +
        '</article>' +
        '<article class="visual">' +
          '<div><h2>Risk</h2><p class="muted">Validate before work.</p></div>' +
          renderRiskMatrix(stats) +
        '</article>' +
      '</section>';
    }

    function renderBranches(advisor) {
      return '<section class="branch-list">' +
        '<div><h2>Methods</h2><p class="muted">Grouped by readiness.</p></div>' +
        advisor.buckets.map(renderBucket).join("") +
      '</section>';
    }

    function renderBucket(bucket) {
      const items = bucket.items || [];
      const prompt = bucket.requiresUserOptIn ? '<p class="muted"><strong>' + escapeHtml(bucket.prompt) + '</strong></p>' : "";
      return '<section class="branch">' +
        '<div class="branch-head"><div><h3>' + escapeHtml(bucketTitle(bucket)) + '</h3><p class="muted">' + escapeHtml(bucketBlurb(bucket)) + '</p></div><span class="status bucket-' + escapeAttr(bucket.id) + '">' + items.length + '</span></div>' +
        prompt +
        (items.length ? items.map(renderMethod).join("") : '<p class="muted">None for this family.</p>') +
      '</section>';
    }

    function renderMethod(method) {
      const gate = first(method.validationGates);
      const rollback = first(method.rollbackConditions);
      const impactClass = isNumberedImpact(method.impact.value) ? "source" : method.impact.value.includes("No portable") ? "boundary" : "";
      const optIn = method.advisorBucket === "experimental-approach" ? '<span class="status bucket-experimental-approach">Experimental</span>' : "";
      return '<article class="method-card">' +
        '<div class="method-title"><strong>' + escapeHtml(methodLabel(method.id)) + '</strong><span class="impact-pill ' + impactClass + '">' + escapeHtml(displayImpact(method.impact.value)) + '</span></div>' +
        '<div class="suggestion-meta">' + optIn + '<span class="tag">' + escapeHtml(method.category) + '</span><span class="tag">' + escapeHtml(method.status) + '</span></div>' +
        '<p class="muted">' + escapeHtml(method.recommendation) + '</p>' +
        '<div class="method-kpis">' +
          kpi("When", method.certainty) +
          kpi("Check", gate || "define parity gate") +
          kpi("Rollback", rollback || "write before trying") +
        '</div>' +
        '<p class="muted"><strong>' + escapeHtml(method.impact.label) + ':</strong> ' + escapeHtml(method.impact.caveat) + '</p>' +
        '<div class="copy-row"><button class="secondary" data-copy="' + escapeAttr("Method: " + method.id + "\\nExpected impact: " + method.impact.value + "\\nFirst validation: " + (gate || "define parity gate") + "\\nRollback trigger: " + (rollback || "write before trying")) + '">Copy method</button></div>' +
        '<details class="method-more"><summary>More</summary><div class="method-body">' +
          list("Checks", method.validationGates) +
          list("Rollback", method.rollbackConditions) +
          list("Tradeoffs", method.tradeoffs) +
        '</div></details>' +
      '</article>';
    }

    function bucketTitle(bucket) {
      const labels = {
        "validated-locally": "Local",
        "validated-source-theory": "Source-backed",
        "benchmark-required": "Measure",
        "experimental-approach": "Experimental",
        "rejected-do-not-use": "Rejected"
      };
      return labels[bucket.id] || bucket.label;
    }

    function bucketBlurb(bucket) {
      const labels = {
        "validated-locally": "Already reproduced.",
        "validated-source-theory": "Known route or source-backed method.",
        "benchmark-required": "Numbers need your workload.",
        "experimental-approach": "Try only with opt-in.",
        "rejected-do-not-use": "Do not use."
      };
      return labels[bucket.id] || bucket.description;
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
      return '<section class="aside-panel section">' +
        '<h2>CLI</h2>' +
        '<pre>' + escapeHtml(text) + '</pre>' +
        '<div class="copy-row"><button class="primary" data-copy="' + escapeAttr(text) + '">Copy steps</button></div>' +
      '</section>';
    }

    function renderNotes(notes) {
      if (!notes || !notes.length) return "";
      return '<details class="aside-panel section collapsible"><summary>Notes</summary><div class="note-list">' +
        notes.map((note) => '<article class="note"><strong>' + escapeHtml(note.id) + '</strong><p>' + escapeHtml(note.summary) + '</p><p class="muted">' + escapeHtml(note.validationGate || note.reasonHeld || "") + '</p></article>').join("") +
      '</div></details>';
    }

    function renderCitations(citations) {
      return '<details class="aside-panel section collapsible"><summary>Sources</summary><div class="citations">' +
        citations.map((source) => '<article class="citation"><strong>' + escapeHtml(source.title) + '</strong><br><a href="' + escapeAttr(source.url) + '" target="_blank" rel="noreferrer">' + escapeHtml(source.url) + '</a><br><span class="muted">' + escapeHtml(source.kind) + ' · ' + escapeHtml(source.reviewDepth || "reviewed") + '</span></article>').join("") +
      '</div></details>';
    }

    function renderDerivativeWarning(model) {
      if (model.repositoryKind !== "derivative") return "";
      const base = readableValue(model.cardData && model.cardData.base_model) || baseQueryFromText(model.id);
      return '<div class="derivative-warning">' +
        '<strong>Variant detected.</strong>' +
        '<p class="muted">Start from the base model.</p>' +
        (base ? '<button class="secondary" type="button" data-query="' + escapeAttr(base) + '">Find base</button>' : '') +
      '</div>';
    }

    function answerLine(model, advisor) {
      const stats = reportStats(advisor);
      const route = advisor.family.targets[0] || "standalone-mlx";
      const family = advisor.family.label || humanize(advisor.family.id);
      const impact = stats.bestImpact;
      const knownRoute = (advisor.modelOutcomes || []).some((outcome) => outcome.status === "local_reproduced" || outcome.status === "source_backed_working");
      const potentialRange = advisor.speedupSummary && advisor.speedupSummary.overallRange;
      const speed = potentialRange && !isFlatSpeedupRange(potentialRange) ? "potential " + potentialRange : knownRoute ? "known route" : isNumberedImpact(impact) ? "reported result" : "measure to claim";
      return route + " · " + family + " · " + speed;
    }

    function firstActionMethod(advisor) {
      const preferred = ["validated-locally", "validated-source-theory", "benchmark-required"];
      for (const id of preferred) {
        const bucket = advisor.buckets.find((item) => item.id === id);
        if (bucket && bucket.items && bucket.items.length) {
          return bucket.items[0];
        }
      }
      return null;
    }

    function methodLabel(id) {
      const labels = {
        "fast-sdpa": "Fast Attention Path",
        "uniform-kv-quantization": "KV Cache Quantization",
        "draft-model-speculation": "Speculative Decoding",
        "compile-stable-region": "Compile Stable Region",
        "lazy-eval-boundaries": "Lazy Eval Boundaries",
        "native-low-bit-weight-quantization": "Low-Bit Weight Quantization",
        "generic-audio-prefix-cache": "Audio Prefix Cache Experiment",
        "prompt-prefix-cache": "Prompt Prefix Cache",
        "content-prefix-cache-vlm": "Vision Content Prefix Cache",
        "multimodal-content-prefix-cache": "Multimodal Content Prefix Cache"
      };
      return labels[id] || humanize(id);
    }

    function displayImpact(value) {
      return value === "Profile-required" ? "Measure to claim" : value;
    }

    function baseQueryFromText(value) {
      return String(value || "")
        .replace(/^[^/]+\\//, "")
        .replace(/whisperkit/ig, "whisper")
        .replace(/mlx|coreml|gguf|ggml|onnx|openvino|tflite|4bit|8bit|quantized|gptq|awq|lora|adapter|int4|int8/ig, " ")
        .replace(/[-_]+/g, " ")
        .replace(/\\s+/g, " ")
        .trim();
    }

    function reportStats(advisor) {
      const all = advisor.buckets.flatMap((bucket) => (bucket.items || []).map((item) => ({ ...item, bucketId: bucket.id })));
      const total = Math.max(all.length, 1);
      const categories = new Map();
      all.forEach((item) => categories.set(item.category, (categories.get(item.category) || 0) + 1));
      const numbered = all.filter((item) => isNumberedImpact(item.impact.value));
      const profile = all.filter((item) => item.impact.value === "Profile-required");
      const firstImpact = numbered[0] || profile[0] || all[0];
      return {
        total,
        validated: bucketCount(advisor, "validated-locally") + bucketCount(advisor, "validated-source-theory"),
        benchmark: bucketCount(advisor, "benchmark-required") + profile.length,
        experimental: bucketCount(advisor, "experimental-approach"),
        rejected: bucketCount(advisor, "rejected-do-not-use"),
        profileRequired: profile.length,
        bestImpact: firstImpact ? displayImpact(firstImpact.impact.value) : "None",
        bestImpactLabel: firstImpact ? firstImpact.impact.label : "no matching methods",
        categories: [...categories.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5)
      };
    }

    function bucketCount(advisor, id) {
      const bucket = advisor.buckets.find((item) => item.id === id);
      return bucket && bucket.items ? bucket.items.length : 0;
    }
    function routeNode(label, value) {
      return '<div class="route-node"><span>' + escapeHtml(label) + '</span><strong>' + escapeHtml(value) + '</strong></div>';
    }
    function evidenceBar(label, count, total, tone) {
      const width = Math.max(4, Math.round((count / Math.max(total, 1)) * 100));
      return '<div class="bar-row"><div class="bar-label"><span>' + escapeHtml(label) + '</span><strong>' + count + '</strong></div><div class="bar-track"><div class="bar-fill ' + tone + '" style="--w:' + width + '%"></div></div></div>';
    }
    function renderCategoryBars(categories, total) {
      if (!categories.length) return '<p class="muted">No methods.</p>';
      return categories.map(([label, count]) => evidenceBar(humanize(label), count, total, "")).join("");
    }
    function renderRiskMatrix(stats) {
      return '<div class="boundary-grid">' +
        boundary("Low risk", String(stats.validated), "source-backed") +
        boundary("Benchmark", String(stats.profileRequired), "measure") +
        boundary("Experimental", String(stats.experimental), "explicit opt-in") +
        boundary("Rejected", String(stats.rejected), "do not use") +
      '</div>';
    }
    function boundary(label, value, detail) {
      return '<div class="boundary"><strong>' + escapeHtml(value) + '</strong><span>' + escapeHtml(label + " · " + detail) + '</span></div>';
    }
    function metric(label, value) {
      return '<div class="metric"><span>' + escapeHtml(label) + '</span><strong>' + escapeHtml(String(value || "")) + '</strong></div>';
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
    function isNumberedImpact(value) {
      return /^\\d|^up to|^about/i.test(value || "");
    }
    function isFlatSpeedupRange(value) {
      return /^1(?:\\.0)?x\\s*-\\s*1(?:\\.0)?x/i.test(value || "");
    }
    function formatNumber(value) {
      return new Intl.NumberFormat().format(value || 0);
    }
    function formatDate(value) {
      if (!value) return "";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return "";
      return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(date);
    }
    function humanize(value) {
      return String(value || "").replace(/[-_]/g, " ").replace(/\\b\\w/g, (char) => char.toUpperCase());
    }
    function readableValue(value) {
      if (Array.isArray(value)) return value.filter(Boolean).join(", ");
      if (typeof value === "string") return value;
      return "";
    }
    function compact(values) {
      return values.filter((value) => value !== undefined && value !== null && String(value).trim() !== "");
    }
    function looksLikeModelId(value) {
      return /^[A-Za-z0-9][A-Za-z0-9._-]*\\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value);
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char] || char));
    }
    function escapeAttr(value) {
      return escapeHtml(value).replace(/\\n/g, "&#10;");
    }
    window.renderAdvice = renderAdvice;
  </script>
</body>
</html>`;
}
