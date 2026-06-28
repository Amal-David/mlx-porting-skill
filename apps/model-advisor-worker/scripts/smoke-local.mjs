#!/usr/bin/env node
const base = process.env.SMOKE_BASE_URL || "http://127.0.0.1:8787";

async function getJson(path) {
  const response = await fetch(new URL(path, base));
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

async function getText(path) {
  const response = await fetch(new URL(path, base));
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}: ${await response.text()}`);
  }
  return response.text();
}

async function getStatus(path, init) {
  const response = await fetch(new URL(path, base), init);
  return {
    status: response.status,
    allowOrigin: response.headers.get("access-control-allow-origin"),
    text: await response.text()
  };
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const html = await getText("/");
assert(html.includes("MLX Model Advisor"), "app shell title missing");
assert(html.includes("Ask for a base model"), "chat-first app shell missing");

const discover = await getJson("/api/search?limit=4");
assert(Array.isArray(discover.results) && discover.results.length > 0, "default discovery returned no models");
assert(discover.results.every((model) => model.repositoryKind !== "derivative"), "default discovery should hide derivative repos");

const search = await getJson("/api/search?q=qwen&limit=4");
assert(Array.isArray(search.results) && search.results.length > 0, "search returned no models");
assert(search.results.every((model) => model.repositoryKind !== "derivative"), "search should hide derivative repos by default");

const exactDerivative = await getJson("/api/search?q=mlx-community%2FQwen2.5-14B-Instruct-4bit&limit=3");
assert(exactDerivative.results.some((model) => model.id === "mlx-community/Qwen2.5-14B-Instruct-4bit"), "exact derivative model id should remain selectable");

const longQuery = await getStatus(`/api/search?q=${"x".repeat(97)}`, { headers: { origin: "https://example.com" } });
assert(longQuery.status === 400, "long search query should be rejected before proxying");
assert(!longQuery.allowOrigin, "cross-origin API calls should not receive permissive CORS headers");

const textAdvice = await getJson("/api/advice?id=mlx-community/Qwen2.5-14B-Instruct-4bit");
assert(textAdvice.advisor.family.id === "dense-decoder-transformer", "Qwen2.5 text model should route to dense decoder");
assert(textAdvice.aiSummary.status === "not_configured" || textAdvice.aiSummary.status === "available", "deterministic advice should not require OpenAI");
assert(textAdvice.advisor.citations.length > 0, "text advice citations missing");

const vlmAdvice = await getJson("/api/advice?id=lmstudio-community/Qwen3.6-27B-MLX-4bit");
assert(vlmAdvice.advisor.family.id === "vision-language-omni", "image-text-to-text model should route to vision-language");
const experimental = vlmAdvice.advisor.buckets.find((bucket) => bucket.id === "experimental-approach");
assert(experimental?.prompt === "This is an experimental approach. Do you want to try it?", "experimental opt-in prompt missing");

const asrAdvice = await getJson("/api/advice?id=argmaxinc/whisperkit-coreml");
const asrExperimental = asrAdvice.advisor.buckets.find((bucket) => bucket.id === "experimental-approach");
assert(asrExperimental?.items.some((item) => item.id === "generic-audio-prefix-cache"), "stt alias should expose generic audio prefix-cache experiment for ASR");

console.log("ok local smoke passed");
