#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const base = process.env.SMOKE_BASE_URL || "http://127.0.0.1:8787";
const generatedDataSource = await readFile(new URL("../src/skill-data.generated.ts", import.meta.url), "utf8");

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

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function registryStackStepCount(id) {
  const pattern = new RegExp(`"id": "${escapeRegExp(id)}"[\\s\\S]*?"steps": \\[([\\s\\S]*?)\\]\\s*,\\s*"compositionNotes"`);
  const match = generatedDataSource.match(pattern);
  assert(match, `${id} stack missing from generated registry`);
  return (match[1].match(/"method":/g) || []).length;
}

function elementMock() {
  return {
    checked: false,
    className: "",
    innerHTML: "",
    textContent: "",
    value: "",
    addEventListener() {},
    focus() {},
    getAttribute() {
      return "";
    },
    querySelector() {
      return null;
    },
    querySelectorAll() {
      return [];
    },
    scrollIntoView() {},
    setAttribute() {},
    classList: {
      add() {},
      remove() {},
      toggle() {}
    }
  };
}

function renderAdviceFromShell(shellHtml, data) {
  const script = shellHtml.match(/<script>([\s\S]*)<\/script>/)?.[1];
  assert(script, "app shell script missing");

  const elements = new Map();
  const context = {
    console,
    document: {
      getElementById(id) {
        if (!elements.has(id)) {
          elements.set(id, elementMock());
        }
        return elements.get(id);
      }
    },
    fetch: async () => ({
      ok: true,
      json: async () => ({ mode: "popular", results: [] })
    }),
    navigator: { clipboard: { writeText: async () => {} } },
    URLSearchParams,
    window: {
      clearTimeout,
      history: { replaceState() {} },
      location: { pathname: "/", search: "" },
      setTimeout
    }
  };
  vm.runInNewContext(script, context, { timeout: 1000 });
  assert(typeof context.window.renderAdvice === "function", "renderAdvice test hook missing");
  return context.window.renderAdvice(data);
}

const html = await getText("/");
assert(html.includes("MLX Model Advisor"), "app shell title missing");
assert(html.includes("Pick a base model"), "chat-first app shell missing");
assert(html.includes("function renderAiBrief"), "AI brief renderer missing");
assert(html.includes("ai-brief"), "AI brief styles missing");

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
assert(textAdvice.advisor.modelOutcomes.some((item) => item.id === "decoder-mlx-lm-working-route"), "decoder working-route outcome missing");
assert(textAdvice.advisor.topCoverage.modelCount >= 250, "top model coverage snapshot missing");
const denseStack = textAdvice.advisor.recommendedStack;
const denseStepCount = registryStackStepCount("dense-decoder-inference");
assert(denseStack?.id === "dense-decoder-inference", "dense decoder advice should include dense-decoder-inference stack");
assert(denseStack.steps.length === denseStepCount, "dense decoder stack step count should match generated registry");
assert(denseStack.compound.hypothesis_ceiling.floor === "1.0x", "dense decoder stack hypothesis floor should remain 1.0x");
assert(denseStack.compound.hypothesis_ceiling.ceiling === "2.64x", "dense decoder stack hypothesis should exclude cross-metric and conflicting bands");
assert(denseStack.compound.hypothesis_ceiling.provenance === "multiplicative_hypothesis", "dense decoder stack product should remain hypothesis provenance");
assert(denseStack.compound.measured?.ratio === "0.21x", "dense decoder stack should surface measured-together ratio");
assert(denseStack.compound.measured?.provenance === "local_reproduced", "dense decoder measured ratio should surface local provenance");
assert(String(denseStack.compound.measured?.caveat || "").includes("plain 4-bit") || String(denseStack.compound.measured?.basis || "").includes("plain 4-bit"), "dense decoder stack should surface plain 4-bit baseline caveat");
const stackCeiling = textAdvice.advisor.speedupSummary.stackCeiling;
assert(stackCeiling?.floor === "1.0x", "speedup summary stack ceiling should expose 1.0x floor");
assert(stackCeiling?.ceiling === "2.64x", "speedup summary stack ceiling should expose corrected hypothesis ceiling");
assert(stackCeiling?.provenance === "multiplicative_hypothesis", "speedup summary stack ceiling should expose hypothesis provenance");
assert(stackCeiling?.measuredRatio === "0.21x", "speedup summary stack ceiling should carry measured stack ratio");
assert(stackCeiling?.measuredProvenance === "local_reproduced", "speedup summary stack ceiling should carry measured provenance separately");

const renderedWithAi = renderAdviceFromShell(html, {
  ...textAdvice,
  aiSummary: {
    status: "ok",
    text: "- **Good MLX-LM fit:** `Qwen/Qwen3-0.6B` is dense decoder transformer. - **Validated optimizations:** use **fast-sdpa**. - **Experimental only:** keep **adaptive KV quantization** separate. - **Next step:** follow `references/runbook-decoder-transformer.md`."
  }
});
const briefHtml = renderedWithAi.match(/<div class="ai-brief">[\s\S]*?<\/div>/)?.[0] || "";
assert(briefHtml.includes("<strong>Good MLX-LM fit:</strong>"), "AI brief should render inline emphasis");
assert((briefHtml.match(/<li>/g) || []).length >= 4, "AI brief should render separate bullets");
assert(!briefHtml.includes("**"), "AI brief should not leak raw Markdown emphasis");
assert(!briefHtml.includes("`"), "AI brief should not leak raw backticks");
assert(!briefHtml.includes("<p"), "AI brief should not render as one paragraph dump");
assert(renderedWithAi.includes("Recommended stack"), "rendered advice should include stack panel");
assert(renderedWithAi.includes("0.21x"), "rendered advice should include measured dense stack headline");
assert(renderedWithAi.includes("1.0x-2.64x"), "rendered advice should include corrected hypothesis dense stack ceiling");
assert(renderedWithAi.includes("Measured together"), "rendered stack ceiling should label the measured-together ratio");
assert(renderedWithAi.includes("plain 4-bit"), "rendered stack ceiling should include plain 4-bit baseline caveat");

const vlmAdvice = await getJson("/api/advice?id=lmstudio-community/Qwen3.6-27B-MLX-4bit");
assert(vlmAdvice.advisor.family.id === "vision-language-omni", "image-text-to-text model should route to vision-language");
const experimental = vlmAdvice.advisor.buckets.find((bucket) => bucket.id === "experimental-approach");
assert(experimental?.prompt === "This is an experimental approach. Do you want to try it?", "experimental opt-in prompt missing");

const asrAdvice = await getJson("/api/advice?id=argmaxinc/whisperkit-coreml");
const asrExperimental = asrAdvice.advisor.buckets.find((bucket) => bucket.id === "experimental-approach");
assert(asrExperimental?.items.some((item) => item.id === "generic-audio-prefix-cache"), "stt alias should expose generic audio prefix-cache experiment for ASR");

console.log("ok local smoke passed");
