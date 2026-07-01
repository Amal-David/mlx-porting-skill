#!/usr/bin/env node
import vm from "node:vm";

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

const malformedJson = await getStatus("/api/advice?id=mlx-community/Qwen2.5-14B-Instruct-4bit", {
  method: "POST",
  headers: {
    "content-type": "application/json",
    origin: "http://127.0.0.1:8787"
  },
  body: "{"
});
assert(malformedJson.status === 400, "malformed JSON body should return 400");

const oversizedJson = await getStatus("/api/advice?id=mlx-community/Qwen2.5-14B-Instruct-4bit", {
  method: "POST",
  headers: {
    "content-type": "application/json",
    origin: "http://127.0.0.1:8787"
  },
  body: JSON.stringify({ id: "mlx-community/Qwen2.5-14B-Instruct-4bit", filler: "x".repeat(5000) })
});
assert(oversizedJson.status === 413, "oversized JSON body should return 413");

const textAdvice = await getJson("/api/advice?id=mlx-community/Qwen2.5-14B-Instruct-4bit");
assert(textAdvice.advisor.family.id === "dense-decoder-transformer", "Qwen2.5 text model should route to dense decoder");
assert(textAdvice.aiSummary.status === "not_configured" || textAdvice.aiSummary.status === "available", "deterministic advice should not require OpenAI");
assert(textAdvice.advisor.citations.length > 0, "text advice citations missing");
assert(textAdvice.advisor.modelOutcomes.some((item) => item.id === "decoder-mlx-lm-working-route"), "decoder working-route outcome missing");
assert(textAdvice.advisor.topCoverage.modelCount >= 250, "top model coverage snapshot missing");

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

const vlmAdvice = await getJson("/api/advice?id=lmstudio-community/Qwen3.6-27B-MLX-4bit");
assert(vlmAdvice.advisor.family.id === "vision-language-omni", "image-text-to-text model should route to vision-language");
const experimental = vlmAdvice.advisor.buckets.find((bucket) => bucket.id === "experimental-approach");
assert(experimental?.prompt === "This is an experimental approach. Do you want to try it?", "experimental opt-in prompt missing");

const asrAdvice = await getJson("/api/advice?id=argmaxinc/whisperkit-coreml");
const asrExperimental = asrAdvice.advisor.buckets.find((bucket) => bucket.id === "experimental-approach");
assert(asrExperimental?.items.some((item) => item.id === "generic-audio-prefix-cache"), "stt alias should expose generic audio prefix-cache experiment for ASR");

console.log("ok local smoke passed");
