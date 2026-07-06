#!/usr/bin/env node
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const scriptPath = fileURLToPath(import.meta.url);
const appRoot = path.resolve(path.dirname(scriptPath), "..");
const repoRoot = path.resolve(appRoot, "..", "..");
const fixturePath = path.join(repoRoot, "tests", "fixtures", "stack_compose_case.json");
const tempDir = await mkdtemp(path.join(os.tmpdir(), "advisor-stack-fixture-"));
const bundlePath = path.join(tempDir, "worker.mjs");

try {
  await build({
    entryPoints: [path.join(appRoot, "src", "index.ts")],
    outfile: bundlePath,
    bundle: true,
    format: "esm",
    platform: "neutral",
    logLevel: "silent"
  });

  const worker = await import(`${pathToFileURL(bundlePath).href}?t=${Date.now()}`);
  assert.equal(typeof worker.composeStackBand, "function", "composeStackBand export missing");

  const fixture = JSON.parse(await readFile(fixturePath, "utf8"));
  const actual = worker.composeStackBand(fixture.stack, fixture.guidance_methods);
  assert.deepEqual(actual, fixture.expected);
  console.log("ok stack fixture parity passed");
} finally {
  await rm(tempDir, { recursive: true, force: true });
}
