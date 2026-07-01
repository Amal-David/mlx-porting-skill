#!/usr/bin/env node
import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";

const scriptPath = fileURLToPath(import.meta.url);
const appRoot = path.resolve(path.dirname(scriptPath), "..");
const outputDir = path.join(appRoot, "dist-pages");

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
await esbuild.build({
  entryPoints: [path.join(appRoot, "src", "index.ts")],
  bundle: true,
  format: "esm",
  target: "es2022",
  platform: "browser",
  outfile: path.join(outputDir, "_worker.js"),
  legalComments: "none"
});
await writeFile(path.join(outputDir, ".assetsignore"), "_worker.js\n", "utf8");
console.log("wrote dist-pages/_worker.js");
