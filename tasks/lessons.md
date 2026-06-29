# Lessons

- 2026-06-28: The model advisor should be chat-first and base-model-first. Hide existing MLX/CoreML/GGUF/quantized ports by default, reveal them only through an explicit toggle, and use autocomplete plus popular/recent base-model suggestions instead of a browsing dashboard.
- 2026-06-29: Advisor website copy should be compact and scannable. Prefer short labels such as route, family, and benchmark gate over explanatory sentences like "No portable speedup is confirmed until a local benchmark passes."
- 2026-06-29: Generated AI briefs must render as structured UI, not escaped prose. Split bullets, strip raw Markdown/backticks, and keep the brief compact.
- 2026-06-29: Do not make every source-backed route read like "needs local benchmark." Show observed working routes, known gaps, and source-reported outcomes separately; reserve benchmark language for numeric speed or memory claims.
- 2026-06-29: Model-advisor outcome cards must answer the upside question. Show an overall potential speedup range and speculative-decoding range, plus the gates that make the high end plausible; use a flat 1.0x-1.0x range when no credible speedup is known.
