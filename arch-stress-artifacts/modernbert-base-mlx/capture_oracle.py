from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, ModernBertModel


FIXTURE_SENTENCE = (
    "ModernBERT parity on Apple Metal must preserve rotary positions, "
    "bidirectional local attention, padding masks, layer normalization, "
    "and GeGLU residuals. "
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=160)
    parser.add_argument("--text-repeats", type=int, default=5)
    parser.add_argument("--text")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    fixture_text = args.text if args.text is not None else (FIXTURE_SENTENCE * args.text_repeats).strip()
    encoded = tokenizer(
        fixture_text,
        max_length=args.sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(dtype=torch.long, device="cpu")
    attention_mask = encoded["attention_mask"].to(dtype=torch.long, device="cpu")

    model = ModernBertModel.from_pretrained(
        args.model,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device="cpu", dtype=torch.float32)
    model.eval()

    embedding_outputs: list[torch.Tensor] = []
    layer_outputs: list[torch.Tensor | None] = [None] * len(model.layers)

    def capture_embedding(_module, _inputs, output):
        embedding_outputs.append(output.detach().cpu().float())

    def capture_layer(index: int):
        def hook(_module, _inputs, output):
            layer_outputs[index] = output.detach().cpu().float()

        return hook

    handles = [model.embeddings.register_forward_hook(capture_embedding)]
    handles.extend(layer.register_forward_hook(capture_layer(i)) for i, layer in enumerate(model.layers))
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    if len(embedding_outputs) != 1:
        raise RuntimeError(f"expected one embedding capture, got {len(embedding_outputs)}")
    if len(layer_outputs) != 22 or any(value is None for value in layer_outputs):
        missing = [i for i, value in enumerate(layer_outputs) if value is None]
        raise RuntimeError(f"expected 22 layer captures; missing={missing}")
    if output.hidden_states is None or len(output.hidden_states) != 23:
        raise RuntimeError(
            f"expected output_hidden_states tuple of length 23, got "
            f"{None if output.hidden_states is None else len(output.hidden_states)}"
        )
    if input_ids.shape != (1, args.sequence_length):
        raise RuntimeError(f"unexpected fixture shape {tuple(input_ids.shape)}")
    if int(input_ids.max()) >= model.config.vocab_size or int(input_ids.min()) < 0:
        raise RuntimeError("fixture contains an out-of-vocabulary token id")
    if not torch.equal(input_ids[attention_mask == 0], torch.full_like(input_ids[attention_mask == 0], 50283)):
        raise RuntimeError("masked fixture positions are not ModernBERT pad tokens")

    arrays: dict[str, np.ndarray] = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "embed": embedding_outputs[0].numpy(),
        "final_norm": output.last_hidden_state.detach().cpu().float().numpy(),
    }
    for i, value in enumerate(layer_outputs):
        assert value is not None
        arrays[f"layer.{i}.hidden"] = value.numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "input_shape": list(input_ids.shape),
                "valid_tokens": int(attention_mask.sum()),
                "captured_layers": len(layer_outputs),
                "hf_hidden_states_len": len(output.hidden_states),
                "attn_implementation": model.config._attn_implementation,
                "dtype": str(output.last_hidden_state.dtype),
                "layer_types": list(model.config.layer_types),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
