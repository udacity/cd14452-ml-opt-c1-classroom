# Exercise 1 — Solution Walkthrough & Expected Outputs

This document explains **how the provided solution notebook addresses each exercise**, what output you should expect, and *why* each step is implemented that way. If you get stuck implementing the starter, read this end‑to‑end and match your work to the behaviors described here.

---

## What this exercise measures

You’ll measure text‑generation **latency** and **throughput** (tokens/sec) under two knobs:

1) **Prompt length**: short → long inputs (holding `max_new_tokens` constant)  
2) **Numeric precision**: `float32` vs `float16` (where supported on GPU)

**Why it matters:** Longer contexts increase prefill work; lower precision can speed up GPU math and memory bandwidth, often improving latency and tokens/sec.

References:
- Hugging Face text generation API: https://huggingface.co/docs/transformers/en/main_classes/text_generation  
- PyTorch inference_mode: https://pytorch.org/docs/stable/generated/torch.inference_mode.html  
- PyTorch AMP / autocast: https://pytorch.org/docs/stable/amp.html#autocast-mode  
- CUDA synchronize: https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html  
- Tokenizer summary: https://huggingface.co/docs/transformers/en/tokenizer_summary

---

## Step 0 — Environment & Model Loading (Solution Behavior)

- Detect device: **GPU if available, else CPU**.  
- Resolve requested precision to a real dtype **with a CPU guard** (see Step 1).  
- Try **a short list of model IDs** in order (the solution falls back if the first choice fails).  
- Load tokenizer + model from the **same model ID**, set `eval()`, and place on the chosen device/dtype.

**What you should see in the solution:**  
A print similar to:  
`Loaded: <model_id_used> | dtype=<dtype_label>`  
For example, on CPU with a float16 request: `dtype=float32 (forced on CPU)`.

**Common pitfalls the solution avoids:**  
- Mismatched tokenizer/model repositories.  
- Silent half‑precision on CPU (falls back to float32 and labels it clearly).

Docs: Auto classes — https://huggingface.co/docs/transformers/en/model_doc/auto

---

## Step 1 — `_resolve_dtype(...)`

**Intent:** Convert a user string (e.g., `"float16"`) into an actual PyTorch dtype **and** a short, user‑facing label.

**What the solution does:**
- Accepts only intended values (`float32`, `float16`, case‑insensitive).  
- If **CPU** and user requested `float16`, it **falls back to float32**.  
- Returns **both**: the dtype **and** a label string. Examples:  
  - GPU + `float16` → `(torch.float16, "float16")`  
  - CPU + `float16` → `(torch.float32, "float32 (forced on CPU)")`  
  - Any + `float32` → `(torch.float32, "float32")`

**How to verify:** On a CPU machine, request `float16` and confirm you get the fallback label.

---

## Step 2 — `load_model_and_tokenizer(...)`

**Intent:** Load a causal LM + tokenizer, move to device, set dtype and eval mode.

**What the solution does:**
- Iterates through **candidate model IDs** (so the notebook works even if your first choice is unavailable).  
- Calls `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` for the chosen ID.  
- Moves model to the resolved dtype/device and sets `eval()`.  
- Returns `(tokenizer, model, model_id_used, dtype_label)`.

**Failure mode handled:** If **all** candidates fail, raises a clear error listing the tried IDs.

**What you should see:** A clean load message and no dtype/type mismatch warnings.

Docs: Auto classes — https://huggingface.co/docs/transformers/en/model_doc/auto

---

## Step 3 — `time_generate(...)`

**Intent:** Time one **text generation** call and compute basic performance metrics.

**What the solution does:**
- Tokenizes the prompt and moves inputs to the **same device** as the model.  
- Performs a **brief warm‑up** generate (not timed) to avoid first‑run overhead in your measurement.  
- Uses a **high‑resolution wall clock** (e.g., `perf_counter`).  
- If on **GPU**, calls `torch.cuda.synchronize()` **immediately before** and **after** timing.  
- Wraps the call in `torch.inference_mode()`.  
- Computes and returns a dictionary with at least:  
  - `total_latency_s`  
  - `tokens_generated`  
  - `tokens_per_sec`  
  - `avg_latency_per_token_ms`

**What you should see:** A metrics dict after the baseline run (e.g., ~50 generated tokens). Exact numbers vary by hardware, model size, and drivers. The important thing is **internal consistency** when you compare settings.

Docs: Generation API — https://huggingface.co/docs/transformers/en/main_classes/text_generation

---

## Exercise A — Prompt‑Length Sweep (`run_for_varied_lengths`) ✔

**Goal:** Measure how latency and throughput change as the **input prompt** grows, holding `max_new_tokens` fixed.

**What the solution does:**
- Chooses several **target token lengths** (e.g., short / medium / long).  
- Builds prompts by **repeating a base string** until the **tokenized** length meets/exceeds each target.  
- For every length:  
  - Calls `time_generate(...)` with the same `max_new_tokens` and `use_cache` setting.  
  - Records metrics.  
- Produces two plots:  
  1. **Latency (s)** vs **prompt length (tokens)**  
  2. **Tokens/sec** vs **prompt length (tokens)**

**Expected outcome / interpretation:**
- **Latency** generally **increases** with longer prompts (more prefill).  
- **Tokens/sec** may **decrease** slightly with longer context (larger KV‑cache / attention cost).  
- The **shape** of the curves matters more than exact values; your environment determines the magnitudes.

**Self‑check:**
- Did you keep `max_new_tokens` constant across all lengths?  
- Do the two plots look reasonable (monotonic-ish latency increase)?  
- Are the x‑axes labeled in **tokens**, not characters?

---

## Exercise B — Precision Sweep (`run_varied_precision`) ✔

**Goal:** Compare **float32** vs **float16** (where GPU supports it) using the same prompt and generation limits.

**What the solution does:**
- Builds a small list of precisions to test.  
- For each precision:  
  - Resolves dtype & label (`_resolve_dtype`).  
  - Uses either **model.to(dtype=...)** once, or an **autocast** context around the generate call.  
  - Calls `time_generate(...)` on a **fixed prompt** with the **same** `max_new_tokens` and cache setting.  
  - Stores the metrics under the precision label.  
- Optionally prints or plots the comparison (table or bar chart).

**Expected outcome / interpretation:**
- On **GPU**, `float16` typically **reduces latency** and **increases tokens/sec** vs `float32`.  
- On **CPU**, a request for `float16` should **fall back to float32**, so results will be **float32** only.

**Self‑check:**
- Are you comparing apples to apples (same prompt, same `max_new_tokens`, same use_cache)?  
- On CPU, is your reported precision label acknowledging the fallback?  
- Does the faster precision produce **comparable text** quality for this short run (sanity‑read a sample)?

Docs: AMP autocast — https://pytorch.org/docs/stable/amp.html#autocast-mode

---

## Interpreting Results — What Story to Tell

- **Prefill vs decode:** Longer prompts increase **prefill** cost (attention over more input tokens), so total latency rises.  
- **KV cache:** With `use_cache=True`, **decoding** new tokens becomes faster after prefill, but cost per token still depends on model size and sequence length.  
- **Precision trade‑offs:** On modern GPUs, half precision usually improves both **math** and **memory bandwidth**, giving better latency/tokens‑per‑second. On CPU, half precision isn’t practically supported → stick to float32.  
- **First‑run effects:** Compilations and caches can distort timings; always warm up and report steady‑state numbers.

---

## Troubleshooting (Matched to Solution Choices)

- **“Half precision on CPU” warnings:** The solution **falls back** to float32 with a clear label; adopt the same behavior.  
- **Tokenizer/model mismatch:** Always load both from the **same** repo ID (solution enforces this).  
- **Inconsistent timings:** Synchronize on GPU; keep generation args identical when comparing.  
- **OOM / slowness:** Use a smaller model, reduce `max_new_tokens`, or run on CPU for correctness checks.  
- **No candidate model loads:** The solution **iterates through a fallback list**; keep that logic in your implementation.

---

## What your notebook should contain at the end

- A printed confirmation of **model ID** and **precision label** actually used.  
- A **baseline metrics dict** from a first timing run.  
- Two **prompt‑length plots**: latency vs tokens, tokens/sec vs tokens.  
- A **precision comparison** (table and/or small plot) with clear labels.  
- A short **written summary** (3–5 bullets) interpreting trends observed in your environment.

---

## Optional Extensions (Aligned with the Solution)

- Try **bf16** on hardware that supports it and compare to `float16`.  
- Compare `use_cache=True` vs `False`.  
- Repeat each run **N times** and average to reduce noise; show **error bars** on plots.  
- Add a small **markdown cell** that explains prefill vs decode time with a simple diagram.

---

### Final checklist

- [ ] Model/tokenizer loaded from the **same** repo and on the intended device/dtype.  
- [ ] `_resolve_dtype` returns a dtype **and** a friendly label; CPU enforces float32.  
- [ ] `time_generate` performs warm‑up, synchronizes on GPU, and returns the 4 key metrics.  
- [ ] Prompt‑length sweep keeps `max_new_tokens` fixed and produces **two plots**.  
- [ ] Precision sweep compares at least `float32` vs `float16` (GPU) or documents CPU fallback.  
- [ ] Results are summarized in a short narrative with correct interpretations.

If your outputs match the descriptions above, your solution aligns with the reference implementation.
