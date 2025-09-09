# Exercise 1 — Solution Walkthrough

This walkthrough highlights the **important parts** of the reference solution for *Exercise_1_solution_revised.ipynb* and skims everything else. Use it to understand how the solution is structured, what to look for in the outputs, and how to validate your own work.

---

## 1) What the solution accomplishes

- Establishes a clean **baseline** for text-generation performance.
- **Static PTQ with Optimum** is implemented to produce a quantized variant suitable for inference.
- **Ablation of `use_cache` and padding** to see how each choice affects throughput/latency.
- **Batched vs single-sample measurement** to observe throughput gains and per-sample latency effects.
- **Perplexity comparison** (baseline vs quantized) to sanity-check quality drift.
- **Results table** built to aggregate metrics across scenarios for quick comparisons.
- **Edge prompts probe** to stress-test prefill-heavy or repetitive inputs.
- **Controller/orchestrator** runs the full pipeline end-to-end and prints a compact summary.

**Why this matters:** The exercise teaches you how to measure generation performance fairly, apply a practical quantization pass, and verify that speedups don’t come at an unacceptable quality cost.

---

## 2) Key implementation choices (the “why”)

### a) Baseline first, then optimize
- The solution measures a **CPU FP32 baseline** (you may also see GPU baselines on capable hardware).  
- It then applies **post‑training quantization** to produce a faster variant (especially relevant on CPU).

> Representative run header:  
> == CPU FP32 Baseline ==

### b) Ablations that actually move the needle
- **KV‑cache**: Turning it **on** should increase **decode throughput** (tokens/sec) after the prefill step.  
- **Padding choices**: Affect attention cost when batching; different padding strategies can shift latency.

### c) Batching for throughput
- Larger **batch sizes** typically **increase total throughput**, though per‑sample latency may not decrease.  
- The solution times batched vs single‑sample runs consistently (same `max_new_tokens`, same prompts).

### d) Quality sanity via perplexity
- **Perplexity (PPX)** is derived from the **average token‑level loss** on a tiny corpus.  
- The solution compares **baseline vs quantized** PPX to verify quality is still acceptable.

### e) One place to compare — a summary table
- Metrics (latency, tokens/sec, ms/token, PPX) are **aggregated** into a compact **[6] Summary table** so you can judge trade‑offs at a glance.

---

## 3) What you should see (outputs)

Your numbers depend on hardware and model size. Look for the **shape** of results and consistent methodology:

- A one‑line **load/baseline header** (model id, device, dtype).  
- A baseline metrics block (e.g., latency in seconds; tokens/sec if reported).  
- Ablation results for **`use_cache`** and **padding** (each with the same prompt and `max_new_tokens`).  
- A batching comparison (batch size vs throughput/latency).  
- A small **PPX** comparison (baseline vs quantized).  
- A final **results table** aggregating the scenarios.

**Representative snippets to expect:**
- Loaded: <model_id> | device=<cpu/gpu> | dtype=<label>
- Example latency (s): 2.702



If any category is missing, re‑run the cell that builds the summary table and ensure you kept parameters **identical** across comparisons.

---

## 4) Interpreting the results (the story to tell)

- **Prefill vs decode:** Long inputs increase **prefill** time; KV‑cache primarily helps during **decode**.  
- **Quantization:** On CPU, PTQ often yields noticeable **latency reductions**; verify PPX to keep quality in check.  
- **Batching:** Great for **throughput**, not always for per‑sample latency.  
- **Fair comparisons:** Keep `max_new_tokens`, prompts, and cache settings fixed across scenarios.

---

## 5) Troubleshooting aligned with the solution

- **Tokenizer/model mismatch:** Load both from the **same** repo to avoid loss spikes.  
- **GPU timing undercount:** If you add GPU runs, synchronize right before/after timing.  
- **Perplexity looks wrong:** Ensure you compute a **token‑weighted average loss** across the dataset before `PPX = exp(loss)`.  
- **No speedup from quantization:** Typical on GPU; PTQ benefits show up best on **CPU**.  
- **Batching regressions:** Verify padding/truncation and that your collation keeps tensors aligned.

---

### Takeaway

The solution prioritizes **reliable measurement** and **fair comparisons**: start with a baseline, add PTQ, ablate cache/padding, test batching, and validate quality with PPX — then consolidate everything into a single, readable table with a short summary.

If your notebook reproduces this flow and your outputs tell the same story (even with different absolute numbers), you’re aligned with the reference solution.
