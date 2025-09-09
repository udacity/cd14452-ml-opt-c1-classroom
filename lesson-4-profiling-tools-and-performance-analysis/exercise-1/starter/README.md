# Exercise 1 — Starter Walkthrough
This guide is tailored to **Exercise_1_starter_revised.ipynb**. It highlights the **important parts you must implement**, explains **what good output looks like**, and links to the right documentation. Use this to finish the starter before checking any solution.

> **Theme:** Baseline vs. optimized text‑generation with **quantization** and **inference tricks** (cache, padding, batching), and basic **quality** (perplexity) checks.

---

## 0) What you’re building

By the end, your notebook should:
1. Load a small causal LM + tokenizer (same repo ID), put it in `eval`, and ensure correct device/dtype.
2. Measure baseline generation throughput/latency on a few representative prompts.
3. Apply **post‑training quantization** to create a faster model and re‑measure.
4. Ablate **KV‑cache** and **padding** choices during generation and observe their impact.
5. Measure **batched** vs **single‑sample** performance.
6. Compare **perplexity** (PPX) between baseline and quantized variants to sanity‑check quality.
7. Assemble a compact **results table** and a short **takeaway**.

Keep your runs *short* (tiny models, few tokens) so you can iterate quickly.

---

## 1) Functions to complete (what & why)

Below are the starter’s key functions detected via TODOs and what each should accomplish.

### A. `run_static_ptq_optimum`
**Goal:** Produce a quantized model (INT8 / dynamic) using **Optimum** and verify it runs for generation.
- **Inputs:** A baseline PyTorch causal LM & tokenizer (or a model ID).
- **Process (conceptual):**
  - Create a quantized variant using Optimum (e.g., dynamic quantization or ORT/Intel backends depending on the scaffold).
  - Ensure the quantized model can **load on your device** and **generate** with the same prompt API.
- **Outputs:** A usable quantized model reference + a short info string about backend/precision chosen.
- **Docs:**
  - Optimum (overview): https://huggingface.co/docs/optimum/index
  - Optimum Intel (PyTorch/PTQ): https://huggingface.co/docs/optimum/intel/optimization_guide/ptq
  - Optimum ONNX Runtime: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization
  - PyTorch dynamic quantization (background): https://pytorch.org/docs/stable/quantization.html#dynamic-quantization

**Sanity checks:**
- Quantized model returns text for a simple prompt.
- Throughput improves vs. baseline on CPU; on GPU, effects vary (quantization is usually CPU‑centric).

---

### B. `ablate_cache_and_padding`
**Goal:** Measure how `use_cache` and padding strategy affect throughput/latency.
- **Inputs:** Model, tokenizer, a prompt (or prompt lengths), fixed `max_new_tokens`.
- **Process (conceptual):**
  - Toggle `use_cache=True/False` during generation.
  - Compare padding modes (e.g., “left” vs “right” or packed vs. padded) if the scaffold exposes it.
  - Record **total latency**, **tokens/sec**, and **avg latency per token** for each setting.
- **Outputs:** Small table/dict keyed by the ablation setting → metrics.
- **Docs:**
  - HF text generation config: https://huggingface.co/docs/transformers/en/main_classes/text_generation
  - KV‑cache concept (Transformers guide): https://huggingface.co/docs/transformers/en/perf_infer_gpu_one#key-value-cache

**Expected trend:**
- `use_cache=True` غالبًا yields higher tokens/sec in decoding; padding strategy mostly affects **batch** behavior and attention cost.

---

### C. `measure_batched`
**Goal:** Compare **single‑sample** vs **batched** generation throughput.
- **Inputs:** List of prompts (varying lengths are useful), batch sizes (e.g., 1 vs. 4/8), `max_new_tokens`.
- **Process (conceptual):**
  - Group prompts into batches and generate per batch.
  - Use a high‑resolution timer; if on GPU, synchronize before/after the measurement.
  - Report **tokens/sec** and **latency** for each batch size; optionally normalize per sample.
- **Outputs:** Dict or table with rows = batch size and key metrics.
- **Docs:**
  - CUDA synchronize: https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html
  - Efficient prompting tips: https://huggingface.co/docs/transformers/en/perf_infer_gpu_one#batching

**Expected trend:**
- Batching boosts **throughput** (tokens/sec overall) but may not reduce **latency per sample**.

---

### D. `compare_ppplx`
**Goal:** Report **perplexity** for baseline vs. quantized model on a tiny corpus.
- **Inputs:** A small list of sentences; both models should support `labels=input_ids` for loss.
- **Process (conceptual):**
  - Evaluate average **token‑level cross‑entropy** across the corpus (no gradient).
  - Convert to PPX with `exp(average_loss)`.
- **Outputs:** A short dict: `{model_label: {"loss": ..., "ppx": ...}}`.
- **Docs:**
  - Tokenizer & labels for causal LMs: https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM
  - Cross‑entropy loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
  - Perplexity basics: https://huggingface.co/docs/transformers/en/perf_train_cpu#perplexity

**Expected behavior:**
- Small quality drift for INT8 dynamic quantization is acceptable; **large** PPX spikes → revisit tokenization or loss aggregation.

---

### E. `build_results_table`
**Goal:** Assemble your measurements into a compact, human‑readable table.
- **Inputs:** Results from the sections above (dicts/lists of metrics).
- **Process (conceptual):**
  - Normalize keys/units (seconds, tokens/sec, ms/token).
  - Produce a single **summary table** (e.g., rows by scenario: baseline, quantized, cache on/off, batch sizes).
- **Outputs:** A small table (DataFrame or list of dicts) that’s easy to screenshot or export.
- **Docs:**
  - Pandas quick start (if used): https://pandas.pydata.org/docs/user_guide/10min.html

**Tip:** Include the **model id**, **device**, and **dtype** (or quantization backend) in the table caption or a header row.

---

### F. `probe_edge_prompts`
**Goal:** Stress‑test with “edge” prompt shapes to see where performance degrades.
- **Inputs:** A few atypical prompts: very short, very long, multi‑line, repetitive tokens.
- **Process (conceptual):**
  - Reuse your timing function with the same `max_new_tokens` to keep comparisons fair.
  - Note unusually high or low throughput cases and hypothesize why (e.g., very long **prefill**).
- **Outputs:** A short observation block with metrics per edge case.
- **Docs:**
  - Prefill vs. decode costs: https://huggingface.co/docs/transformers/en/perf_infer_gpu_one#gpu-inference

---

### G. `run_quantization_exercise` (controller)
**Goal:** A clean **orchestrator** that runs A→F in order and prints/saves a succinct report.
- **Inputs:** Config knobs (model id, batch sizes, prompt lengths, `max_new_tokens`, device).
- **Process (conceptual):**
  - Run baseline → PTQ → ablations → batching → PPX compare → table.
  - Print a 3–5 bullet **takeaway** at the end.
- **Outputs:** Final results table + summary text; optional serialized artifacts (CSV/PNG).
- **Docs:**
  - General performance tips: https://huggingface.co/docs/transformers/en/perf_infer_gpu_one

---

## 2) What “good” output looks like

- A one‑liner showing **model id**, **device**, **dtype/quant‑backend** actually used.
- A **baseline metrics block** (latency, tokens/sec, ms/token).
- An **ablation** summary for `use_cache` and padding differences.
- A **batching** table (batch size vs. latency/tokens/sec).
- A **PPX** comparison (baseline vs. quantized) with a short comment on quality drift.
- A single **results table** that aggregates the scenarios above.
- A short **takeaway** (3–5 bullets) stating the biggest wins and when they apply.

---

## 3) Sanity checks & pitfalls

- **Tokenizer/model mismatch:** Must load both from the **same** repo ID.
- **Timing without sync (GPU):** Always synchronize right before/after your timed block.
- **Apples‑to‑apples:** Keep `max_new_tokens` and prompts **constant** when comparing scenarios.
- **Perplexity math:** Average **over tokens**, not just per‑batch means; then `PPX = exp(avg_loss)`.
- **Quantization expectations:** Biggest wins often on **CPU**; on GPU, prefer other accelerators (tensor cores, CUDA graphs, paged attention, etc.).
- **Batching:** Throughput ↑; per‑sample latency may not improve.

---

## 4) References (bookmark these)

- **Transformers Auto classes:** https://huggingface.co/docs/transformers/en/model_doc/auto  
- **Text generation:** https://huggingface.co/docs/transformers/en/main_classes/text_generation  
- **Performance (GPU inference):** https://huggingface.co/docs/transformers/en/perf_infer_gpu_one  
- **Optimum (overview):** https://huggingface.co/docs/optimum/index  
- **Optimum Intel PTQ:** https://huggingface.co/docs/optimum/intel/optimization_guide/ptq  
- **Optimum ONNX Runtime quantization:** https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization  
- **PyTorch AMP/autocast:** https://pytorch.org/docs/stable/amp.html#autocast-mode  
- **CUDA synchronize:** https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html  
- **Perplexity:** https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

---

### Final checklist

- [ ] Baseline timings captured and reported clearly.  
- [ ] Quantized model constructed and runnable; benefits noted.  
- [ ] Ablations for cache/padding measured and explained.  
- [ ] Batching vs single‑sample compared fairly.  
- [ ] Perplexity compared (baseline vs. quantized), with sensible interpretation.  
- [ ] Results table + takeaways present.

If you can tick all boxes above, your starter is implemented to spec and ready for review.
