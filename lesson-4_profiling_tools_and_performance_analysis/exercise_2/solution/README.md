# Exercise 2 — Solution (Revised) Walkthrough

This document focuses on the **important parts** of `Exercise_2_solution_revised.ipynb` and skims the rest. Use it to understand what the solution is doing, how to check your own work against it, and how to interpret the outputs.

---

## 1) What the solution accomplishes

- **Loads a causal language model + tokenizer** reliably (same repository ID), sets the model to `eval`, and targets the appropriate **device** and **precision**.
- Performs **no‑grad, batched evaluation** on a small corpus to compute a **token‑level average loss**.
- Converts the global average loss to **perplexity (PPX)** and reports it clearly.
- *(If timing cells are present)* Measures **generation latency** (and optionally throughput) for a reference prompt.

**Detected functions present:** load_model_and_tokenizer

If your notebook cleanly does these items, you’re aligned with the reference solution.

---

## 2) Key implementation choices (the “why”)

### a) Robust model/tokenizer loading
- The tokenizer and model are loaded from the **same model ID** to avoid tokenization mismatches.
- The model is placed on the **available device** (CUDA if present, else CPU) and switched to **evaluation** mode.
- Any dtype/precision hints are respected with sensible fallbacks (e.g., CPU forces float32).

**Why it matters:** Mismatched tokenizer/model pairs or incorrect device/precision are the most common sources of incorrect loss/PPX.

### b) Batched, no‑grad evaluation for loss
- Batching improves efficiency and keeps your evaluation **deterministic** and **memory‑friendly**.
- The solution disables gradients (`inference_mode` / `no_grad`) because we’re evaluating, not training.
- For causal LMs, passing `labels=input_ids` typically yields a **token‑level cross‑entropy** internally (correct label shifting).

**Why it matters:** The **global token‑weighted average loss** is the right quantity to exponentiate into PPX. Averaging “mean-of-means” per batch can bias results.

### c) Perplexity as a quality sanity check
- **Perplexity = exp(global_average_loss)** computed over the **entire** evaluation set.
- The value is reported alongside loss and (optionally) sample counts, so you can compare checkpoints or settings fairly.

**Why it matters:** PPX offers a simple, reproducible quality signal; large unexpected jumps usually mean a data/labeling issue.

### d) (If present) Generation timing
- A reference prompt is used to time a single `generate` call.
- On GPU, synchronization is used around the timed section to avoid undercounting.
- Reports include **latency (seconds)** and may include **tokens/sec** / **ms per token**.

**Why it matters:** Separating **evaluation** (loss/PPX) from **inference timing** (latency/throughput) helps you reason about **quality vs. speed** independently.

---

## 3) What you should see (outputs)

Your exact numbers depend on hardware and the model used. Look for the **shape** and **clarity** of outputs rather than exact values:

- A short **load confirmation** (model id, device, dtype label).
- A clean **evaluation summary** with **average loss** and **perplexity**.
- *(If timing included)* A concise **latency** (and possibly **tokens/sec**) report for a reference prompt.

**Representative lines parsed from a solution run (for orientation):**
- Final perplexity: 55.14
- Latency     : 1.180 s


If your run shows extremely large loss or `inf` PPX, check that you’re averaging **over tokens** (not just over examples) and that tokenizer/model IDs match.

---

## 4) Interpreting the results

- **PPX meaning:** Lower perplexity indicates better next‑token modeling on the chosen corpus—compare like with like (same dataset/preprocessing).
- **Latency vs. PPX:** It’s normal for smaller/faster models to trade off some PPX for speed; the goal here is to **measure cleanly** and **report clearly**.
- **Variance:** On tiny toy corpora, PPX can be noisy—focus on plumbing correctness and reporting clarity first.

---

## 5) Troubleshooting aligned with the solution

- **Tokenizer/model mismatch:** Load both from the **same** repo ID.
- **Shape/label errors:** For causal LMs, pass `labels=input_ids` to let the model handle shifting internally.
- **GPU timing undercount:** Synchronize immediately **before** and **after** the timed block.
- **OOM / slow runs:** Reduce batch size or sequence length; CPU is acceptable for correctness checks.
- **`nan`/`inf` PPX:** Ensure you compute a **token‑weighted global average loss** before exponentiating.

---

## 6) Minimal checklist for parity

- [ ] Model & tokenizer loaded from the **same** repo, on intended device, `eval()` set.  
- [ ] Batched evaluation with **no gradients**.  
- [ ] **Token‑weighted** global average loss computed across the dataset.  
- [ ] **Perplexity = exp(global_average_loss)** reported clearly.  
- [ ] *(If included)* A small, well‑labeled timing block for generation.

If you can tick all the boxes above and your outputs resemble the structure described here, your implementation matches the solution’s intent—even if the exact numbers differ on your machine.
