# Exercise 2 — Solution Walkthrough

This walkthrough highlights only the **important parts** of the reference solution for *Exercise 2* and **skims the rest**. Use it to understand *why* each step is implemented, what you should see, and how to validate your own work.

---

## 1) What the solution accomplishes

- Reliably **loads a causal language model** and matching tokenizer.
- Runs a **batched, no‑grad evaluation** over a small text corpus.
- Computes a **token‑level average loss** and converts it to **perplexity**.
- Prints a concise **summary** (and may include simple per‑batch logs or a table).

If your notebook does these four things cleanly, you’re aligned with the solution.

---

## 2) Key implementation choices (the “why” behind the code)

### a) Robust model & tokenizer loading
- Detects **device** (CUDA if available, else CPU) and sets `eval()` for inference.
- Loads tokenizer and model from the **same repo ID** to avoid tokenization mismatches.
- Provides a clear confirmation line like:
> Loaded: <model_id> | device=<cpu/gpu> | dtype=<label>

**Why this matters:** Mismatched tokenizer/model pairs and silent CPU half‑precision are two of the most common pitfalls. The solution avoids both.

### b) Batched evaluation with a DataLoader
- Uses a minimal **collate function** that tokenizes with padding/truncation.
- Ensures every batch is moved to the **same device** as the model.
- Disables gradients globally (e.g., **inference mode** / **no_grad**).

**Why this matters:** Batch processing is both faster and less error‑prone. No gradients prevents unnecessary memory use and speed loss.

### c) Correct loss aggregation → global average
- For causal LMs, passing `labels=input_ids` yields the token‑level **cross‑entropy** automatically (the solution uses this path).
- Accumulates **loss × token_count** and **token_count** across batches to compute a **global average loss**.
- Converts to **perplexity** as `exp(average_loss)` in a small helper function (`compute_perplexity`).

**Why this matters:** Averaging only per‑batch means ignores batch sizes; the correct metric is **token‑weighted**.

### d) Timing & synchronization (if included)
- Uses a high‑resolution clock for any latency numbers.
- On GPU, performs **CUDA synchronize** around timed segments to avoid under‑counting.

---

## 3) What you should see (outputs)

Your exact numbers depend on your hardware and dataset, but the **shape** of results should match:

- A one‑line **load confirmation** with model id, device, and dtype label.
- Per‑batch short logs *(optional in the reference)* such as “batch i/N — loss …, tokens …”.
- A final **summary** reporting:
  - **Average loss** across all tokens
  - **Perplexity** = exp(average loss)

**Representative snippets from the solution run:**
- Loaded: <model_id> | device=<cpu/gpu> | dtype=<label>
- Perplexity: 3.425

If your run shows wildly large loss or `inf` perplexity, check tokenization, padding/truncation, and that you’re averaging **over tokens**, not just examples.

---

## 4) What’s intentionally skimmed

- Fancy logging/progress bars, argument parsing, or dataset plumbing → helpful, but not essential to the learning goal.
- Cosmetic plotting or tables → nice to have; focus on **correct loss aggregation** and **perplexity** first.
- Precision control (fp16/bf16) → optional here; correctness comes before speed.

---

## 5) Self‑check — does your implementation match the solution’s intent?

- **Loading**: same‑repo tokenizer/model; deterministic device/dtype; `eval()` set.
- **Batches**: collation pads/truncates appropriately; tensors moved to the correct device.
- **No gradients**: evaluation wrapped in `inference_mode` / `no_grad`.
- **Loss aggregation**: token‑weighted global average; not just “mean of batch means”.
- **Perplexity**: computed as `exp(global_average_loss)` and reported alongside the loss.

If you answered “yes” to all, you’ve captured the essence of the reference solution.

---

## 6) Troubleshooting aligned with the solution

- **Shape mismatch**: Ensure the model receives `labels=input_ids` (causal LMs handle shifting internally).  
- **Exploding perplexity**: Confirm you’re averaging over **tokens** and not mixing CPU/GPU tensors inadvertently.  
- **OOM / slow**: Reduce batch size or sequence length; try CPU for correctness.  
- **Tokenizer mismatch**: Always load both from the **same** model id.

---

### Takeaway

The solution prioritizes **correctness and reproducibility**: deterministic loading, batched no‑grad evaluation, token‑weighted loss, and a clean perplexity readout. Reproduce these behaviors and your Exercise 2 will be solid—even if absolute numbers differ on your machine.
