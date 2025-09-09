# Exercise 2 — Student Implementation Guide (No‑Code Hints)

**Objective:** Implement the missing pieces to evaluate **language‑model quality and efficiency** on a fixed dataset by:
- Loading a model/tokenizer reliably on CPU or GPU
- Running **batched** evaluation (no gradient) over a small corpus
- Computing **loss** and deriving **perplexity**
- Reporting summary metrics (and optionally simple plots)

This guide gives you **nudges only** (no code) plus the right docs to finish each step.

---

## 0) Setup & Assumptions

- Libraries: **PyTorch**, **Transformers**, and optionally **Matplotlib / Pandas** for summaries.
- Device: Prefer **GPU** if available; CPU works fine but will be slower.
- Inference only: Make sure to disable gradients and keep the model in **eval** mode.

**Docs you’ll use:**
- Transformers Auto classes: https://huggingface.co/docs/transformers/en/model_doc/auto
- Causal LM docs (generate, logits, labels): https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM
- Tokenizer summary: https://huggingface.co/docs/transformers/en/tokenizer_summary
- `torch.inference_mode()`: https://pytorch.org/docs/stable/generated/torch.inference_mode.html
- `torch.no_grad()`: https://pytorch.org/docs/stable/generated/torch.no_grad.html
- `torch.utils.data.DataLoader`: https://pytorch.org/docs/stable/data.html
- Cross‑entropy loss & perplexity basics: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

---

## 1) Robust loading — `load_model_and_tokenizer(...)`

**Intent:** Make model loading **predictable** and **device‑aware**, with a small fallback list of model IDs so the notebook still runs if the preferred choice is unavailable.

**Nudges:**
- Detect the device once at the start (CUDA if available, else CPU).
- Keep a short ordered list of **candidate model IDs**. Try each until one loads successfully.
- Always load **tokenizer and model from the same ID**. Put the model in **eval** mode.
- Move the model to the chosen device. If the rest of the exercise mentions precision control, respect that when moving.
- Return: tokenizer, model, the actual model ID used, and a small string label for the dtype/device combination.

**What to verify:**
- If no candidate loads, raise a **clear error** listing tried IDs.
- Print a short confirmation like: `Loaded: <model_id> on <device> (dtype=<...>)`.

**Docs:**
- https://huggingface.co/docs/transformers/en/model_doc/auto

---

## 2) Batch preparation & collation

**Intent:** Evaluate multiple samples efficiently.

**Nudges:**
- Build a tiny dataset (or use the provided text list) and a **DataLoader** with a simple **collate_fn** that tokenizes to `input_ids` and `attention_mask`.
- Keep batch size modest; ensure each batch is moved to the **same device** as the model.
- Use `padding="longest"` (or a fixed `max_length` with truncation) so all sequences align in a batch.

**Docs:**
- Tokenizer usage and padding: https://huggingface.co/docs/transformers/en/pad_truncation
- DataLoader basics: https://pytorch.org/docs/stable/data.html

---

## 3) Forward pass for loss (no gradient)

**Intent:** Compute a **language modeling loss** (cross‑entropy) over the dataset, without updating weights.

**Nudges:**
- Use `torch.inference_mode()` or `torch.no_grad()` to disable gradient tracking.
- For causal LMs, many models compute loss directly if you pass `labels=input_ids` (shifted internally). Otherwise, use the logits and shift yourself.
- Aggregate the **sum of loss × tokens** and the **sum of tokens** across batches so you can compute a **global average loss** later.

**Docs:**
- Causal LM loss / labels: https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM

---

## 4) Perplexity — `compute_perplexity(loss: float)`

**Intent:** Convert an **average negative log‑likelihood** to **perplexity**.

**Nudges:**
- Perplexity = **exp(average loss)**. Make sure your loss is the **mean** over tokens (global).
- Guard against numerical overflow if your loss is very large; clipping or returning `inf` is acceptable.
- Report both the average loss and perplexity with a few decimals.

**Docs:**
- Cross‑entropy & NLL: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

---

## 5) Putting it together — evaluation loop

**Intent:** Run a clean loop that returns both **per‑batch** and **overall** metrics.

**Nudges:**
- For each batch:
  - Move tensors to device.
  - Compute loss (and optionally tokens/sec if timing is included).
  - Accumulate totals.
- After the loop:
  - Compute **global average loss** and **perplexity**.
  - Optionally build a small **DataFrame** with per‑batch stats (loss, tokens, length) and a summary row.

**Optional timing tips:**
- Use a high‑resolution timer (e.g., `time.perf_counter()`).
- On GPU, synchronize before/after timing if you measure latency precisely.

**Docs:**
- CUDA synchronize: https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html

---

## 6) Expected outputs & sanity checks

**What a correct run looks like (qualitatively):**
- A short **load confirmation** (model id, device, dtype label).
- A neat printout or table with per‑batch **loss** (and token counts), followed by:
  - **Global average loss** across the dataset
  - **Perplexity = exp(global_loss)**
- On **tiny toy corpora**, absolute values may be high/unstable. That’s fine; focus on the **plumbing** being correct.

**Checklist:**
- [ ] Tokenizer & model loaded from the same repository
- [ ] Model in eval mode; inference without gradients
- [ ] Batches on the same device as the model
- [ ] Loss aggregated over **tokens**, not over examples only
- [ ] Perplexity computed from **global average loss**
- [ ] Clear, labeled outputs (no silent assumptions)

---

## 7) Nice‑to‑have extensions (optional)

- Add a simple **bar chart** of per‑batch loss or sequence length vs loss.
- Try a different model checkpoint and compare perplexities.
- Add **precision control** (float16 on GPU) to check speed vs stability for evaluation (loss should remain comparable).

---

## 8) Troubleshooting

- **Shape mismatch / target size errors:** Ensure labels align with logits’ time dimension (shift handled internally by many causal LMs when `labels=input_ids`).  
- **Tokenizer mismatch:** Load both tokenizer and model from the **same** model id.  
- **OOM on GPU:** Reduce batch size or max sequence length; try CPU for correctness.  
- **Very large/`inf` perplexity:** Confirm you’re using **mean loss per token** before exponentiating.

---

### You’re ready to implement
Follow the nudges above and consult the linked docs. Your final notebook should reliably load a model, iterate over a small dataset in batches without gradients, compute a stable global loss, and convert it to **perplexity** with a clear, reproducible report.
