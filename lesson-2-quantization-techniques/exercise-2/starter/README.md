# Exercise 2 (Revised Starter) — Student Walkthrough (No‑Code, Focused)

This guide is tailored to **Exercise_2_starter_revised.ipynb**. It highlights the **important parts to implement**, what good outputs look like, and the exact docs to reference. No code is included—only nudges to help you complete the starter confidently.

---

## 1) What you’re building

You will:
- Load a causal language model and matching tokenizer **reliably**.
- Evaluate **token‑level loss** on a small corpus using **batches** and **no gradients**.
- Convert the global average loss to **perplexity** and report clear summary metrics.
- *(Optional if present in your starter)* Measure **generation latency / tokens‑per‑second** and run small ablations.

**Detected key routines to complete (from the notebook):**
- **Model/tokenizer loading** — device-aware, dtype-aware, with fallbacks.
- **Batched evaluation** — DataLoader + no-grad loop to accumulate token-level loss.
- **Perplexity computation** — exp(global average loss); treat token weighting carefully.
- **(Optional) Generation timing** — baseline latency and tokens/sec on a fixed prompt.
- **Results summary** — small table/printout with clear labels.

---

## 2) Implementation nudges (no code)

### A) Model & tokenizer loading
- Detect the **device**: CUDA if available, else CPU; set `eval()` for inference.
- Load **tokenizer and model from the same repo ID**.
- If the notebook specifies dtype or precision, **resolve it** sensibly (e.g., float16 → fallback to float32 on CPU).
- Return a short confirmation string (model id, device, dtype label) for later logging.

**Docs:**
- Auto classes (tokenizer/model): https://huggingface.co/docs/transformers/en/model_doc/auto

---

### B) Batched data preparation
- Use a **DataLoader** with a **collate function** that tokenizes a batch of texts with **padding** (and optional truncation).
- Ensure `input_ids` (and `attention_mask` if used) are moved to the **same device** as the model.
- Keep batch sizes modest to avoid OOM.

**Docs:**
- Tokenizer padding/truncation: https://huggingface.co/docs/transformers/en/pad_truncation  
- DataLoader basics: https://pytorch.org/docs/stable/data.html

---

### C) No‑grad evaluation & global loss
- Wrap your loop in **`torch.inference_mode()`** (or `no_grad`) and make a single forward pass per batch.
- For causal LMs, passing `labels=input_ids` usually yields the **token‑level cross‑entropy** directly.
- Accumulate **(loss × token_count)** and **token_count** across batches → compute the **global average loss** at the end.

**Docs:**
- Causal LM labels/loss: https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM  
- Inference mode: https://pytorch.org/docs/stable/generated/torch.inference_mode.html

---

### D) Perplexity
- Compute **perplexity = exp(global_average_loss)**.
- Guard against numerical blow‑ups (very large loss → `inf` perplexity is acceptable to report).
- Log both **average loss** and **perplexity** with a couple of decimals.

**Docs:**
- Cross‑entropy/perplexity basics: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

---

### E) (Optional) Generation timing & ablations
- If the starter includes a timing function, use a **high‑resolution wall clock** and, on GPU, **synchronize** just before/after the timed block.
- Keep `max_new_tokens` **fixed** when comparing different settings (e.g., prompt length, precision).
- Report at least: total latency (s), tokens generated, tokens/sec, and ms/token.

**Docs:**
- Text generation API: https://huggingface.co/docs/transformers/en/main_classes/text_generation  
- CUDA synchronize: https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html

---

### F) Results summary
- Assemble a **compact table** (or clear prints) for: average loss, perplexity, and (if measured) latency/tokens‑per‑second.
- Include the **model id**, **device**, **dtype/precision label**, and **batch size** somewhere near the table.

**Docs:**
- Pandas quickstart (optional): https://pandas.pydata.org/docs/user_guide/10min.html

---

## 3) What “good” output looks like

- A one‑liner: **Loaded** `<model_id>`, **device** `<cpu/gpu>`, **dtype** `<float32/float16_fallback>`.
- A per‑batch log or concise table showing **loss** and **token counts** (optional but useful).
- A final **global average loss** and **perplexity** (PPX = exp(loss)).
- *(If timing present)* A small block with **latency**, **tokens/sec**, **ms/token** for a reference prompt; optional plots or comparisons.

**Sanity checks:**
- Tokenizer and model **match** (same repo id).
- All tensors are on the **same device** as the model.
- You **average over tokens**, not just over batches.
- If you tried `float16` on CPU, your code **falls back to float32**.

---

## 4) Troubleshooting

- **Shape mismatch / target errors:** Confirm you’re using `labels=input_ids` (causal LMs shift internally).  
- **OOM:** Reduce batch size or sequence length; switch to CPU for correctness.  
- **`inf` perplexity:** Check that your **global loss** is a *mean over tokens* before exponentiating.  
- **Inconsistent timings (GPU):** Synchronize around timed code; keep settings identical across comparisons.

---

## 5) References (bookmark these)

- Transformers Auto classes: https://huggingface.co/docs/transformers/en/model_doc/auto  
- Tokenizer padding/truncation: https://huggingface.co/docs/transformers/en/pad_truncation  
- Text generation API: https://huggingface.co/docs/transformers/en/main_classes/text_generation  
- Performance tips (GPU inference): https://huggingface.co/docs/transformers/en/perf_infer_gpu_one  
- PyTorch inference_mode: https://pytorch.org/docs/stable/generated/torch.inference_mode.html  
- CUDA synchronize: https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html  
- Cross‑entropy & perplexity: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

---

### Final checklist

- [ ] Device/dtype resolved; model/tokenizer loaded from the **same** repo; `eval()` set.  
- [ ] Batches tokenized with padding; tensors moved to the correct **device**.  
- [ ] No‑grad evaluation; **token‑weighted** global average loss computed.  
- [ ] **Perplexity = exp(global_average_loss)** logged clearly.  
- [ ] *(If timing)* Latency/tokens‑per‑second measured fairly with identical settings.  
- [ ] A compact **results summary** with model id, device, dtype, and batch size.

If you can tick each box above, your starter implementation is on target.
