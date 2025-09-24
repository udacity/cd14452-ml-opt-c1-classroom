# Purpose of this Folder

This folder should contain the starter code and instructions for the exercise.

## Exercise 1 — Student Implementation Guide (No‑Code Hints)

**Objective:** Fill in the starter notebook so you can measure text‑generation latency and throughput:
- As a function of prompt length
- Across numeric precisions (e.g., float32 vs float16 where supported)

This document gives you **nudges only** (no code) and the **right docs** to complete each step confidently.

---

## 0) Before you start

- Ensure you have PyTorch, Hugging Face Transformers, and Matplotlib installed.
- Prefer a GPU machine for the precision comparison; CPU is fine for the prompt‑length experiment.
- Plan to do one warm‑up generate call before timing your real measurement.

**Useful documentation (skim now, return as needed):**
- PyTorch automatic mixed precision (autocast): https://pytorch.org/docs/stable/amp.html#autocast-mode
- PyTorch inference_mode: https://pytorch.org/docs/stable/generated/torch.inference_mode.html
- PyTorch CUDA synchronize: https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html
- HF Auto classes (tokenizer + causal LM): https://huggingface.co/docs/transformers/en/model_doc/auto
- HF Text generation (generate, GenerationConfig): https://huggingface.co/docs/transformers/en/main_classes/text_generation
- HF Tokenizer summary: https://huggingface.co/docs/transformers/en/tokenizer_summary
- Matplotlib quick start: https://matplotlib.org/stable/users/explain/quick_start.html

---

## 1) Resolve requested numeric precision

**Your task:** Convert a user‑provided precision string into the actual dtype and a human‑readable label.

**Nudges:**
- Accept only the intended values (float32 and float16). Normalize capitalization.
- If running on CPU and the user asks for float16, fall back to float32 and make the label say so clearly.
- Return both the dtype and a short label string (you’ll display this later).

**What to check:**
- Invalid inputs should raise a helpful error message that tells the student what values are accepted.

---

## 2) Load the model and tokenizer

**Your task:** Load a causal language model and its matching tokenizer; respect the chosen dtype and device.

**Nudges:**
- Detect device (GPU if available, else CPU).
- Try a short list of candidate model IDs in order. First one that loads successfully “wins”.
- Put the model in eval mode. Use the resolved dtype and chosen device.
- Return tokenizer, model, the exact model ID used, and your dtype label string.

**What to check:**
- If no candidate loads, raise an error that lists which models you tried.
- Make sure tokenizer and model are from the same repository name.

**Docs:** AutoModelForCausalLM / AutoTokenizer: https://huggingface.co/docs/transformers/en/model_doc/auto

---

## 3) Measure one generation run

**Your task:** Time a single text generate call and report performance metrics.

**Nudges:**
- Tokenize the prompt and move inputs to the same device as the model.
- Do a brief warm‑up generate (small number of tokens) that you do not time.
- Use a high‑resolution timer. If using GPU, synchronize right before starting and right after finishing.
- Compute and return at least:
  - total latency in seconds
  - number of new tokens generated
  - tokens per second
  - average latency per token (milliseconds)

**What to check:**
- Keep max_new_tokens fixed when comparing across settings.
- Confirm that inputs and model truly live on the same device.

**Docs:** Text generation: https://huggingface.co/docs/transformers/en/main_classes/text_generation

---

## 4) Prompt‑length sweep

**Your task:** See how performance changes as the input gets longer.

**Nudges:**
- Choose a handful of target token lengths (for example: small, medium, large).
- Build longer prompts by repeating a base text until the tokenized length meets your target.
- For each length:
  - Run your measurement function with the same max_new_tokens and use_cache setting.
  - Record latency, tokens/sec, and any other useful numbers.
- Make two simple plots:
  - Latency (seconds) vs. prompt length (tokens)
  - Tokens per second vs. prompt length (tokens)

**What to expect:**
- Total latency typically grows with longer inputs; tokens/sec can dip as context gets longer.

**Docs:** Tokenizer basics: https://huggingface.co/docs/transformers/en/tokenizer_summary  
Matplotlib quick start: https://matplotlib.org/stable/users/explain/quick_start.html

---

## 5) Precision sweep

**Your task:** Compare float32 and float16 (where supported).

**Nudges:**
- Build a small list of precisions to try. On CPU you’ll effectively only have float32.
- For each precision:
  - Resolve dtype and label (from Step 1).
  - Either move the model to that dtype once, or use an autocast context for the timed generate.
  - Measure with the same prompt, the same max_new_tokens, and the same cache setting.
  - Store results in a dictionary keyed by the precision label.
- Optionally add a bar chart or table to compare.

**What to expect:**
- On modern GPUs, float16 often reduces latency and increases throughput vs. float32.
- On CPU, a float16 request should have fallen back to float32 by design.

**Docs:** AMP autocast: https://pytorch.org/docs/stable/amp.html#autocast-mode

---

## 6) Sanity checks & troubleshooting

**Consistency:** Keep generation parameters identical when you compare runs.

**GPU timing:** Always synchronize around the timed block to avoid counting work that hasn’t finished yet.

**Caching:** Turning the generation cache on usually helps; feel free to try both values but be consistent inside each comparison.

**Reproducibility:** If you want repeatable text, set a random seed (not required for timing alone).

**OOM errors:** Use a smaller model or reduce max_new_tokens. GPU memory is limited.

**Tokenizer mismatch:** Model and tokenizer must come from the same model ID.

**First‑run effects:** Don’t compare the warm‑up against real runs; compilation/caching can distort timing.

---

## 7) Presenting your results

**Minimum deliverables:**
- A short paragraph describing your setup (device, model ID, precisions tested).
- A table and/or plots for the prompt‑length sweep.
- A table and/or plots for the precision sweep.
- Two‑to‑three bullet points explaining the trends you observed.

**Optional extras:**
- Try bf16 on hardware that supports it.
- Compare use_cache on vs off.
- Add error bars by running multiple trials per setting.

---

### You’re ready to implement
Use the nudges above alongside the documentation links. If you tick off each checklist item, your notebook will produce clear, comparable latency and throughput measurements across prompt lengths and precisions.
