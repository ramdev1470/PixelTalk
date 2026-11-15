# PixelTalk v3 - Conversational Image Chatbot

Upload a photo (or pick a sample) and ask about it in plain language. Follow-ups work ("what colour is *it*?"),
answers stream in word by word, and "can I eat this?" questions go through a safety layer that also catches
mold, rot, wild mushrooms and non-food.

| | Multimodal VLM (default) | Baseline (SIH1604, kept for comparison) |
|---|---|---|
| How it sees | A chat vision-language model gets the **real image on every question** | A captioner writes one sentence; a text-only model answers from it |
| Follow-ups | Native multi-turn memory | Rule-assisted |
| Weak spot | Slower on CPU; can still misread tiny or crowded details | Loses everything the caption missed |

Default model: `Qwen/Qwen2-VL-2B-Instruct` (Apache-2.0). If it cannot be loaded the app falls back to
`Salesforce/blip-vqa-base` (short answers only) and **says so in the sidebar, with the reason and the fix**.

---

## 1. Quick start (Windows)

Easiest: double-click **`setup_windows.bat`** (creates `.venv`, installs, downloads ~6 GB of models, runs a
real-model check). Afterwards start the app with **`run_pixeltalk.bat`**.

Or by hand, in PowerShell from this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py          # one time, needs internet
python smoke_test.py               # proves the real model works on your PC, prints timings
streamlit run app.py
```

Using your existing `.venv` is fine: run `pip install -r requirements.txt` in it. **`torchvision` is required**
(the Qwen2-VL image processor imports it).

NVIDIA GPU? Install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/ and answers drop from
tens of seconds to a few seconds. It is detected automatically.

## 2. What to expect (speed and memory)

Your earlier environment had a **CPU-only** PyTorch, so these are CPU expectations. They are estimates, not
measurements of your PC; `python smoke_test.py` prints real timings including "first words after ...s".

* A 2B-parameter model on a laptop CPU: roughly **10-40 seconds per answer**. Answers **stream**, so text appears
  as soon as the model starts writing instead of after a blank wait.
* Memory: about 4.5 GB for the model in bfloat16 (chosen automatically below 24 GB RAM; float32 needs about 9 GB).
* **Image detail** slider (sidebar): *Fast* is quickest, *Detailed* reads small text better.
* The safety check costs one extra model call the first time per image, then it is cached.

## 3. The safety layer

For "is it safe to eat / drink / edible / poisonous" questions the image is classified first, and the policy is
deterministic (the language model never gets to say "yes, it's safe"):

| Category | Result |
|---|---|
| Not food (laptop, galaxy, animal...) | Refused: "this is not food" |
| **Spoiled** (visible mold or rot) | Refused: do not eat, throw it away (mold spreads through bread) |
| Wild mushroom / berry / foraged plant | Refused with a foraging warning |
| Ordinary food | Described **plus an advisory**; a photo can't show bacteria, cooking or allergens |
| Can't tell | Cautious answer, asks for a clearer photo |

Two extra layers catch a model that labels moldy food as "fine": the subject name it gives, and the food
description it writes, are both scanned for mold/rot (with negation: "no visible mold" is not flagged).
Matching is whole-word, so "carrots" is never "car" and "wooden" is never "wood".

## 4. Tests

```powershell
python -m pytest tests -q            # 79 tests, about 10 seconds, no downloads
python smoke_test.py                 # the REAL model: answers, timings, safety checks
python -m multimodal.evaluate_comparison   # accuracy + latency of both pipelines on 43 labelled questions
```

What the automated tests cover:

* **Logic (fake model):** safety rules, history window, streaming (pieces, early close, mid-stream failure, empty
  output, fallback), error containment, image preprocessing (EXIF rotation, transparency, palette, tiny/huge).
* **Real engine code on tiny random-weight models** (`tests/tiny_qwen.py`, `tests/tiny_blip.py`): the actual
  `transformers`/`torch` code paths: loading, chat template, image processor, `generate()`, streaming, cancelling,
  the BLIP fallback and the whole fallback chain. Answers are gibberish by design; what is checked is that it runs.
* **UI (`tests/test_app_ui.py`):** the real Streamlit app driven end to end, including loading **every** sample image.
* **Baseline:** your original model and dialogue tests.

`evaluate_comparison` writes `results/eval_results.json`, which the app then shows. The labels were checked against
the images by hand; the sample images are simple, so treat the score as a regression check, not a promise.

## 5. Configuration (environment variables)

| Variable | Meaning |
|---|---|
| `PIXELTALK_VLM` | Another Hugging Face chat VLM id (Qwen2-VL / Qwen2.5-VL / Qwen3-VL family) or a local folder |
| `PIXELTALK_FALLBACK_VLM` | Another BLIP-VQA model id for the fallback engine |
| `PIXELTALK_DTYPE` | `float32`, `bfloat16` or `float16` |
| `PIXELTALK_MAX_SIDE` | Longest image side in pixels sent to the model (smaller = faster) |
| `PIXELTALK_OFFLINE` | `1` = never touch the network |

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar says "Running on the BLIP-VQA fallback" | The main model failed to load; the warning under it says why. Usually `pip install torchvision` or `python download_models.py` |
| First start looks frozen | It is loading or downloading the model; later starts are faster |
| Out of memory | Close other programs; `PIXELTALK_DTYPE=bfloat16`; use *Fast* detail |
| Very slow answers | CPU-only PyTorch: use *Fast* detail, or install CUDA PyTorch if you have an NVIDIA GPU |
| Wrong or vague answer | Ask something more specific, use *Detailed*, lower the creativity slider |

## 7. What is and is not verified

Verified (automated, in this package): everything in section 4, run against `transformers` 5.23 and `torch` 2.15
on Linux with Python 3.12.

**Not verified:** the real model weights were not run (the build environment could not reach Hugging Face), so
answer quality and CPU speed on your PC are unmeasured, and Windows itself was not exercised. `smoke_test.py`
closes that gap on your machine in a few minutes.

Known limits:

* No vision model is always right: tiny details, crowded scenes, counting and unusual objects can still be wrong.
  The system prompt makes the model say "I can't tell" instead of guessing, which reduces but does not remove errors.
* Spoilage detection depends on the model noticing the mold; the layers above make a miss less likely, not impossible.
* The baseline keeps its state in one shared object (single local user). The multimodal pipeline is per session.
* The custom seq2seq checkpoint was trained on synthetic data that contains a "safe to consume" answer
  (`data/data_generator.py`). It is not the default and the guardrail overrides food-safety questions; retrain it if you rely on it.
* No LoRA adapter is shipped; `multimodal/train_lora.py` still trains one for the BLIP fallback only.

## 8. How this compares with the earlier versions

| | Original | ARENA project | v2 | **v3 (this)** |
|---|---|---|---|---|
| Sees the image | Caption / 1-word VQA | Caption only | Pixels every turn | Pixels every turn |
| Answers invented by templates | Yes | Learned from hand-written data | No | No |
| Spoiled / wild / non-food safety | Substring bugs | None | Wild + non-food | **Plus spoilage, 3 layers** |
| Streaming answers | No | No | No | **Yes, with safe fallback** |
| Realistic sample photos | No | 5 | No | **5 (from ARENA) + originals** |
| Real-engine tests | No | No | No (fake model only) | **Yes (tiny real models)** |
| UI tests | No | No | Stubbed models | **Real engine, every sample** |
| Fake numbers in the UI | Yes | No | No | No |
| Measured accuracy | Fake | Weak (ROUGE-L 0.24) | None | Script on 43 questions |

## Layout

```
app.py                  Streamlit app
multimodal/             Chat VLM engines, safety guardrail, evaluation, LoRA training (fallback)
models/                 Baseline: BLIP captioner + dialogue manager + seq2seq
sample_images/          13 samples (5 realistic photos + drawings + space)
tests/                  Unit, real-engine integration and UI tests
download_models.py      One-time model downloader      setup_windows.bat / run_pixeltalk.bat
smoke_test.py           Real-model check
```
