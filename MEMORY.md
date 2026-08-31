# MOBILE TRACKER — MASTER MEMORY (2026-08-30)

> How to use: new session → open ~/Projects/mobile_tracker/MEMORY.md and say "continue".

## 1. THE PROJECT
Privacy-first dashcam vehicle recognition (cars/make+model, plates, signs, road context).
Live at ~/Projects/mobile_tracker (Python 3.14, .venv). Dashboard localhost:8500.

## 2. MODEL STATE (what matters) — as of 2026-08-30
- **CURRENT KING (stays deployed): models/r50_dashcam.pt = v19_tiny224** (646 classes).
  - Honest holdout (v19, 2,605 crops): top-1 56.9% / top-5 77.7%.
  - Backups: r50_dashcam_v18_backup.pt, r50_dashcam_v17_backup.pt.
- **v20 (788 classes, 90,963 imgs) = STATISTICALLY TIED with king — NOT deployed.**
  - Intersection test (597 crops neither trained on): king 57.2/79.7 vs v20 56.5/79.4.
  - The "+20.5 win" and "54% loss" reports were BOTH artifacts of holdout contamination
    (each model had seen part of the other's holdout). No real change.
- **CRITICAL EVAL LESSON (2026-08-30): comparing models across rebuilds is INVALID
  unless tested on crops held out from BOTH. Fix for v21: freeze ONE permanent
  evaluation holdout; exclude it from ALL future training builds.**
- **THE WINNER FORMULA (proven v16→v20, keep using):**
  1. MORE unique real dashcam crops (videos → extract_crops → DeepSeek label).
  2. Oversample 5x (NOT 30x — 30x caused memorization collapse).
  3. Train from scratch (NO --init from king).
  4. Leak-free holdout (build_merged.py dedupes filenames globally).
  5. tiny (convnext_tiny) is enough — bigger archs don't beat it; 336px HURT.
  6. **+5k crops did NOT help (v20 tie) — at data plateau for current taxonomy.
     v21 needs the permanent-holdout methodology + much bigger variety to break past.**
- Dataset: dashcam_raw ~20.3k crops (19.9k DeepSeek + 337 Gemini-recovered rejects).
  1,824 rejects remain (Gemini quota resets daily → rerun gemini_recover_rejects.py).
- TTA enabled in classifier.py (flip + 240px scale avg) — +0.6 top-1 / +1.2 top-5.
- Checkpoints: models/v18/, models/v19/, models/v20/ (all fetched, pods terminated).

## 4. POD STATUS / RENTAL
- **v20 pod ACTIVE: tjz5fd4sdhwd8s (4090, port 10790)** — training tiny224×12 + ×16.
  Finisher: /tmp/v20_finisher.sh (fetches to models/v20/, terminates when done).
- Pod terminate via API: podTerminate(input:{podId}) with RUNPOD_API_KEY from .env.
- GPU picks: 4090 ~$0.39/hr (best); 3090 ~$0.20/hr (half price, same 24GB, 1.5x slower);
  5090 NOT worth it (3-4x price); 3070 NO (8GB too small for base/large).
- Pod templates have torch+CUDA but NO cv2 — always `pip install opencv-python-headless numpy`.
- **UPLOAD LESSON (2026-08-30): interrupted scp = silent incomplete tar!** For big tars:
  split into 700MB parts locally (split -b 700m), upload parts, cat on pod, then VERIFY
  tar entry count matches local. Also verify ALL bundle files (incl. scripts) landed.
- RunPod MCP configured in opencode (~/.config/opencode/opencode.jsonc): runpod (API,
  OAuth on first use) + runpod-docs. Restart opencode to activate.

## 5. KEY TOOLS/SCRIPTS
- train_classifier.py: --arch resnet18/50, convnext_tiny/small/base/large; --size; --init; --batch
  NOTE: num_workers=4 on CUDA, 0 on MPS (single-thread decode = slow local training ~1.5h/ep).
- compare_ckpts.py: holdout eval, width-based arch detection (768/1024/1536)
- classifier.py: runtime loader (same width detection), MPS inference, TTA, road hooks
- label_crops.py: DeepSeek/Gemini labeling (--provider; --max-images must be raised for big
  batches! default 250 caps the run). DeepSeek = paid, no cap.
- gemini_recover_rejects.py: second opinion on DeepSeek's "Unknown" rejects (77-100% rate).
  Run after Gemini quota resets; shrinks label_rejects.txt.
- hdd_download.sh: yt-dlp batch download to 4TB HDD (1080p mp4, -N 8, resumable, dedup by ID).
- cross_check_labels.py / two_opinion.py: two-provider verification flow.
- build_merged.py: dataset builder; --oversample (30 default — USE 5); leak-fixed (dedupe).
- extract_crops.py: crop mining from videos (multi-crop harvester, far/mid/near buckets)
- road_context.py: traffic lights (EU 3-lamp band), speed signs (GTSRB), road module
- frontend/index.html: near-fullscreen driving UI, on-frame light boxes + MAX overlay
- PROJECT_NOTES.md / PROJECT_REPORT.txt: full history + lessons

## 6. LOCAL LLM (REMOVED 2026-08-27) — do not re-add
- Ollama + qwen3.8:27b removed (OOM'd on 24GB M5; weaker than DeepSeek V4 Flash).
- /Applications/Ollama.app, ~/.ollama, caches, opencode provider config all removed.

## 7. NEXT STEPS (priority)
1. **TRAINING PHASE CLOSED (2026-08-31):** king v19 stays deployed (57.2%/79.7% honest
   intersection). v20/v21/v22 all verified BELOW king (56.5/54.7/53.7 top-1) — more
   YouTube-derived data + recovered rejects do NOT help (plateau confirmed). No more
   pod rounds on this data.
2. **LIVE-DRIVE (the real next step):** one drive validates the full system + collects
   the user's OWN footage — the only data source with zero rights issues and a real
   distribution shift. blur_plates: true already set. Follow LIVE_DRIVE_CHECKLIST.md.
3. **PUBLISH:** GitHub repo is ready (README/LICENSE/.gitignore/secrets-audit clean,
   honest metrics in RELEASE_PROMOTION.md). Push code+model anytime; HF dataset only
   after real blurred footage exists (or with provenance note for research use).
4. Known: 224px sweet spot; tiny enough; 5x oversample + from-scratch + leak-free +
   permanent frozen eval_holdout = the recipe. eval_all.py = honest intersection test.
5. Storage rule: big data on 4TB HDD only, never on the Mac.
6. Class strategy: keep fine-grained (user decision).
