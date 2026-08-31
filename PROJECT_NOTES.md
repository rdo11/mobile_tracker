# mobile_tracker — Project Notes

Live document: captures the whole coding journey, learnings, tools, and
decisions. Append as we go; used to write the final project report.

---

## 1. What we're building

A **dashcam make/model recognition pipeline** running live at ~20-25 FPS on a
base M2 MacBook Air (8GB, 7-core GPU):

- **Detection**: YOLOv8n (ultralytics), `detect_every_n_frames: 2`
- **Classification**: locally trained ResNet50 fine-tuned on dashcam crops
  (247 classes), fully local — no API calls at runtime (as of 2026-08-19)
- **Fallback**: Grok Vision (xAI, `grok-4.6`, reasoning_effort low) and batched
  Gemini (built, disabled) for low-confidence results / offline labeling
- **Coarse labels** (truck/bus/bike/motorcycle) come straight from YOLO class
  names — never sent to classifier or API
- **Plate logic** (todo): DK free-text, DE "E"=electric/hybrid, "H"=historic
- **Camera**: Iriun WiFi virtual webcam (iPhone), `source: 0`
- **Dashboard**: local web server on `localhost:8500`
- **Min-size gates**: `min_classify_area: 12000`, `min_box_side: 90` — distant
  tiny cars stay "Unknown" instead of confident hallucinations
- **Honesty over hallucination**: `min_display_conf: 0.45` — low-confidence
  predictions display "Unknown", never a random guess

## 2. Tech stack

- Python 3.14 (Mac, venv at `.venv`)
- torch 2.13.0 (MPS on Mac, CUDA on RunPod)
- torchvision 0.28.0
- ultralytics YOLOv8n for detection
- opencv-python (cv2) for image loading
- FastAPI / uvicorn (dashcam server), websockets 17.0.1
- Grok (xAI) + Gemini (Google) API clients for fallback classification
- RunPod GPU (RTX 4090 / RTX A4000 / RTX A4500 / RTX 2000 Ada) via SSH —
  training (CUDA) + eval
- SSL_CERT_FILE=certifi path needed for torch downloads on Mac (no CA certs in
  the system Python)

## 3. Dataset

- Scraped from **Wikimedia Commons** via `wikimedia_scraper.py`
- **249 classes** (after cleanup), ~10,570 images (~600MB), 0 corrupt
- All generations merged into model-level classes (`class_key()`):
  - Roman numerals (Golf VII -> Golf), chassis codes ((F30)), single letters
    (Corsa E -> Corsa), letter+digits (Passat B8), plain digits (Fortwo 451)
  - Fixed: Tesla "Model S/X/Y" kept (letter IS the model name, not a generation)
- Aliases fix `ALIASES`: Skoda/Citroen accent names, Mazda 2/3/6, MG4 EV,
  MG ZS EV, Range Rover, Renault Megane/Scenic, Macan EV -> Macan (XAB)
- `_fold()` normalizes accents (SK/CZ names vs ASCII folder names)
- COARSE_ONLY_MAKES: trucks/bikes (MAN, Scania, KTM, Ducati...) never trained
- Interiors removed (233 auto + user manually filtered the rest)
- Files have URL-encoded names with `?utm_source=...` — valid, decode fine
- Skoda/Citroen re-scraped at depth 3 (`category_files` recursion) to get
  generation photos (Octavia, Fabia, Superb, Kodiaq, Karoq, Kamiq)
- Filtered set (`storage/dataset/filtered`): 9,521 imgs (9.9% noise dropped by
  YOLOv8n car detection, `filter_crops.py --no-crop`)
- **Dashcam crop collection** (the big accuracy win):
  - `label_drive.py`: one offline Grok call per unique tracked car; picks the
    sharpest best-size crop; saves to `storage/dataset/dashcam_raw`
  - Requires `yolo.track(persist=True)` (plain detection has no track IDs),
    dense sampling `step = max(1, n // 1200)` (sparse sampling lost tracks),
    min 90px box gate
  - 13-min EU drive -> 32-35 labeled crops (~$0.35 in Grok tokens);
    a US drive showed the model can't learn classes it never saw (see §4)
  - Classes only kept if they exist in the 247-class checkpoint — unmatched
    labels (Kia Ceed SW, Kia Rio, Suzuki SX4, Hyundai Kona, Peugeot 406) are
    valid EU cars but have NO model class -> honest "Unknown" at runtime
- **Merged training set** (`storage/dataset/merged`): filtered (9,521, hardlinked)
  + dashcam crops injected x30 oversample (~3,120) into canonical class dirs;
  class names normalized to checkpoint names (Skoda Fabia, Citroen C3, VW Golf)
- **5 missing EU classes added** (2026-08-19): Kia Ceed (35), Kia Rio (53),
  Suzuki SX4 (40), Hyundai Kona (33), Peugeot 406 (53) scraped from Wikimedia
  (scraper MODELS dict updated) and injected x30 into merged → 252 classes,
  ~19k images. Checkpoint `r50_dashcam.pt` (247 classes) needs a head-extend
  retrain (`--init` loads backbone, new fc for 252).

## 4. Training saga (the big debugging story)

### v1: local MPS training FAILED silently
- ResNet18, 8 epochs, lr 1e-3, MPS: val_acc 6.1% — essentially random (ln 245 = 5.5 loss)
- ResNet18, 15 epochs, lr 3e-3: val 3.2%, train 3.9% — STILL not learning
- Higher lr learned SLOWER (inverted) — huge red flag

### Diagnosis process
- Ran two-class fine-tune (Golf vs Polo) on MPS: **it learned** (acc 0.9) → machinery OK
- Frozen-head linear probe on 247 classes: barely learned → suspected data or weights
- Zero-shot ImageNet test: pretrained weights ARE good (Golf -> "minivan", Tesla -> "sports car")
- Batch-32 same-image test: loss -> 0.0, works
- Conclusion: **MPS (Apple GPU) was silently corrupting gradients during full
  multi-class training** — the 2-class test used the same code path and worked,
  but full 247-class on MPS consistently failed to learn

### The fix: train on CUDA instead
- User started a RunPod pod (RTX 2000 Ada, 16GB VRAM)
- SSH: `ssh -i ~/.ssh/RunPod root@<ip> -p <port>` (the `id_ed25519` path
  doesn't exist on this Mac — real key is `~/.ssh/RunPod`)
- Added CUDA support to `train_classifier.py` (was MPS-only check!)
- Added `num_workers=4` for CUDA (data decode was starving the GPU)
- Uploaded dataset via tar-over-ssh (rsync from macOS fails: old rsync + port)

### CUDA training results (this is what works)
- ResNet18, batch 128, lr 1e-3, cosine, 60 epochs
- epoch 3: train 33.7% / val 12.6%
- epoch 9: train 94.5% / val 36.7%
- ~30s/epoch on RTX 2000 Ada; 60 epochs ≈ 30 min
- Val is noisy (small per-class val sets, noisy Wikimedia images) — expected

### v2: Grok-hybrid live mode (2026-08-19, before fine-tune)
- Local deep classifier DISABLED (model_path "") — Grok became the only
  make/model source: `fallback_conf: 1.0` = ALWAYS query Grok for cars
- `reasoning_effort: "low"` in grok payload: 20-30s/call -> ~5s/call with no
  quality loss; grok-4.6 (xAI's best vision model) kept over grok-3-mini
  (tested: grok-4.6 low = 4.9s, Toyota HiAce conf 0.85; grok-3-mini = 6.1s,
  Ford Transit conf 0.4)
- Worker crash found+fixed: `_save_sample` stored `result["label"]` (str) in
  `self._saved[track_id]` -> `TypeError: '>=' not supported between instances
  of 'str' and 'int'` at grok_classifier.py:193; fixed to counter increment

### v3: dashcam fine-tune #1 (the Land Rover lesson)
- `train_classifier.py` gained `--init <checkpoint>` + `--freeze` flags;
  checkpoint class ORDER verified identical to merged dirs (0 missing/unexpected
  at `--min-per-class 1`)
- RTX 4090 pod (EU-CZ-1, 24GB, $~0.16-0.30/hr), 8 epochs, batch 128, lr 3e-4
  from `r50_filt_60.pt` (57.6% baseline): val_acc 93.2% — BUT that number was
  on the merged set incl. the same 32 oversampled crops (memorization, not
  generalization)
- On a real US drive the model collapsed to "Land Rover Discovery Sport" on
  ~90% of cars: those crops had classes (Honda Fit, Ford Expedition, Lincoln
  Aviator, Chrysler Pacifica...) that DON'T EXIST in the 247 — the model
  literally cannot say "Honda Fit", so it fell back to its Land Rover prior.
  32 unique dashcam crops x30 oversample could NOT override the web-image prior
- Lesson: dashcam crops must match existing classes; fine-tuning needs way more
  unique real crops; val_acc on oversampled sets is meaningless

### v4: dashcam fine-tune #2 (current, deployed)
- Dataset rebuilt: ~100 unique EU crops (2 drives) normalized to canonical
  class names (Škoda Fabia -> Skoda Fabia, Citroën C3 -> Citroen C3,
  VW Golf 6 Variant -> Volkswagen Golf, Opel Crossland_X -> Opel Crossland,
  Porsche Macan (XAB) -> Porsche Macan)
- RTX A4000 pod (direct IP 157.157.221.29:20719), 8 epochs, batch 128, lr 3e-4,
  init = v3 checkpoint: val_acc 97.6%
- Sanity check on real crops (local, M2): Skoda Fabia 1.00, VW Golf 0.97,
  Peugeot 208 1.00, Ford Fiesta 0.99, Toyota Aygo 1.00 — no Land Rover bias
- Deployed: `models/r50_dashcam.pt`, `grok.enabled: false` (fully local),
  `fallback_conf: 0.45` (hybrid if re-enabled)

### v5: 252-class upgrade (5 missing EU classes added)
- Added Kia Ceed (35), Kia Rio (53), Suzuki SX4 (40), Hyundai Kona (33),
  Peugeot 406 (54) from Wikimedia (scraper MODELS dict updated), injected
  x30 into merged → 252 classes, ~19k images
- First run used default `--min-per-class 15` → silently DROPPED 5 old
  classes with <15 images (Dacia Jogger, Fiat Tipo, Land Rover Defender,
  Mercedes A-Class, Mercedes G-Class) — caught by comparing class sets
- `--init` needs the head stripped for class-count changes: `load_state_dict`
  crashes on fc size mismatch → patched train script to drop `fc.*` keys
  from the init checkpoint (fresh head, backbone preserved)
- 247→252 run (no dashcam crops for new classes): val 95.5% but only
  Peugeot 406 generalized to real crops — web-photo domain gap, the Land
  Rover lesson in reverse
- Injected the 2 real dashcam crops per new class (x30) → val 95.6%;
  ALL 5 new classes now 0.99+ on real crops; old classes unaffected
- Deployed: `models/r50_dashcam.pt` (252 classes, 92MB)
- Pod note: files uploaded via tar carry uid 501 → not writable by root;
  `chown root:root` after extract. `setsid bash -c '...' </dev/null` is
  the reliable background-launch pattern.

### RunPod lessons
- Pod: RTX 2000 Ada 16GB, 251GB RAM, 64 cores, 10GB disk, torch 2.4.1+cu124
- `pkill -f train_classifier.py` matched the SSH command line itself (self-kill)
- Launch via `setsid bash launch.sh </dev/null >/dev/null 2>&1 &` pattern
- Tar-over-ssh upload: `tar -cf - ... | ssh ... "tar -xf -"` (watch nested paths)
- **SSH gateway (ssh.runpod.io) kills non-PTY sessions**:
  "Your SSH client doesn't support PTY"; scp/sftp subsystem requests fail;
  expect-based interactive sessions work (`spawn ssh ...` + expect prompt)
- **Direct public IP + port pod = plain ssh/scp works** (the reliable pattern)
- RunPod GraphQL: `podFindAndDeployOnDemand` mutation, `Authorization: Bearer`
  header, must send `User-Agent: Mozilla/5.0` (Cloudflare 1010 otherwise),
  certifi SSL context required; `podTerminate` returns Void (no selection block)
- `dataCenterId` EU options: EU-CZ-1 (Prague), EU-RO-1, EU-NL-1, EU-FR-1,
  EU-SE-1; stock varies by minute — 4090/A4000/A4500/2000Ada all seen
- Image: `runpod/pytorch:1.1.0-rc.154-cu1290-torch291-ubuntu2404` (older
  `2.4.1-cuda12.4.1-cudnn9.2` no longer exists); needs
  `pip install opencv-python-headless numpy pillow tqdm`
- First pod failed: `volumeInGb` without `volumeMountPath` -> invalid mount
  config; use `containerDiskInGb` only, no volume
- API key 401s can mean the key was rotated (check `.env` vs user-provided)

## 5. Current state & todos

- [x] Dataset complete (249 classes, ~10.5k images)
- [x] ResNet18 baseline (CUDA, raw, 60ep): 46.9% — saved
- [x] YOLO-crop rejected (40.5%), YOLO filter-only accepted (10% noise dropped)
- [x] **FINAL MODEL v2: r50 fine-tuned on merged (filtered + dashcam crops)
      8 epochs — val 97.6%** → `models/r50_dashcam.pt`, wired into config ✓
- [x] Local filtered dataset rebuilt: `storage/dataset/filtered` (9,521 imgs)
- [x] `validate_dataset.py` — Gemini batch validation gate for future scrapes
- [x] HF dataset card drafted: `HF_DATASET_CARD.md`
- [x] Grok-only live mode (fallback_conf 1.0, reasoning_effort low, ~5s/call)
- [x] `label_drive.py` — offline one-call-per-car labeling from recordings
- [x] Grok worker crash fixed (`_saved` string-vs-int bug)
- [x] Min-size gates (`min_classify_area: 12000`, `min_box_side: 90`)
- [x] Fully local runtime: `grok.enabled: false`, local r50 does cars
- [x] **5 missing EU classes dataset ready** (2026-08-19): Kia Ceed, Kia Rio,
      Suzuki SX4, Hyundai Kona, Peugeot 406 scraped + injected x30 into merged
      (252 classes, ~19k images) — needs head-extend retrain on a pod
- [ ] Retrain with new head (247 → 252 classes): 8 epochs
      `--init models/r50_dashcam.pt --arch resnet50 --batch 128 --lr 3e-4`,
      bundle = merged + train_classifier.py; needs FRESH RunPod API key
      (old one 401'd — rotated) — user action: paste new key
- [ ] FPS test with USB iPhone (Iriun USB mode)
- [ ] DK/DE plate logic
- [ ] Hugging Face dataset upload + description (AFTER everything)
- [ ] Rotate xAI API keys (exposed in chat history!) — user action
- [ ] Gemini fallback: built, disabled — enable by user choice later
- [ ] **Gemini free tier** (Google AI Studio): 5 requests/day, 250k token
      limit — worth using for SMALL jobs (validate a few crops, label 1-2
      hard cars/day) at zero cost; NOT for bulk labeling. `gemini.enabled`
      toggles it. Add GEMINI_API_KEY in .env.
- [ ] RunPod remote inference support (future, for live tracking without lags)
- [ ] Grok voice project (save xAI credits for it)
- [ ] Re-ID memory: keep a car's tag across track re-entry (new track_id on
      re-entry loses the label today)
- [x] **Plate-based re-ID** (2026-08-19): plate text -> stored make/model/color;
      next sighting reuses attrs and SKIPS the classifier (`plate_reid: true`)
- [x] **Plate country detection** (2026-08-19): format heuristics -> DE (incl.
      E/H electric/historic), CZ, SK, AT, PL, HU, IT, ES, UK, BE, FR, SE, NL,
      CH. Displayed as `PLATE: ANONYMIZED [DE 80%]`. Ambiguities (SK/IT,
      DE/HU) are unresolvable without EU-stripe OCR — accepted.
- [x] **Replay mode** (2026-08-19): `source: "<file>.mp4"` runs the whole
      pipeline on a recording at real-time speed, loops at EOF — dev without
      driving (tested 27 fps on the 13-min drive)
- [x] **Blur deferred** (2026-08-19): `anpr.enabled: true` +
      `blur_plates: false` = plates READ locally, never anonymized; flip
      `blur_plates: true` when the project is finished/shared (GDPR)
- [x] **Real plate reading** (2026-08-19): contour heuristic alone found only
      wrong regions (huge dark blobs). Switched to ultralytics' free plate
      detector `anpr-demo-model.pt` → saved as `storage/models/plate.pt`
      (URL: github.com/ultralytics/assets/releases/download/v0.0.0/
      anpr-demo-model.pt). KEY: run it on the (padded) vehicle crop, NOT the
      full frame — a plate at 1920px is too small to survive imgsz 640
      downscaling (`_yolo_plates_crop`, coords offset back to frame space).
      Real read on the 13-min recording: `0472953` (distant cars = small
      plates = low-OCR confidence; upscaling did NOT help — 3x nearest made it
      worse, reverted). Chain verified end-to-end: plate detect → OCR → country
      → re-ID memory.

### 2026-08-20 — audit + performance pass (see §14 for details)

- [x] **Privacy/security audit fixes**: `/api/vehicles` no longer serves raw
      `plate_text` (local DB only); `server.host` default → `127.0.0.1`; SQLite
      log in WAL mode + `session_retention_days` prune (default 30) +
      `model_conf` column (migrated in place); `update_grok` no longer writes
      into `color_conf`; plate re-ID memory has a TTL
      (`anpr.plate_memory_ttl`, 3600s default); `privacy.require_blur` hard
      gate + loud warning when `blur_plates: false`.
- [x] **Smoothness fix — async plate reader**: plate YOLO (~75ms) + easyocr
      (~334ms) moved to a background worker (`AsyncPlateReader`), rate-limited
      to `anpr.plate_refresh_interval` (1.5s) per track. Loop uses cached
      results re-anchored to the car's current box each frame → blur follows
      smoothly. This was THE freeze cause (used to stall ~0.4-1.4s/frame).
- [x] **Smoothness fix — async classifier**: ResNet50 (~40ms) moved off the
      capture thread (`AsyncClassifier`, `classify_refresh_interval` 1.5s);
      color stays synchronous (~1ms). Loop shows last cached label.
- [x] **Detection tuned**: `detection.imgsz` 480 → 320 → 86ms → 12ms/det (7×).
- [x] **Real-time recordings**: writer runs at camera fps with a frame-duplication
      accumulator → playback spans true duration (was fast-forward ~1.4×).
- [x] **Measured**: ~8 fps w/ freezes → **25.5 fps, max frame gap 56ms, zero
      gaps >150ms**, recording real-time match ~94%.
- [x] **Training footguns**: `--min-per-class` default now 1 (+ loud drop
      warning); MPS training warns; tracker mixed-ID re-ID churn fixed.
- [x] `.gitignore` added (models, storage, .env, venv). Not yet `git init`.

## 6. Key files

- `train_classifier.py` — training (MPS+CUDA), `class_key()`, ImageFolderDS,
  `--init`/`--freeze` fine-tune flags
- `label_drive.py` — offline dashcam labeling (yolo.track persist, dense
  sampling, sharpness pick, one Grok call per car)
- `wikimedia_scraper.py` — dataset scraping, ALIASES, _fold
- `grok_classifier.py` — Grok vision fallback (reasoning_effort low, certifi ctx)
- `gemini_classifier.py` — batched Gemini fallback (batch 50, flush 600s, disabled)
- `main.py` — dashcam server, async plate + classifier workers, replay mode,
  plate re-ID, country badge, privacy gates
- `classifier.py` — VehicleClassifier (torchvision or transformers checkpoint) +
  `AsyncClassifier` (deep inference worker thread)
- `anpr_privacy.py` — PrivacyEngine (plate find yolo-crop / OCR / country /
  blur) + `AsyncPlateReader` (background OCR worker)
- `tracker.py` — VehicleTracker (detection source swap point for RunPod)
- `config.yaml` — model_path, imgsz 320, async refresh intervals, host, gates
- `models/r50_dashcam.pt` — **live model** (252 classes, fine-tuned v5)
- `models/r50_filt_60.pt` — pre-dashcam checkpoint (57.6% baseline)
- `.env` — XAI_API_KEY (exposed — rotate), GEMINI_API_KEY, RUNPOD_API_KEY
- `.gitignore` — excludes .env, models, storage, venv, __pycache__, logs

## 7. Commands cheat-sheet

```bash
# Mac training (MPS — DON'T trust results until validated)
SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")
nohup .venv/bin/python train_classifier.py --data storage/dataset/wikimedia \
  --arch resnet18 --epochs 8 --min-per-class 1 \
  --out models/vehicle_make_model_r18.pt > /tmp/train_r18.log 2>&1 < /dev/null & disown

# RunPod training (CUDA — the working path)
ssh -i ~/.ssh/RunPod root@157.157.221.29 -p 57145
# upload: tar -cf - storage/dataset/wikimedia | ssh ... "cd /root/train && tar -xf -"
setsid bash /root/train/launch.sh </dev/null >/dev/null 2>&1 &
# progress: grep '^epoch' /tmp/train_pod.log

# Server
cd ~/mobile_tracker && nohup .venv/bin/python main.py > /tmp/mt_run.log 2>&1 &
# -> localhost:8500
```

## 8. M2 vs NVIDIA (hardware notes)

- M2 Air GPU ≈ GTX 1650 (~2.6 TFLOPS)
- RTX 3060/4060 ≈ 4-5x faster for YOLO; 4090 ≈ 20-30x
- RTX 2000 Ada (16GB) — workstation card, plenty for ResNet50 fine-tune
- Camera feed + latency are the real constraints, not GPU compute
- MPS: torch 2.13 MPS had silent training corruption on this machine — always
  validate MPS-trained models (loss should drop like CUDA does)

## 9. Notes for the report

- The full debugging arc (MPS silent failure -> diagnostics -> CUDA fix) is the
  most interesting story: two-class test, frozen-head probe, zero-shot ImageNet
  check, batch-32 memorization test
- Dataset quality matters more than model size for fine-grained recognition
- Wikimedia categories contain lots of non-car photos (street scenes) — noisy
  data hurts val accuracy
- RunPod workflow: key auth via `~/.ssh/RunPod`, tar-over-ssh upload, setsid
  launcher scripts; direct-IP pods (plain ssh/scp) beat SSH-gateway pods
  (PTY-only, no scp/sftp)
- The dashcam fine-tune arc (v3 -> v4) is the second big story: overconfident
  hallucinations on unseen classes (Land Rover collapse), oversampled-val
  overconfidence, label normalization (accents/underscores), and why honest
  "Unknown" gates matter
- Live Grok hybrid mode: `reasoning_effort: low` cut 20-30s -> ~5s/call
  without quality loss — cheap (pennies per drive) and accurate (0.85-0.99)

## 10. Experiment log (accuracy table)

| Run | Data | Epochs | Train % | Val % | Verdict |
|-----|------|--------|---------|-------|---------|
| r18 lr1e-3 | raw | 8  | 7.5  | 6.1  | MPS broken |
| r18 lr3e-3 | raw | 15 | 3.9  | 3.2  | MPS broken |
| r18 CUDA   | raw | 60 | 99.7 | 46.9 | baseline, saved |
| r18 + crop | crop | 20 | 99.7 | 40.5 | rejected (wrong-car crops) |
| r18 filter | filt | 20 | 99.7 | 48.2 | accepted |
| r18 60ep   | filt | 60 | 99.7 | 48.1 | saved (r18_filt_60.pt) |
| r50 60ep   | filt | 32* | 99.6 | **57.3** | web baseline (57.6/75.6 final) |
| r50 dashcam v1 | merged (32 crops x30) | 8 | 99.7 | 93.2 | memorized — Land Rover collapse on unseen cars |
| r50 dashcam v2 | merged (~100 crops x30) | 8 | 99.7 | **97.6** | **LIVE** — correct on EU crops 0.97-1.00 |
| r50 v3 (247 cl) | merged +5 web classes (min-per-class 15!) | 8 | 99.8 | 91.2 | **WRONG** — silently dropped 5 old classes, web-only new classes failed on real crops |
| r50 v4 (252 cl) | merged +5 web x30 | 8 | 99.8 | 95.5 | 4/5 new classes failed real crops (web domain gap) |
| r50 v5 (252 cl) | merged +5 web x30 + 2 real crops x30/class | 8 | 99.8 | **95.6** | **LIVE** — all 5 new classes 0.99+ on real crops |

\* r50 stopped early at epoch 32 (val plateaued at 57.3% since ~epoch 25); the
best checkpoint was already saved. r50 final eval (local, filtered val split,
n=1317): **top-1 57.6% / top-5 75.6%**. Worst classes are tiny ones (2-4 val
images each) — expected.

- Filtered dataset: 10,573 -> 9,520 images (10.0% noise dropped by YOLOv8n
  car detection, `filter_crops.py --no-crop`) — also rebuilt locally at
  `storage/dataset/filtered` (9,521 imgs, 9.9% dropped — matches)
- `eval_model.py` now auto-detects arch (r18 vs r50 via fc in_features) and
  computes top-1 + top-5 + worst classes
- **config.yaml model_path = models/r50_dashcam.pt** (the live model)
- Classifier smoke-tested on Mac: Golf/Tesla Model 3/Skoda Octavia all correct
  (conf 0.99-1.00), load time 1.9s
- HF dataset card drafted: `HF_DATASET_CARD.md`

## 11. RunPod auto-shutdown (worked)

- `auto_shutdown.sh` watcher: polls chain log -> downloads checkpoints ->
  verifies -> terminates pod via RunPod GraphQL API (`podTerminate` returns
  Void — no selection block!)
- API key in `.env` as RUNPOD_API_KEY (gitignored)
- Hard-deadline safety + "don't terminate if verification fails"
- Pod terminated 00:39, checkpoints safe on Mac. Lesson: podTerminate mutation
  must NOT have `{ id desiredStatus }` selection.

## 12. SSH gateway vs direct-IP pods (2026-08-19)

- Gateway pods (`<podid>-<port>@ssh.runpod.io`): interactive SSH needs a PTY —
  plain `ssh`/`scp`/`sftp` all fail with "Your SSH client doesn't support PTY"
  or "subsystem request failed". Workaround: `expect` wrapper driving an
  interactive session (`/tmp/pod_run.sh` pattern) — works for commands.
- File transfer to gateway pods: no scp/sftp/rsync; tried port-forward
  (`-L 22222:localhost:22` — resets; `-R 8000:...` — gateway doesn't bind),
  HTTP reverse proxy unreachable, public file hosts fail for 869MB
  (tmpfiles.org 413).
- **Solution: use pods with a public IP + SSH port** (cloudType SECURE,
  sometimes community) — plain ssh/scp/tar-over-ssh all work. When a gateway
  pod is the only option, expect-based interactive upload in base64 chunks is
  the fallback (slow).
- Username suffix changes between pod rebuilds (`-64411a2d` vs `-64412484`) —
  re-read the SSH command RunPod shows in the UI.

## 13. COMPACTION — FULL SESSION STATE (2026-08-19, handoff)

> Read this first. Everything the agent needs to resume the project.
>
> **UPDATE 2026-08-20**: the project was resumed, audited and smoothed — see
> §14 for the audit findings, the performance pass and the new config. §13's
> objective/structure is unchanged; several of its "next steps" (retrain,
> blur-on-shipping) are still open.

### Objective
Fully-local dashcam make/model recognition: fine-tuned ResNet50 (252 classes),
Grok OUT of the live path. Live pipeline: YOLOv8n car detect -> resnet50
classify -> color + plate (country + re-ID) enrichment. GDPR blur deferred.

### Where things stand (DONE)
- **LIVE model**: `models/r50_dashcam.pt` — 252 classes, val 95.6%,
  deployed. All 5 new EU classes (Kia Ceed, Kia Rio, Suzuki SX4, Hyundai
  Kona, Peugeot 406) verified 0.99+ on real dashcam crops; old classes
  unaffected. Previous models saved: `/tmp/r50_dashcam_247b.pt` (v4, 247
  classes, val 97.6%), `/tmp/r50_dashcam_247c.pt`.
- **config.yaml**: `source: 0` (Iriun phone camera, default), `model_path:
  models/r50_dashcam.pt`, `grok.enabled: false`, `fallback_conf: 0.45`,
  `anpr.enabled: true`, `anpr.blur_plates: false` (local dev — plates read,
  not blurred; flip true when shipping), `anpr.plate_reid: true`,
  `plate_model: storage/models/plate.pt`, `min_classify_area: 12000`,
  `min_box_side: 90`, `min_display_conf: 0.45`.
- **Dataset** `storage/dataset/merged`: 252 class dirs, ~19.3k jpg
  (9,521 filtered web hardlinks + ~3,120 dashcam crops x30 + 6,420 new
  web x30 + 300 new dashcam crops x30). Scraper MODELS dict updated with
  the 5 new cars.
- **Features completed**: replay mode (source = mp4 path), plate country
  detection (PLATE_PATTERNS: DE/CZ/SK/AT/PL/HU/IT/ES/UK/BE/FR/SE/NL/CH),
  plate re-ID memory (same plate -> reuse attrs, SKIP classifier).
- **Security**: ALL API keys wiped from `.env` (empty placeholders remain:
  XAI_API_KEY, RUNPOD_API_KEY, GEMINI_API_KEY). Old keys were exposed in
  chat history — user must rotate/paste fresh keys. No git repo, nothing
  to scrub from history. Verified: no keys in code, configs, /tmp, zsh
  history.
- **Server stopped** (user's request). Next run: `nohup .venv/bin/python
  main.py > /tmp/mt_run.log 2>&1 & disown`; API at localhost:8500.

### Environment gotchas (CRITICAL)
- **MPS training silently broken** — never train on this Mac (M2). CUDA
  (RunPod pod) only. Local Mac = eval/inference.
- **SSH key**: `~/.ssh/RunPod` (id_ed25519 does NOT exist on this Mac).
  User provides direct-IP pods: `ssh -i ~/.ssh/RunPod root@<ip> -p <port>`.
  Direct-IP pods = plain ssh/scp/tar work (gateway ssh.runpod.io pods are
  PTY-only, no file transfer — avoid).
- **Background launch on pod**: `setsid bash -c '...' </dev/null
  >/dev/null 2>&1 &` (plain nohup dies when SSH session closes).
- **Pod file ownership**: tar-extracted files carry uid 501 (Mac) -> not
  writable by root; `chown root:root` after extract.
- **RunPod GraphQL**: bearer auth + `User-Agent: Mozilla/5.0` + certifi ctx;
  `podTerminate` returns Void (no selection). EU pods: EU-CZ-1, EU-RO-1,
  EU-NL-1, EU-FR-1, EU-SE-1. Image `runpod/pytorch:1.1.0-rc.154-cu1290-
  torch291-ubuntu2404` + `pip install opencv-python-headless numpy pillow
  tqdm`. 401 = key rotated, ask user.
- **Training flags**: MUST use `--min-per-class 1` (default 15 silently
  DROPS classes with <15 images — cost us a rerun). For class-count
  changes, `--init` needs fc stripped: patched script drops keys starting
  with `fc.` (fresh head, backbone preserved). Patch is in the LOCAL
  train_classifier.py; if uploading fresh, re-apply.

### Key lessons (why, not just what)
- Land Rover collapse: classes unseen in training (US cars) -> model falls
  back to its strongest prior. Web-only training doesn't generalize to
  dashcam (domain gap) — always inject real crops, even 2 per class x30.
- Val on oversampled sets is meaningless (memorization). Real-crop sanity
  check is the truth.
- Checkpoint class names use SPACES ("Audi A6"), folders use underscores.
  Accent folding: Škoda->Skoda. All mapping must normalize.
- Plate contour heuristic useless on real footage (found dark blobs);
  YOLO plate model (`storage/models/plate.pt`, ultralytics asset:
  github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-model.pt)
  must run on the VEHICLE CROP not full frame (plates too small at 1920px
  for imgsz 640 downscale). Upscaling crops before OCR hurt — reverted.
- Gemini free tier: 5 req/day, 250k tokens — small validation jobs only.

### Next steps (when user returns)
1. User pastes fresh API keys into `.env` when needed (RunPod/Gemini).
2. More EU drives -> more dashcam crops -> future fine-tunes (crop
   collection: `label_drive.py <mp4>` — Grok labels offline, ~$0.35/drive).
3. Optional: big open vision model (Qwen2.5-VL 7B) on rented GPU for
   offline labeling at ~$0.10/drive vs tokens.
4. When shipping: `anpr.blur_plates: true` (GDPR), HF dataset upload,
   PROJECT_REPORT.txt review.
5. User is starting ANOTHER project after this handoff — treat
   mobile_tracker as parked until explicitly resumed.

### Key files
- `main.py` — pipeline, replay mode, plate re-ID, country badge
- `anpr_privacy.py` — plate find (yolo-crop), OCR, country, blur
- `classifier.py` — VehicleClassifier (loads r50_dashcam.pt)
- `train_classifier.py` — training (fc-strip patch, min-per-class 1)
- `label_drive.py` — offline labeling from recordings (Grok)
- `wikimedia_scraper.py` — web dataset (MODELS dict, ALIASES)
- `config.yaml`, `PROJECT_NOTES.md`, `PROJECT_REPORT.txt`
- `storage/recordings/session_20260819_131447.mp4` (13-min EU drive),
  `session_20260819_152400.mp4` (US drive)
- `storage/dataset/{merged,filtered,dashcam_raw,wikimedia}`
- `models/r50_dashcam.pt` (LIVE), `r50_filt_60.pt` (web baseline 57.6%),
  `r18_filt_60.pt` (r18 baseline)

## 14. AUDIT + PERFORMANCE PASS (2026-08-20)

> Full code audit of the live pipeline (every module + smoke test), then a
> privacy/security fix set and a smoothness/performance pass. Server was NOT
> running; smoke test passes after all changes.

### 14.1 Privacy & security (fixes shipped)

| Finding | Fix |
|---|---|
| `GET /api/vehicles` returned raw `plate_text` over an unauthenticated LAN API | `main.py` strips `plate_text` (stays in local DB only); `server.host` default → `127.0.0.1` |
| Plates stored in plaintext SQLite, no retention | `SessionLog` WAL mode + `recorder.session_retention_days` prune (30d default); `privacy.require_blur` hard gate + startup warning when `blur_plates: false` |
| `update_grok` wrote model conf into `color_conf` column | new `model_conf` column (auto-migrated) + corrected mapping |
| Plate re-ID memory grew unbounded | `anpr.plate_memory_ttl` (3600s default) with periodic eviction |
| `.env` keys / no repo hygiene | keys already wiped; `.gitignore` added (models, storage, .env, venv) |

### 14.2 The lag / freeze — root cause

Measured per-frame costs running INLINE in the single capture thread:
- plate OCR (easyocr) **~334ms**, plate YOLO **~75ms** — ran for every car every frame → **0.4-1.4s stalls** whenever a car with a plate passed. That was the "freezing few frames" symptom.
- ResNet50 classify **~41ms** — inline once per car per 30-frame window → a ~60-80ms hitch every second.
- Detection `imgsz 480` **~86ms** on M2/MPS.

Conclusion: **not primarily hardware** — the architecture was the problem. A faster machine would still stutter on inline OCR.

### 14.3 Performance fixes

1. **`AsyncPlateReader`** (`anpr_privacy.py`) — plate YOLO + OCR on a background
   thread, rate-limited to `anpr.plate_refresh_interval` (1.5s) per track.
   Loop reads cached regions re-anchored to the car's current box each frame →
   blur follows smoothly, re-ID/status/DB-match still work.
2. **`AsyncClassifier`** (`classifier.py`) — ResNet50 on a background thread
   (`classification.classify_refresh_interval`, 1.5s). Color stays sync (~1ms).
   Loop shows last cached label; fresh labels appear between windows.
3. **Detection**: `detection.imgsz` 480 → **320** (12ms vs 86ms per det).
4. **Real-time recordings** (`recorder.py`): writer at camera fps + frame-
   duplication accumulator → playback matches real duration (was ~1.4× fast).

### 14.4 Measured (real pipeline, M2 base, replay of 13-min drive)

| Metric | Before | After |
|---|---|---|
| Sustained fps | ~8 (with freezes) | **25.5** |
| Max frame gap | 400ms+ (visible freeze) | **56ms** |
| Frames with gap >150ms | many | **0** |
| Recording vs real-time | ~60-70% (fast-forward) | **~94%** |

### 14.5 New / changed config keys

```yaml
camera:
  source: "storage/recordings/session_*.mp4"   # replay; 0 = Iriun phone
detection:
  imgsz: 320                                   # was 480
classification:
  classify_refresh_interval: 1.5              # async deep-classify cadence
anpr:
  plate_refresh_interval: 1.5                 # async plate/OCR cadence
  plate_memory_ttl: 3600                      # re-ID memory expiry (s)
privacy:
  require_blur: false                         # true = refuse to run if blur off
recorder:
  session_retention_days: 30                  # prune session DB (0 = keep all)
server:
  host: "127.0.0.1"                           # was 0.0.0.0
```

### 14.6 Still open (unchanged from §13)

- Retrain 247→252 head-extend (`--min-per-class 1` now the default; needs a fresh RunPod key).
- Rotate xAI keys; user pastes fresh keys into `.env` when needed.
- Before sharing: `anpr.blur_plates: true` (GDPR) — `privacy.require_blur` can enforce it.
- Optional: `git init` (`.gitignore` is ready); delete stale models
  (`models/stanford_cars_convnext` 750MB unused + old `*_filt_60.pt` checkpoints, ~1GB total).
- Re-verify on real phone footage (replay tests used recorded .mp4s).
## 15. OVERNIGHT SESSION (2026-08-22 -> 23)

- Migrated to M5 MacBook Pro 24GB; env rebuilt; MPS training VALIDATED here
  (2-class protocol: loss drops normally). M5 = ~5-6x faster inference than
  M2 Air; r50 epochs ~10 min locally; RunPod only worth it for big jobs.
- THE BIG BUG: AsyncClassifier froze classifier.available at construction
  (before model load) -> local labels NEVER appeared live. Fixed with a live
  property. Land Rover outputs had been Grok masking this.
- v6 retrained on M5 (285 cl): holdout 31% -> 53.3% top-1. Deployed;
  r50_dashcam_v5_backup.pt kept. Zero LR bias in EU-drive replay.
- Pedestrian/coarse labels fixed (person/bicycle were excluded before).
- label_crops.py: --provider gemini|deepseek; DeepSeek needs
  "thinking":{"type":"disabled"} (else content comes back empty); skips
  already-labeled crops; folds accents at save time. 139 unlabeled crops
  labeled via DeepSeek tonight (+51 earlier via Gemini) -> dashcam_raw=578.
- build_merged.py (reproducible canon/variant-fold/oversample/holdout):
  305 classes, ~25.8k imgs, 56-crop holdout in dashcam_val.
- v7 (305 cl, all new data) trained overnight; result + deploy decision:
  SEE /tmp/v7_compare.txt -> KEPT v6 (holdout 0.364 vs 0.491 — gain < 2 pts)
- Recordings have boxes burned in (recorder writes annotated buffer) — use
  clean YouTube footage for crop mining. 3 long raw YT drives NOT yet on the
  4TB HDD (searched; user will re-download).
- Report rewritten for the M5 era: PROJECT_REPORT.txt.

### Next steps
1. Read /tmp/v7_compare.txt; if KEPT v6, consider more epochs/data next time
2. Label more crops when Gemini quota resets (free) or via DeepSeek
3. Restart dashboard when wanted: SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())") .venv/bin/python main.py &
4. RunPod one-command launcher still to be built (needs fresh RUNPOD_API_KEY)
5. git init; HF dataset upload; blur_plates=true before sharing

## 15. OVERNIGHT SESSION (2026-08-22 -> 23)

- Migrated to M5 MacBook Pro 24GB; env rebuilt; MPS training VALIDATED here
  (2-class protocol: loss drops normally). M5 = ~5-6x faster inference than
  M2 Air; r50 epochs ~10 min locally; RunPod only worth it for big jobs.
- THE BIG BUG: AsyncClassifier froze classifier.available at construction
  (before model load) -> local labels NEVER appeared live. Fixed with a live
  property. Land Rover outputs had been Grok masking this.
- v6 retrained on M5 (285 cl): holdout 31% -> 53.3% top-1. Deployed;
  r50_dashcam_v5_backup.pt kept. Zero LR bias in EU-drive replay.
- Pedestrian/coarse labels fixed (person/bicycle were excluded before).
- label_crops.py: --provider gemini|deepseek; DeepSeek needs
  "thinking":{"type":"disabled"} (else content comes back empty); skips
  already-labeled crops; folds accents at save time. 139 unlabeled crops
  labeled via DeepSeek tonight (+51 earlier via Gemini) -> dashcam_raw=578.
- build_merged.py (reproducible canon/variant-fold/oversample/holdout):
  305 classes, ~25.8k imgs, 56-crop holdout in dashcam_val.
- v8 (305 cl, all new data) trained overnight; result + deploy decision:
  SEE /tmp/v8_compare.txt -> KEPT v6 (holdout 0.567 vs 0.567 — gain < 2 pts)
- Recordings have boxes burned in (recorder writes annotated buffer) — use
  clean YouTube footage for crop mining. 3 long raw YT drives NOT yet on the
  4TB HDD (searched; user will re-download).
- Report rewritten for the M5 era: PROJECT_REPORT.txt.

### Next steps
1. Read /tmp/v8_compare.txt; if KEPT v6, consider more epochs/data next time
2. Label more crops when Gemini quota resets (free) or via DeepSeek
3. Restart dashboard when wanted: SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())") .venv/bin/python main.py &
4. RunPod one-command launcher still to be built (needs fresh RUNPOD_API_KEY)
5. git init; HF dataset upload; blur_plates=true before sharing

### 2026-08-23 morning — v8 verdict + deploy
- Night pipeline: trained, compared, KEPT v6 by the +2pts rule (top-1 tie
  56.7% on n=30 holdout). Manual override: DEPLOYED v8 anyway — equal top-1,
  top-5 86.7% vs 76.7% (much better ranking), and it is the first model
  trained exclusively on verified-clean labels. r50_dashcam.pt = v8 now.
- cross_check_labels.py added: two-provider verification, newest-first,
  moves disagreements to storage/dataset/quarantine/ (201 moved).
- Next real separator = MORE holdout footage; 3 YT drives still to download.

### Performance rules for the M5 (learned 2026-08-23)
- ONE heavy GPU job at a time. Running the dashboard replay (27fps, MPS)
  alongside extract/train pushed temps to 95-100degC (throttling) at full
  fans. Killing the server -> 89degC and more speed for the real job.
- Before training/extraction: pkill -if main.py (restart server afterwards
  with SSL_CERT_FILE=certifi .venv/bin/python main.py &).
- Keep lid open + caffeinate -w <pid> for any unattended run; pmset sleepnow
  at end of chain scripts (see after_v8.sh pattern).
- num_workers=0 on MPS DataLoaders (fork workers segfault), 4 on CUDA.

## 15. OVERNIGHT SESSION (2026-08-22 -> 23)

- Migrated to M5 MacBook Pro 24GB; env rebuilt; MPS training VALIDATED here
  (2-class protocol: loss drops normally). M5 = ~5-6x faster inference than
  M2 Air; r50 epochs ~10 min locally; RunPod only worth it for big jobs.
- THE BIG BUG: AsyncClassifier froze classifier.available at construction
  (before model load) -> local labels NEVER appeared live. Fixed with a live
  property. Land Rover outputs had been Grok masking this.
- v6 retrained on M5 (285 cl): holdout 31% -> 53.3% top-1. Deployed;
  r50_dashcam_v5_backup.pt kept. Zero LR bias in EU-drive replay.
- Pedestrian/coarse labels fixed (person/bicycle were excluded before).
- label_crops.py: --provider gemini|deepseek; DeepSeek needs
  "thinking":{"type":"disabled"} (else content comes back empty); skips
  already-labeled crops; folds accents at save time. 139 unlabeled crops
  labeled via DeepSeek tonight (+51 earlier via Gemini) -> dashcam_raw=578.
- build_merged.py (reproducible canon/variant-fold/oversample/holdout):
  305 classes, ~25.8k imgs, 56-crop holdout in dashcam_val.
- v9 (305 cl, all new data) trained overnight; result + deploy decision:
  SEE /tmp/v9_compare.txt -> DEPLOYED v9 (holdout 0.717 vs 0.473)
- Recordings have boxes burned in (recorder writes annotated buffer) — use
  clean YouTube footage for crop mining. 3 long raw YT drives NOT yet on the
  4TB HDD (searched; user will re-download).
- Report rewritten for the M5 era: PROJECT_REPORT.txt.

### Next steps
1. Read /tmp/v9_compare.txt; if KEPT v6, consider more epochs/data next time
2. Label more crops when Gemini quota resets (free) or via DeepSeek
3. Restart dashboard when wanted: SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())") .venv/bin/python main.py &
4. RunPod one-command launcher still to be built (needs fresh RUNPOD_API_KEY)
5. git init; HF dataset upload; blur_plates=true before sharing

### 2026-08-23 evening — batch #2 flywheel (automated)
- Second YT batch: Brussels/Paris/Prague/Frankfurt/Vienna/+1 (290 min).
- Extraction -> DeepSeek labeling -> rebuild -> v10 (from v9 init), all run
  by after_v10.sh. Outcome: SEE /tmp/v10_compare.txt -> KEPT v9 (0.521 vs 0.514)
- Gemini cross-check of tonight's labels: PENDING (free quota resets next
  morning) — run cross_check_labels.py --provider gemini before trusting v11.

### 2026-08-24 morning status
- v10 (336 cl): holdout top-1 52.1 / top-5 78.1 vs v9 51.4/65.3 — KEPT v9 by
  the +2 rule, but v10 ranks much better. v11 (batch-3 data) will settle it.
- Batch 3 extraction: parallel-by-video workers (3x processes) — CPU 40->80%,
  ~2x faster. Pool peaked at 4,200+ crops pre-labeling.
- DeepSeek batch fix: max_tokens 250/img + batch 20 = zero parse failures.
- cross_check_labels.py: --model flag (free quota is PER MODEL:
  gemini-3.5-flash-lite AND gemini-3.6-flash each give daily requests);
  resilient to transient parse failures (skip chunk, continue).
- Plate A/B quick test inconclusive (tiny sample); real verdict = live drive.
- road_context.py added (opt-in road.enabled): light states + confirmed-only
  speed limit w/ explicit '?' unknown. Light TTL=1s per user preference.
- morning_start.sh: one-command live drive mode (Iriun + road on).
- Local video copies deleted after mining (originals stay on 4TB HDD).
- RunPod: user launches pods manually via web UI + hands SSH string (API key
  stays out of .env by choice). Only needed for ConvNeXt-class upgrades.

### 2026-08-24 overnight — batch #3 mega-run (automated)
- Batch 3: 17 videos / 26GB — DK x3, PL x2, IT, HU, HR, SK, PT, LU, BE, AT,
  UK-night. Full flywheel run by after_v11.sh.
- Labels are DeepSeek-only tonight; Gemini cross-check STILL PENDING — run
  cross_check_labels.py --provider gemini (cycle models: 3.5-flash-lite /
  3.6-flash) before trusting v12 training on these.
- Outcome: SEE /tmp/v11_compare.txt -> KEPT current (v11 gain < 2pts or parse issue)

### RunPod + power lessons (2026-08-24)
- Pod SSH keys are PER-POD: keys injected via env PUBLIC_KEY only apply on
  (re)start; port CHANGES after restart (56963->56898). Old Mac's key was
  registered, new Mac wasn't -> Permission denied until fixed.
- Fix flow that WORKED: REST api.runpod.io/v2/pods (GET list -> PATCH with
  {name,image,env,ports} minimal payload — full dict 422s on read-only
  fields) to append M5 pubkey into PUBLIC_KEY, then user clicks Restart in
  dashboard UI. GraphQL podUpdatePod/podStopPod no longer exist;
  introspection blocked; rest.runpod.io is the website not API.
- macOS has NO setsid — use plain nohup+disown for local chains.
  setsid pattern is Linux-pod-only.
- Launching remote scripts: put the & INSIDE the ssh quoted command
  ('nohup ... & echo LAUNCHED') or the tool timeout kills the job.
- Wait-for-copy race: poll pgrep only AFTER sleep(5) so a freshly spawned cp
  exists; verify file count+sizes before declaring done.
- 30W charger + sustained GPU training = break-even (~25-32W burn): battery
  hovers, epochs stretch ~40min under SoC power cap. 96W = full speed.
  In-car 20-30W + needing to WORK => move training to pod, free the Mac.
- Two architectures racing (r50 local-safe vs ConvNeXt pod-experiment),
  single holdout compare, deploy winner = the pattern that works.
- DeepSeek labeling at scale: batch 20, max_tokens 250/img+500,
  thinking disabled. Gemini verify next morning cycling models per quota.

### 2026-08-24 afternoon — CONVNEXT DEPLOYED (v12-era model)
- Pod 3-way holdout verdict: ConvNeXt-Tiny 47.9/61.9 > r50-v11 42.3/61.9 >
  v9 25.7/38.8 (571-crop leak-free holdout). DEPLOYED convnext_v1.pt as
  models/r50_dashcam.pt; r50_v9_deployed_backup.pt + r50_v11.pt kept.
- CRITICAL loader fixes for ConvNeXt checkpoints:
  * compare_ckpts.py load_model: branch on fc.weight / classifier.2.weight
  * classifier.py: torchvision ConvNeXt head = model.classifier[-1] (NOT
    .head); ALSO move model to MPS at load — inference-only, immune to the
    old training bug, and the difference is 193ms CPU vs 6.6ms MPS!
- Live verified: 26.6fps, classifier loads 1.2s, 371 classes.
- Pod artifacts fetched (convnext_v1.pt, r50_v11.pt, pod_compare.txt) —
  POD CAN BE TERMINATED in dashboard when user wants.

### WORKFLOW PREFERENCE (user, 2026-08-24)
- WORKING DAYS: train on RunPod (user can't watch the notebook). User rents
  pod via web UI + hands SSH string/API key; terminate via dashboard when done.
- HOME DAYS: M5 MacBook local training is fine.
- User prefers RunPod when possible — keep pods short-lived, fetch artifacts,
  terminate promptly.

### 2026-08-24 evening — GTSRB sign reader DONE
- gtsrb_train.py: resnet18 fine-tune on 43-class GTSRB (39k signs) -> val
  99.7%. Checkpoint models/signs/gtsrb_signs.pt.
- SignClassifier in road_context.py: replaces digit-OCR for speed signs;
  real-crop test 28/28 speed limits read correctly (20/30/50/60/80/100/120).
  NOTE: synthetic drawn signs fail (font mismatch) — test with real crops only.
- road.enabled=true + sign_model=models/signs/gtsrb_signs.pt activates it.
- v12 (ConvNeXt on batch-3 labels) did NOT beat v1 (42.9 vs 47.9 top-1) —
  unverified labels hurt; v1 stays deployed. Batch-3 labels need full Gemini
  verification before next training round.
- ALL pod artifacts fetched -> POD CAN BE TERMINATED (nothing left on it).

### 2026-08-24 night — ROAD MODULE LIVE-VERIFIED
- Replay test with road.enabled=true + gtsrb_signs.pt: speed limits READ AND
  CONFIRMED from real footage (60->50->80, conf .84-.88). Device-mismatch bug
  fixed (normalize AFTER .to(mps)). Zero worker errors, 25-27fps.
- Batch 4 incoming tonight: Amsterdam (bikes/peds — coarse labels handle),
  night (expect low-conf rejects = honest), rainy (new conditions!).
- Next session: morning drive validation, Gemini verify batch 3 leftovers +
  all of batch 4, then v13 training.

### 2026-08-24 night — after_v13.sh armed (full autonomous night)
- extract batch4 (3 workers) -> deepseek label rounds -> SLEEP until 08:10 ->
  gemini cross-check (3.6-flash + 3.5-lite) -> rebuild -> train convnext_v13
  (init=deployed) -> compare vs deployed (+2 rule) -> deploy -> notes -> sleep.
- User decision: LOCAL training tonight (Mac home, free). Pod option open if
  they carry notebook to work: hand SSH before 08:30 to redirect training.

### 2026-08-25 overnight — batch 4 flywheel (automated)
- Batch 4: Madrid / rainy Paris / Rome night / Amsterdam / Zurich.
- after_v13.sh ran the full loop incl. morning Gemini verification BEFORE
  training (v12 lesson applied). Outcome: SEE /tmp/v13_compare.txt -> KEPT current (0.644 vs 0.695)

### 2026-08-25 afternoon — plate A/B + v14 prep
- Plate OCR A/B on EU drive (2000f, 73 plates): raw 56% vs enhanced 59% read
  rate (+3%) — keep the LANCZOS/CLAHE enhancement (cheap win, no harm), but
  plates are resolution/OCR-limited, not pre-processing-limited.
- label_crops now skips known-reject crops (label_rejects.txt) = stops
  re-burning DeepSeek tokens on the ~2,179 low-conf rejects.
- after_v14.sh: auto Gemini-verify batch-4+tail labels in morning, rebuild,
  package /tmp/v14_bundle.tar for a pod run. v14 = next training round.
- v13 deployed (454 classes, holdout top-1 69.3 / top-5 84.0 — top-5 +4.5
  and +83 classes over v1). ConvNeXt-Small experiment training on pod now.

### NEXT-RUN PLAN (written 2026-08-25 night, for v14+)
- Pod: ConvNeXt-Base @336px b32 training NOW (12 ep, ~4-5h — exceeds the
  3.5h credit window, but best-checkpoint auto-fetches every 15 min via
  pod_guardian.sh -> models/convnext_base_336_latest.pt. Nothing lost on
  credit cutoff; resume later = restart training, it re-converges fast).
- v14 candidates (compare all on leak-free holdout):
    a) convnext_base_336 (pod)     b) convnext_v13 tiny @224 (local)
    c) deployed convnext_v1
- Before v14 training: Gemini cross-check batch-4 + tail labels (quota resets
  ~09:00 CEST). Quarantine disagreements FIRST, then train.
- Known fixes shipped tonight: label shuffle (tail starvation), reject-skip
  list (token burn), person->motorbike aspect fix, EU band light classifier,
  color LAB-kmeans behind color_algorithm flag (needs live A/B).
- Open user-feedback items: motorcycle detection in Amsterdam footage,
  traffic-light state quality re-check with new band logic on real lights,
  color A/B live.

### POD SSH STABILITY (2026-08-25 night — read before every pod session)
- THE FIX THAT WORKS: add agent's pubkey to RunPod ACCOUNT Settings -> SSH
  Keys (not just env injection!). Then STOP+START the pod. Port CHANGES after
  every restart - always get fresh "SSH over exposed TCP" line from dashboard.
- Injection via REST (PATCH /v2/pods/{id} env PUBLIC_KEY) works but needs a
  restart to apply, and restarts CHANGE the port + sometimes reset container.
- sshd rate-limits rapid reconnects ("Permission denied" storms): wait 3-5
  min between attempts, use ONE ControlMaster connection for all commands:
    ssh -o ControlMaster=auto -o ControlPath=/tmp/pod_cm -o ControlPersist=1800 ...
- scp uploads drop mid-transfer on flaky pods: verify remote size == local
  size after upload; re-run rsync/scp if mismatch.
- Web terminal (JupyterLab) always works even when SSH auth fails - fallback.

### 5090 LESSON (2026-08-26) — DO NOT RENT 5090s ON COMMUNITY CLOUD
- Two separate RTX 5090 pods failed identically: cuInit error 999 (container
  GPU binding broken: /dev/nvidia5 mis-mapped), random VM crashes wiping
  container disk. Blackwell deployments on RunPod community cloud are
  immature. 4090 / 2000-3000 Ada = proven stable path.
- Account-level SSH key registration works (Settings -> SSH Keys) — new pods
  accept our key at creation. Always add key BEFORE renting.
- Tonight's plan: LOCAL M5 trains Tiny-v14 on newest verified dataset
  (~47min/epoch x10 ≈ 8h, done by morning). Pod experiments resume on 4090s.

---

## 2026-08-28/29 — Data-diversity breakthrough session

### Discoveries (in order)
1. **Leaky holdout found**: build_merged.py split dup files across dirs (BMW_3_Series +
   BMW_3_Series_Touring → same class) putting the same track on train AND val. Fixed with
   global filename dedupe. ALL pre-08-28 numbers suspect (incl. "v16 71.9%" and old arch scores).
2. **6-config sweep on clean holdout (890 crops) — everything loses to king (67.6%)**:
   tiny224 44.4 / small224 43.0 / base224 44.5 / large224 44.9 / tiny336 38.7 / base336 38.0.
   Bigger archs don't help; 336px HURTS.
3. **Root cause identified**: train_v6 was 93% oversampled dashcam copies (~4,728 unique crops
   x30 = 141,840). Models memorized duplicates → 97% internal val, 44% holdout.
   The lever = MORE UNIQUE CROPS, not architecture/resolution.
4. **MPS local training is fine** — the "gradient corruption" fear was a false alarm;
   the confusion came from the leaky data, not Apple GPU.

### Data work done
- Extracted 11,019 unique crops from 32 dashcam videos (4TB drive /Volumes/4TB/Driving videos/).
- DeepSeek-labeled 4,511 new crops into dashcam_raw (10,398 crops / 540+ classes total).
- SKIPPED Gemini re-verify (user decision — flash-lite too aggressive, false-quarantines).
- Rebuilt train_v6: 580 classes, 51,158 imgs, oversample 30x→5x, holdout 1,678 crops/243
  classes, leak-free (verified 0 same-class track leaks).

### Pods
- u3116rfml40v38 (4090): ran the 6-config sweep, all checkpoints fetched, pod terminated.
- Night runner auto-fetch worked but lagged; manual fetch of last 2 ckpts was done instead.

### Next
- Rent fresh pod → upload pod_bundle_v17/ → v17_train_pod.sh (tiny/base 224, 12-16 epochs,
  from-scratch) → compare vs king → deploy if +2pts → live-drive validation.

---

## 2026-08-29 — v17: data-diversity fix, first results

### Setup
- Rebuilt train_v6 with the 11,019 new extracted crops + DeepSeek labels:
  580 classes, 51,158 imgs (10,783 web + 40,375 dashcam @ 5x oversample, down from 30x).
- Holdout enlarged to 1,678 crops / 243 classes (leak-free, verified).
- Pod z7sg1djt3qxove (4090), v17_train_pod.sh: tiny224×12, base224×12, tiny224×16 (from scratch).

### v17 RESULTS (1,678-crop holdout)
| model | top-1 | top-5 |
|---|---|---|
| king r50_dashcam | 62.5% | 73.9% |
| v17 tiny224 (12ep) | 60.4% | 78.0% |
| v17 base224 (12ep) | 60.0% | 77.5% |
| v17 tiny224e16 (16ep) | (running 08:03 UTC) | |

- MASSIVE improvement over the v16 sweep (was 23pts behind king at 44%; now only 2.5pts
  behind top-1 and AHEAD on top-5 by ~+4pts).
- Confirms: data diversity was the bottleneck, not architecture.
- Internal val ~83-84% (down from 97% in v15/v16) because holdout is bigger+harder and
  oversampling reduced — this is HEALTHY (no more memorization).
- v17 checkpoints auto-fetched to models/v17/ + pod auto-terminated when done (finisher).

### v17 FINAL (all done, pod terminated 08-29 ~10:35 UTC)
| model | top-1 | top-5 |
|---|---|---|
| king r50_dashcam | 62.5% | 73.9% |
| v17 tiny224 (12ep) | 60.4% | 78.0% |
| v17 base224 (12ep) | 60.0% | 77.5% |
| v17 tiny224e16 (16ep) | 59.7% | 77.0% |

- More epochs did NOT help (59.7 vs 60.4) — saturated; limit is data, not training length.
- All v17 models within ~2.5pts of king top-1 but +3-4pts better top-5.
- King still leads top-1 (trained on original real-dashcam distribution longer).
- Checkpoints: models/v17/v17_{tiny224,base224,tiny224e16}.pt (fetched, pod off).
- DECISION: king stays deployed for now (top-1 lead). v17 close enough that the next
  data bump (more unique crops) should push past it. Consider live-drive A/B of v17_tiny224
  (best top-5) vs king for real-world feel.

## 2026-08-29 — fifth batch videos (v18 prep)
- User added 6 new dashcam videos (18GB) to /Volumes/4TB/Driving videos/fifth batch:
  Prague, Barcelona, Rome, Munich, Stuttgart, Hamburg (long 4K city drives).
- Processing DIRECTLY from the HDD (no Mac copy) via extract_crops.py (PID 7665).
- Plan: extract → DeepSeek label → rebuild train_v6 (5x oversample) → train v18 on pod.

## 2026-08-29 — v18 dataset ready (fifth batch integrated)
- All 6 fifth-batch videos extracted: +3,147 crops (14,166 total in dashcam_youtube).
- DeepSeek-labeled: +2,785 new (dashcam_raw now 13,183 crops / 630+ classes).
- train_v6 rebuilt: 655 classes, 62,288 imgs (10,783 web + 51,505 dashcam @ 5x).
- Holdout: 2,237 crops / 269 classes, leak-free.
- pod_bundle_v18/ ready: train_v6_v18.tar (1.18GB) + dashcam_val_v18.tar + v18_train_pod.sh
  (tiny224×12, base224×12, tiny224×16, from-scratch, auto-compare vs king).

## 2026-08-29 — v18 BREAKTHROUGH: KING BEATEN 🏆
- 655 classes, 62k imgs (13,183 dashcam crops, 5x oversample, leak-free).
- Holdout: 2,237 crops / 269 classes (harder than before).
- Pod vpofq2xbd0l1ku (4090), v18_train_pod.sh: tiny224×12, base224×12, tiny224×16.

### RESULTS (2,237-crop holdout)
| model | top-1 | top-5 |
|---|---|---|
| king r50_dashcam | 55.8% | 71.4% |
| v18 tiny224 (12ep) | **58.5%** | **79.5%** |
| v18 base224 (12ep) | 58.4% | 78.8% |
| v18 tiny224e16 (16ep) | 57.5% | 77.2% |

- ALL v18 models beat king on top-1 (+1.7 to +2.7) and top-5 (+5.8 to +8.1).
- v18_tiny224 is the new best model — DEPLOY CANDIDATE.
- Checkpoints: models/v18/*.pt (fetched, pod terminated).
- NEXT: deploy v18_tiny224 as new king (backup r50_dashcam.pt) → live-drive validation.

## 2026-08-29 — data quality audit + decision
- wikimedia images: median 341x512, 0% <200px — FINE as supplement (model resizes to 224).
- dashcam_raw: 778 classes (many empty from labeling), 13,183 crops; 451 classes <5 crops.
- DECISION: skip wikimedia re-scrape (low value). The proven lever is REAL dashcam crops.
- Next data round: more YouTube dashcam videos on HDD (/Volumes/4TB/Driving videos/) →
  extract_crops → DeepSeek label → rebuild. Focus on weak classes (<5 crops) if possible.

## 2026-08-29 night → 08-30 — v19 overnight pipeline
- 5 more videos (sixth batch) extracted, DeepSeek-labeled, dataset rebuilt.
- pod_bundle_v19/ packaged (tiny224x12 + tiny224x16 configs).
- See models/v19/MORNING_REPORT.txt for final numbers.

## 2026-08-30 — v19: bigger dataset, new king deployed
- 679 classes, 69,928 imgs (15,080 raw crops, 5x oversample). Holdout 2,606 crops / 286 classes.
- Pod v7d93478j2wvl8 (4090). v19_train_pod.sh: tiny224x12, tiny224x16.

### RESULTS (2,605-crop holdout)
| model | top-1 | top-5 |
|---|---|---|
| v18 king (r50_dashcam_v18_backup) | 52.3% | 67.1% |
| v19 tiny224 (12ep) | **56.9%** | **77.7%** |
| v19 tiny224e16 (16ep) | 56.4% | 76.7% |

- v19_tiny224 wins by +4.6 top-1 / +10.6 top-5 vs the v18 king on the same (bigger) holdout.
- Deployed: models/r50_dashcam.pt = v19_tiny224 (646 classes). v18 backed up.
- Checkpoints: models/v19/*.pt. Pod terminated.
- NOTE: holdout grew again (2,606), so raw numbers differ from v18's 2,237-crop run —
  fair comparison is king-vs-v19 on the SAME set (above).

## 2026-08-30 — v20 dataset (19 new videos) + Gemini reject recovery
- 19 new videos downloaded (seventh batch, ~21 usable; Munich-in-rain too broken to keep).
- Extraction: 21,603 crops total in dashcam_youtube.
- DeepSeek labeled 19,917; 2,526 rejects (Unknown/low-conf).
- Gemini recovery (gemini_recover_rejects.py): 337 recovered before free-tier quota hit.
  Recovery rate 77-100% — DeepSeek is over-conservative on hard crops.
  NOTE: 1,824 rejects remain — re-run recovery after Gemini quota resets (daily).
- Final v20: train 788 classes / 90,963 imgs (5x oversample), holdout 340 classes /
  3,570 crops, LEAK-FREE. Bundle pod_bundle_v20/ (1.5GB).
- Scripts: gemini_recover_rejects.py (second-opinion recovery), hdd_download.sh (yt-dlp to HDD).
- DeepSeek alone on rejects: 0% (it genuinely can't do them) — Gemini is the recovery tool.

## 2026-08-30 — v20 training + RunPod agent setup

### v20 data (final)
- 19 new videos (seventh batch) → 21,603 crops extracted
- DeepSeek labeled 19,917; Gemini recovered +337 (gemini_recover_rejects.py, 77-100% rate)
- 1,824 rejects remain — Gemini quota reset tomorrow → rerun recovery for v21
- Final v20: train 788 classes / 90,963 imgs (5x oversample), holdout 340 classes / 3,570
  crops, LEAK-FREE. Bundle pod_bundle_v20/ (1.5GB).

### v20 training (pod tjz5fd4sdhwd8s, 4090, port 10790)
- Launched 19:08 UTC: tiny224×12 + tiny224×16, from-scratch, auto-compare vs v19 king (56.9%).
- Finisher: /tmp/v20_finisher.sh (fetches both ckpts + results, terminates pod).
- Lessons from uploads: interrupted scp = incomplete tar (silent!). Fix: split into
  700MB parts (train_part_aa/ab) + cat on pod; ALWAYS verify tar entries count after.
  Also: v20_train_pod.sh was missing after interrupted upload — verify ALL bundle files
  landed before launching.

### RunPod agent (opencode MCP) — SET UP
- Added to ~/.config/opencode/opencode.jsonc:
  - mcp.runpod = https://mcp.getrunpod.io/ (hosted API MCP, OAuth sign-in on first use)
  - mcp.runpod-docs = https://docs.runpod.io/mcp (docs, no auth)
- Schema-verified valid. NEEDS: restart opencode + "Sign in with RunPod" OAuth once.
- Alternative: API-key bearer header (RUNPOD_API_KEY in .env) instead of OAuth.
- No node/npx on this Mac — manual config used instead of the npx guided installer.

### Next (v21 plan)
1. User downloads ~100GB more videos (eighth batch, new cities/variety) while v20 trains
2. Gemini reject recovery (quota resets) → +~1,400 crops
3. Extract → label → rebuild (~30k+ crops) → package pod_bundle_v21/ → train on pod
4. Expected +4-7pts over current king — the "big step"

## 2026-08-30 — v20 CORRECTED RESULT (important lesson)
- v20 (788 classes, 90,963 imgs, +5k more crops) trained fine (internal val 90.5%).
- NAIVE compare on v20 holdout: "king 74.5% vs v20 54%" — WRONG. That holdout
  contained tracks the KING had trained on historically (near-duplicates) → inflated.
- NAIVE compare on v19 holdout: "v20 85.3% vs king 56.8%" — WRONG. v20 had trained
  on v19-holdout crops (they re-entered the bigger raw pool) → inflated v20.
- HONEST test = INTERSECTION of both holdouts (597 crops neither model trained on):
  king 57.2%/79.7% vs v20 56.5%/79.4% → STATISTICALLY TIED.
- CONCLUSION: +5k crops did NOT improve accuracy. We're at the data plateau for this
  taxonomy. v20 NOT deployed; king (v19, 56.9%) stays.
- LESSON: comparing models across rebuilds is invalid unless you test on crops held
  out from BOTH. Always build a persistent, never-changing evaluation holdout that
  no future training set may include. (Intersection method = 597 crops, now deleted
  — rebuild from the two tars if needed.)
- NEXT: v21 with ~100GB more data must ALSO fix the eval methodology: freeze ONE
  permanent holdout set and exclude it from ALL future training builds.

## 2026-08-30 — PERMANENT EVAL HOLDOUT created (v21-critical fix)
- storage/dataset/eval_holdout/ = 3,570 crops / 340 classes, FROZEN (never regenerated).
- build_merged.py now EXCLUDES all eval_holdout crops from training (verified 0 leaks).
- Every future training round will be compared on THIS set — comparisons are finally
  valid across rebuilds (the v20 contamination problem is fixed at the source).
- Rebuilt train_v6: 76,673 imgs (was 90,963 with contaminated crops). 788 classes.

## 2026-08-30 night — ENSEMBLE result (frozen eval_holdout, 3,558 crops)
- KING (v19): top-1 74.5% / top-5 87.4%  <- INFLATED (frozen set contains king-memorized crops)
- V20:        top-1 54.0% / top-5 75.4%
- ENSEMBLE (avg softmax king+v20, shared class space): top-1 71.6% / top-5 89.3%
- Verdict: ensemble improves top-5 (+1.9 over inflated king) but lower top-1. NOT a
  clear deploy win on this contaminated set. On the honest 597-crop intersection both
  single models were tied (57.2 vs 56.5) — ensemble likely a small real gain on top-5.
- DECISION: keep v19 king deployed. Ensemble is available as an optional runtime
  feature but not adopted (marginal, adds inference cost on-device).
- v21 needs the frozen holdout (done) + genuinely new car variety to break the plateau.

## 2026-08-31 — honest intersection test (eval_all.py) + release prep
- eval_all.py built: intersection of frozen eval_holdout ∩ v19-holdout = 597 crops
  held out from ALL models. Honest four-way comparison.
- RESULTS (597 crops, n=596 usable):
  king v19 57.2%/79.7% | v20 56.5%/79.4% | v21 54.7%/77.9% | v22 (training)
- VERDICT: king v19 still best on the honest set. v21's exclusion of contaminated
  crops cost ~1.5pts vs v20 (honest now). v22 tests whether +993 recovered crops help.
- RELEASE_PROMOTION.md updated with honest numbers + methodology note (74.5%/85.3%
  58.5% were contaminated-measurement artifacts — never quote them).
- blur_plates: true set in config.yaml (verified). README.md + LICENSE (MIT) written,
  .gitignore extended (pod tars, personal docs). Secrets audit: clean.
- Pod CPU insurance: king-vs-frozen compare run on pod (74.5%, matches local — eval
  path validated for v22).

## 2026-08-31 — v22 FINAL VERDICT + project training phase closed
- v22 (822 classes, +993 recovered crops) trained on 5090, all checkpoints fetched.
- FROZEN-holdout compare (naive, contaminated for king): king 74.5% vs v22 51.8/51.0%.
- HONEST INTERSECTION (597 crops, no model trained on any):
  king v19 57.2%/79.7% | v20 56.5%/79.4% | v21 54.7%/77.9% | v22 53.7%/79.9%
- VERDICT: KING v19 REMAINS DEPLOYED. v22's +993 recovered crops did NOT help top-1
  (recovery data is marginal noise; top-5 ties 79.9%). v21/v22 are honest but below king.
- DECISION: training loop on YouTube-derived data is CLOSED. More of the same = ~0 gain.
  Next real lever = user's OWN real footage from live drives (zero rights issues + real
  distribution), or new-region variety at scale.
- All pods terminated. models/v20,v21,v22 checkpoints saved locally for reference.

## 2026-08-31 — PROJECT PUBLISHED 🎉
- GITHUB: https://github.com/rdo11/mobile_tracker (public, clean 1-commit history,
  no large files, no secrets, no private paths, no personal docs)
- HUGGING FACE MODEL: https://huggingface.co/rdo11/euro-dashcam-vehicle-classifier
  (model.pt + card with honest metrics)
- HUGGING FACE DATASET: https://huggingface.co/datasets/rdo11/euro-dashcam-vehicle-dataset
  (550MB crops tar + CC-BY-NC-4.0 provenance card)
- LINKEDIN PDF: PROJECT_REPORT_2026.pdf (3 pages, honest numbers)
- Privacy fixes: absolute user paths → portable; .git 8.6GB → 280KB via fresh history
- Next: live-drive validation (checklist ready, blur on) → own-footage dataset round
