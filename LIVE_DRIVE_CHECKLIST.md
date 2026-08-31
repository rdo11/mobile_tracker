# LIVE-DRIVE READINESS CHECKLIST (2026-08-29)

## Status: DEPLOYED = v18_tiny224 (new king, beats old by +2.7 top-1 / +8.1 top-5)

## Before the drive (on the Mac, plugged in)
- [ ] Run: `.venv/bin/python main.py`
- [ ] Check log: "Deep classifier loaded: 630 classes" AND "Camera opened"
- [ ] Open http://127.0.0.1:8500 — dashboard shows video + boxes
- [ ] If using the phone as camera: launch the phone app, pair it
- [ ] Confirm config.yaml camera.source points to the phone (not a file)
- [ ] Battery: keep Mac on AC (96W charger) — training drains at 67W otherwise

## During the drive (on the road)
- [ ] Cars: boxes track vehicles, label shows make+model + confidence
- [ ] Plates: ANPR reads + auto-blurs on stream/recording
- [ ] Signs: speed limits detected (log: "Speed limit 30 confirmed")
- [ ] Note any class that is CONSISTENTLY wrong (same car, repeated misses)
- [ ] Note UI crashes / freezes / MPS errors

## After the drive
- [ ] Stop main.py cleanly (Ctrl-C) so the .mp4 finalizes
- [ ] Pull the recording from storage/recordings/
- [ ] Extract crops from the recording (extract_crops.py) — REAL footage beats YouTube
- [ ] DeepSeek-label them → rebuild → next training round (the proven lever)

## If a class is consistently wrong
- Its crops may be mislabeled in dashcam_raw → check that folder
- Add more of that car's footage from new videos
- Rebuild + retrain (no pod needed for data prep)

## Known limits (from confusion analysis)
- Near-identical cars confuse the model ~9.4% (C↔E-Class, Model 3↔Y, 3↔5 Series).
  Decision: keep fine-grained — confidence in the UI shows uncertainty.
- 336px training hurt; 224px is the sweet spot. Bigger archs didn't help.
- Model is MPS-ready (classifier.py), no GPU needed on the Mac.
