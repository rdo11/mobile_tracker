#!/usr/bin/env python3
"""jupyter_deploy.py — deploy + launch v13/v14 training on the pod's Jupyter,
bypassing SSH entirely (auth is broken on this pod).

Flow:
  1. Login to Jupyter with JUPYTER_PASSWORD (from RunPod env via API)
  2. Chunked-upload bundle files into /root/
  3. Open a Jupyter terminal and run extract+deps+train detached
  4. Verify training process alive
"""
import base64
import json
import os
import re
import sys
import time

import requests

POD_ID = None
BASE = None


def api_get_pods():
    key = [l.split("=", 1)[1].strip() for l in open(".env")
           if l.startswith("RUNPOD_API_KEY=")][0]
    ctx = ssl_create_ctx()
    req = urllib_request_req("https://api.runpod.io/v2/pods", key, ctx)
    with urllib_open(req) as r:
        d = json.loads(r.read().decode())
    pods = d.get("pods") if isinstance(d, dict) else d
    return [p for p in pods if isinstance(p, dict) and p.get("status") == "RUNNING"][0]


def ssl_create_ctx():
    import ssl, certifi
    return ssl.create_default_context(cafile=certifi.where())


def urllib_open(req):
    return urlopen_with_ctx(req)


def urlopen_with_ctx(req):
    import urllib.request
    return urllib.request.urlopen(req, timeout=60)


def urllib_request_req(url, key, ctx_unused=None):
    import urllib.request
    return urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0"})


def main():
    global POD_ID, BASE
    me = api_get_pods()
    POD_ID = me["id"]
    BASE = f"https://{POD_ID}-8888.proxy.runpod.net"
    pwd = (me.get("env") or {}).get("JUPYTER_PASSWORD", "")
    print(f"[1] pod {POD_ID} | jupyter proxy ready")

    s = requests.Session()
    verify = True
    lr = s.get(BASE + "/login", timeout=60)
    m = re.search(r'name="_xsrf"\s+value="([^"]+)"', lr.text)
    xsrf = m.group(1) if m else ""
    s.post(BASE + "/login?next=%2Ftree",
           data={"_xsrf": xsrf, "password": pwd}, timeout=60)
    print("[2] logged into jupyter")

    hdr = {"X-XSRFToken": xsrf, "X-CSRFToken": xsrf}

    # ---- files to upload (path on mac -> dest under /root/)
    bundle_dir = os.path.expanduser("~/Projects/mobile_tracker/pod_bundle")
    files = ["train_v6.tar", "dashcam_val.tar", "r50_dashcam.pt",
             "train_classifier.py", "compare_ckpts.py"]

    CHUNK = 4 * 1024 * 1024
    for fname in files:
        src = os.path.join(bundle_dir, fname)
        if not os.path.exists(src):
            print(f"  !! missing locally, skip: {fname}")
            continue
        size = os.path.getsize(src)
        dest = "/root/" + fname
        # check if already complete on pod
        st = s.get(BASE + "/api/contents" + dest.replace("/root/", ""), timeout=60)
        if st.status_code == 200:
            remote_size = (st.json().get("size") or 0)
            if remote_size == size:
                print(f"  {fname}: already present ({size}) — skip")
                continue
        n_chunks = (size + CHUNK - 1) // CHUNK
        t0 = time.time()
        print(f"  UPLOADING {fname} ({size/1e6:.0f} MB, {n_chunks} chunks)...",
              flush=True)
        sent = 0
        with open(src, "rb") as fh:
            for ci in range(n_chunks):
                chunk = fh.read(CHUNK)
                body = {
                    "type": "file", "format": "base64",
                    "chunk_id": ci + 1, "num_chunks": n_chunks,
                    "name": fname, "path": dest,
                    "content": base64.b64encode(chunk).decode(),
                }
                if ci == 0:
                    body["size"] = size
                    body["mimetype"] = "application/octet-stream"
                tt0 = time.time()
                rr = s.put(BASE + "/api/contents" + dest, json=body,
                           headers=hdr, timeout=600)
                dt = time.time() - tt0
                sent += len(chunk)
                mbps = len(chunk) / 1e6 / max(0.001, dt)
                eta = (n_chunks - ci - 1) * max(dt, 0.001) / 60
                if rr.status_code not in (200, 201):
                    print(f"  UPLOAD FAIL {fname} chunk {ci+1}/{n_chunks}: "
                          f"{rr.status_code} {rr.text[:120]}", flush=True)
                    sys.exit(1)
                print(f"    chunk {ci+1}/{n_chunks} | {mbps:.1f} MB/s | "
                      f"ETA {eta:.1f} min", flush=True)
        print(f"  DONE {fname} in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- terminal: run everything detached
    print("[3] creating jupyter terminal...")
    tr = s.post(BASE + "/api/terminals", json={}, headers=hdr, timeout=60)
    tname = tr.json().get("name")
    print("  terminal:", tname)

    import websockets
    import asyncio
    import ssl as ssl_mod
    ctx = ssl_mod.create_default_context(cafile=__import__("certifi").where())

    CMD = (
        "cd /root && mkdir -p mt/storage/dataset mt/models && "
        "tar -xf train_v6.tar -C mt/storage/dataset/ && "
        "tar -xf dashcam_val.tar -C mt/storage/dataset/ && "
        "mv r50_dashcam.pt train_classifier.py compare_ckpts.py mt/ && "
        "pip install --break-system-packages -q opencv-python-headless pillow tqdm; "
        "cd mt && rm -f train.log && setsid nohup python3 -u train_classifier.py "
        "--data storage/dataset/train_v6 --arch convnext_base --epochs 12 "
        "--batch 32 --size 336 --lr 0.0004 --init models/r50_dashcam.pt "
        "--out models/convnext_v14_base336.pt > /root/train.log 2>&1 < /dev/null & "
        "sleep 20; echo MARKER_PROCS=$(pgrep -fc train_classifier.py); "
        "tail -c 200 /root/train.log"
    )

    async def drive():
        ws_url = (BASE + "/terminals/websocket/" + tname).replace("https://", "wss://")
        cookies = "; ".join(f"{c.name}={c.value}" for c in s.cookies)
        async with websockets.connect(
                ws_url, ssl=ctx, additional_headers={"Cookie": cookies},
                max_size=None) as ws:
            await asyncio.sleep(2)
            await ws.send(CMD + "\n")
            out = ""
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=25)
                    out += msg if isinstance(msg, str) else msg.decode(errors="ignore")
                    if "MARKER_PROCS=" in out:
                        break
            except asyncio.TimeoutError:
                pass
        return out[-600:]

    out = asyncio.run(drive())
    print(out)
    print("DONE — check /root/train.log on pod next")


if __name__ == "__main__":
    main()
