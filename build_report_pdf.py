#!/usr/bin/env python3
"""build_report_pdf.py — generate the LinkedIn-ready project report PDF."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, PageBreak)

OUT = os.path.join(os.path.dirname(__file__), "PROJECT_REPORT_2026.pdf")

NAVY = HexColor("#0F2A43")
ACCENT = HexColor("#1F6FEB")
GREY = HexColor("#5B6572")
LIGHT = HexColor("#EEF3F9")

ss = getSampleStyleSheet()
title = ParagraphStyle("title", parent=ss["Title"], fontSize=26, textColor=NAVY,
                       spaceAfter=6)
subtitle = ParagraphStyle("subtitle", parent=ss["Normal"], fontSize=14,
                          textColor=GREY, alignment=TA_CENTER, spaceAfter=18)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, textColor=NAVY,
                    spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=4)
body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10.5, leading=15,
                      textColor=HexColor("#222222"))
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4,
                        spaceAfter=2)
small = ParagraphStyle("small", parent=body, fontSize=8.5, textColor=GREY)
cell = ParagraphStyle("cell", parent=body, fontSize=9.5, leading=12)

def P(text, style=body):
    return Paragraph(text, style)

def B(text):
    return Paragraph(f"• {text}", bullet)

def make_table(rows, widths, header_color=NAVY):
    data = [[Paragraph(f"<b>{c}</b>", ParagraphStyle('h', parent=cell,
            textColor=white)) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(str(c), cell) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#C9D4E0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t

story = []
story.append(P("Privacy-First Dashcam Vehicle Recognition", title))
story.append(P("A self-built computer-vision system that reads traffic — make, model, "
               "generation, plates, and speed limits — fully offline on a MacBook", subtitle))
story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=16))

story.append(P("Executive Summary", h1))
story.append(P("I built, from scratch, a privacy-first dashcam system that understands the "
               "street: it detects and tracks every vehicle in real time, identifies make + "
               "model + generation offline on Apple Silicon, reads license plates (including "
               "DK/DE electric and historic-plate detection), recognizes speed-limit signs, "
               "and blurs every plate for GDPR — all without any cloud API at inference. "
               "The system also auto-collects its own training data from every drive and "
               "was used to train a 646-class European vehicle classifier through seven "
               "verified training iterations."))

story.append(P("Why it matters", h2))
story.append(B("Dashcams record, but don't understand. This one reads traffic — and does it "
               "privacy-first (plates auto-blurred on stream and recordings)."))
story.append(B("The whole pipeline — data collection, labeling, training, evaluation, "
               "deployment — runs on a single MacBook plus rented GPU hours."))
story.append(B("Built-in honest evaluation methodology: a permanent frozen holdout that no "
               "training build may ever touch."))

story.append(Spacer(1, 6))
story.append(P("System Capabilities", h1))
story.append(make_table([
    ["Capability", "Implementation", "Detail"],
    ["Vehicle detection + tracking", "YOLOv8 + ByteTrack", "~25 ms/frame on Apple Silicon (MPS)"],
    ["Make/model/generation (offline)", "ConvNeXt-Tiny classifier", "646 European classes, from scratch"],
    ["License plates", "EasyOCR + heuristics", "DK/DE plates, electric (E) / historic (H) tags"],
    ["Speed-limit signs", "GTSRB classifier", "EU signs, two-read confirmation"],
    ["Privacy", "GDPR auto-blur", "plates blurred on stream + recordings, on by default"],
    ["Live dashboard", "FastAPI + WebSockets", "localhost:8500, near-fullscreen driving UI"],
    ["Dataset growth", "Auto crop harvesting", "every drive labels new crops for retraining"],
], [5.2 * cm, 5.0 * cm, 6.6 * cm]))

story.append(PageBreak())
story.append(P("Model & Results", h1))
story.append(P("The deployed classifier is a ConvNeXt-Tiny (28M parameters) trained from "
               "scratch on 646 European vehicle classes. It runs fully offline via MPS "
               "inference with test-time augmentation."))
story.append(P("Honest evaluation", h2))
story.append(P("Every number below was measured on an intersection of permanent holdout "
               "sets: 597 real dashcam crops that <b>no model ever trained on</b>. This "
               "methodology was built deliberately — naive holdout comparisons were found "
               "to be contaminated because models had seen parts of each other's eval sets."))
story.append(make_table([
    ["Model", "Top-1", "Top-5", "Verdict"],
    ["v19 (deployed king)", "57.2%", "79.7%", "Best honest model"],
    ["v20 (+5k crops)", "56.5%", "79.4%", "Tied — data plateau"],
    ["v21 (frozen-holdout)", "54.7%", "77.9%", "Honest baseline"],
    ["v22 (+1k recovered)", "53.7%", "79.9%", "Recovery = noise"],
], [4.6 * cm, 2.8 * cm, 2.8 * cm, 6.6 * cm]))

story.append(P("Engineering journey — what actually moved the needle", h2))
story.append(B("<b>30x oversampling was a trap:</b> the model memorized duplicate crops "
               "(97% internal validation → 44% on real holdout). 5x + from-scratch fixed it."))
story.append(B("<b>Data diversity beats architecture:</b> tiny/base/large and 224px/336px "
               "were all tested; tiny @ 224px won every time."))
story.append(B("<b>Evaluation methodology matters as much as the model:</b> a track-level "
               "leak was found and fixed; a permanent frozen holdout now guarantees honest "
               "comparisons forever."))
story.append(B("<b>More of the same data has a hard plateau</b> — proven with three "
               "consecutive experiments. The next real gain is new-domain footage "
               "(own recordings), not more similar YouTube data."))

story.append(Spacer(1, 8))
story.append(P("Training scale", h2))
story.append(make_table([
    ["Metric", "Value"],
    ["Dataset (raw labeled crops)", "~21,300 crops"],
    ["Training set (5x oversample)", "~80,000 images"],
    ["Classes", "646 deployed / 822 total"],
    ["Training rounds (verified)", "7 (v15→v22)"],
    ["Rented GPU", "RTX 4090 / 5090 (RunPod)"],
    ["Local hardware", "MacBook M5, 24 GB unified"],
], [6.5 * cm, 10.3 * cm]))

story.append(PageBreak())
story.append(P("Pipeline & Tooling", h1))
story.append(B("extract_crops — mines unique crops per car per distance bucket from video."))
story.append(B("label_crops — batch labeling via DeepSeek vision (paid, no quota)."))
story.append(B("build_merged — dataset builder: canonicalization, group-aware holdout, "
               "oversampling, leak-free frozen eval set."))
story.append(B("train_classifier — from-scratch training (MPS local or CUDA pod)."))
story.append(B("compare_ckpts / eval_all — honest holdout + intersection evaluation."))
story.append(B("main.py + frontend — live FastAPI dashboard for driving."))
story.append(B("Grok/Gemini — second-opinion reject recovery for hard crops."))

story.append(P("Privacy & Ethics", h1))
story.append(B("Plates are blurred on stream and recordings (GDPR), on by default."))
story.append(B("Plates used for in-memory vehicle re-ID only — never streamed or stored."))
story.append(B("Training data derived from publicly available dashcam footage; the public "
               "dataset card documents provenance under a research license."))

story.append(P("Tech Stack", h1))
story.append(P("Python 3.14 · PyTorch (MPS + CUDA) · ConvNeXt · YOLOv8 · ByteTrack · "
               "OpenCV · FastAPI + WebSockets · EasyOCR · GTSRB · yt-dlp · RunPod · "
               "DeepSeek/Gemini/Grok vision APIs", body))

story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=0.8, color=ACCENT, spaceAfter=8))
story.append(P("Built from scratch: video mining → labeling → dataset engineering → "
               "training → honest evaluation → deployment → live dashboard. "
               "All on a MacBook.", small))
story.append(P("Report generated 2026-08-31 · Full code: github (see repo) · "
               "Model + dataset: Hugging Face (see cards)", small))

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                        title="Privacy-First Dashcam Vehicle Recognition",
                        author="Mobile Tracker Project")
doc.build(story)
print(f"PDF written: {OUT} ({os.path.getsize(OUT)} bytes)")
