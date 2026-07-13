# Early Grooming Detection Using Language and Emotion Analysis

A hybrid deep learning system for the **early detection of online grooming conversations**, combining transformer-based semantic embeddings with fine-grained emotional representations from GoEmotions.

> Diploma Thesis — Babeș-Bolyai University Cluj-Napoca, Faculty of Mathematics and Computer Science
> Author: Teodora-Elena Cătănaș · Supervisor: Lect. PhD. Diana-Laura Borza · 2026

---

## Overview

Online grooming is a gradual, deceptive process that is hard to detect before predatory intent becomes explicit. This project explores whether **fusing linguistic and emotional signals** can help models flag grooming behavior earlier in a conversation, rather than only after the fact.

The system is trained and evaluated on the **PAN12 Sexual Predator Identification** dataset, split into progressive conversation windows (10%–100% of a chat) to simulate realistic early-detection conditions.

## Key Features

- **Data pipeline** that converts PAN12 XML logs into labeled, progress-tagged conversation windows
- **Baseline model**: TF-IDF + Logistic Regression
- **Transformer models**: DistilBERT and RoBERTa (frozen encoders + linear classifier)
- **Emotion features**: 28-dimensional GoEmotions probability vectors
- **Feature fusion architectures** combining semantic embeddings with emotion vectors:
  - BERT + GoEmotions 
  - RoBERTa + GoEmotions 
- **Web application** for analyzing new conversations with the best-performing model
- **Privacy-first design**: local browser storage, end-to-end encrypted prediction requests

## Results Summary

F1-score across conversation progress levels (PAN12 test set):

| Progress | Baseline | GoEmotions | DistilBERT | RoBERTa | DistilBERT+GoE | RoBERTa+GoE |
|---|---|---|---|---|---|---|
| 10%  | 0.666 | 0.010 | 0.719 | 0.746 | 0.724 | 0.738 |
| 40%  | 0.648 | 0.014 | 0.754 | 0.756 | 0.757 | 0.778 |
| 80%  | 0.673 | 0.019 | 0.774 | 0.816 | 0.773 | **0.816** |
| 100% | 0.672 | 0.028 | 0.764 | 0.800 | 0.760 | 0.805 |

The **RoBERTa + GoEmotions fusion model** achieved the strongest early-detection performance, with the largest gains between 20%–60% conversation progress — the region that matters most for timely intervention.

## Architecture

```
Conversation text ──► RoBERTa encoder ──► 768-d embedding ─┐
                                                             ├─► Concatenate ─► Linear classifier ─► Predatory / Safe
Conversation text ──► GoEmotions model ──► 28-d emotion vec ┘
```

### Deployed system

```
Browser client (HTML/JS + SQLite via sql.js/IndexedDB)
        │  ECDH P-256 handshake + AES-256-GCM
        ▼
FastAPI backend  ──►  RoBERTa + GoEmotions inference pipeline
        │
        └── /predict/secure   (encrypted request/response)
```

- Conversations are stored **locally in the browser** and only sent for analysis on explicit user request.
- Requests to `/predict/secure` are end-to-end encrypted using an ECDH-derived session key (HKDF-SHA256 + AES-256-GCM), on top of standard TLS.
- Sessions are short-lived (15-minute TTL); the plaintext prediction endpoint is disabled by default.

## Tech Stack

- **Modeling**: PyTorch, HuggingFace Transformers (DistilBERT, RoBERTa, `SamLowe/roberta-base-go_emotions`), scikit-learn
- **Backend**: FastAPI
- **Frontend**: HTML / CSS / vanilla JavaScript, sql.js (SQLite in-browser), IndexedDB
- **Dataset**: PAN12 Sexual Predator Identification Corpus

