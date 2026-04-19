# Document Forgery Detection Engine — v2
## Track C: Explainable AI for Document Forgery Detection

---

### Quick start

#### Local
```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr        # Ubuntu/Debian
# brew install tesseract                  # Mac

python test_detector.py                   # self-test (synthetic docs)
python app.py                             # REST API on http://localhost:8000
```

#### Kaggle
1. Upload **`detector.py`** as a Kaggle dataset (Add Data → New Dataset)
2. Upload **`kaggle_notebook.ipynb`** as a new notebook
3. In the notebook sidebar, add your dataset so `detector.py` is visible
4. Set `USE_SYNTHETIC = True` to test immediately — or point `IMAGE_PATH` at your document
5. Run all cells; reports are saved to `/kaggle/working/`

---

### API

```bash
python app.py   # starts on http://localhost:8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Upload a document image (multipart or raw bytes) |
| GET  | `/health` | Health check |
| GET  | `/methods` | List detection methods + weights |
| GET  | `/report/{id}` | Retrieve cached report |

**Example:**
```bash
curl -X POST http://localhost:8000/detect -F "file=@document.jpg"
```

**Response:**
```json
{
  "filename": "document.jpg",
  "overall_score": 0.18,
  "verdict": "GENUINE",
  "confidence": 0.73,
  "language": "english",
  "summary": "...",
  "heatmap_b64": "<base64 PNG>",
  "detections": [
    {
      "method": "ela",
      "score": 0.12,
      "explanation": "Error levels within normal range. Found 0 suspicious region(s).",
      "suspicious_regions": []
    }
  ]
}
```

---

### Detection methods

| Method | Weight | What it detects |
|--------|--------|-----------------|
| ELA | 25% | Image editing artifacts via re-compression |
| Noise Analysis | 20% | Splicing via inconsistent Laplacian variance |
| Font Consistency | 15% | Text insertion via character geometry mismatch |
| Copy-Move | 15% | Cloned stamps/signatures via ORB self-matching |
| Layout Anomaly | 10% | Misaligned/inserted text via spacing analysis |
| Metadata | 8% | Missing DPI, format mismatch, multiple JPEG qtables |
| OCR Confidence | 7% | Degraded/tampered text quality |

### Verdicts

| Score | Verdict |
|-------|---------|
| 0.00–0.25 | **GENUINE** |
| 0.25–0.50 | **SUSPICIOUS** |
| 0.50–0.75 | **LIKELY_FORGED** |
| 0.75–1.00 | **FORGED** |

---

### v2 calibration — false positive fixes

The original v1 thresholds were calibrated for synthetic clean images and
produced false positives on real-world document photos. v2 fixes:

| Method | Problem | Fix |
|--------|---------|-----|
| Noise | Real docs score CV=2.5+ naturally | Score offset: silent below CV=2.0 |
| Font | Headings + body text give raw=0.5-0.9 | Score offset: silent below raw=0.6 |
| OCR | Real scans give Tesseract 30-55% | Baseline lowered 70% → 45% |
| ELA | All JPEGs have compression artifacts | Baseline offset: mean_ela=20 |

---

### Files

| File | Purpose |
|------|---------|
| `detector.py` | Core detection engine |
| `app.py` | REST API server |
| `test_detector.py` | CLI self-test with synthetic documents |
| `kaggle_notebook.ipynb` | Kaggle notebook — upload and run directly |
| `requirements.txt` | Python dependencies |

---

### Judging criteria

| Criterion | Weight | How addressed |
|-----------|--------|---------------|
| Detection Accuracy | 30% | 7-method ensemble; v2 calibration reduces false positives on real docs |
| Explainability | 25% | Per-method scores, natural-language explanations, colour-coded heatmap |
| Language Robustness | 15% | Tesseract multi-language; non-ASCII ratio flags regional scripts |
| UI/UX | 15% | Dark-theme Kaggle notebook; REST API; JSON reports |
| Documentation | 15% | This README + inline docstrings + notebook markdown |
