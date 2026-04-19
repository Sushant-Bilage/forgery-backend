"""
Document Forgery Detection Engine  —  v2 (recalibrated for real-world docs)
============================================================================
Track C: Explainable AI for Document Forgery Detection

Detection methods:
  1. Error Level Analysis (ELA)       — image editing artifacts
  2. Noise Pattern Analysis           — inconsistent noise floors
  3. Font Consistency Analysis        — mixed/inserted fonts
  4. Copy-Move Detection              — cloned regions
  5. Layout Anomaly Detection         — spacing / alignment issues
  6. Metadata Forensics               — file-level tampering signals
  7. OCR Confidence Analysis          — degraded / tampered text

Each method returns score 0.0 (genuine) → 1.0 (forged) + explanation.
Final score = confidence-weighted ensemble of all methods.

v2 calibration changes (fixes false positives on real photos):
  ELA      baseline offset  mean=20   (real JPEGs naturally compress noisily)
  Noise    baseline offset  CV=2.0    (text+margin+image regions differ naturally)
  Font     baseline offset  raw=0.6   (headings vs body text are legitimately different)
  OCR      baseline lowered 70 → 45   (real scans give Tesseract 30-55% confidence)
"""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import pytesseract
import io
import os
import hashlib
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import base64

warnings.filterwarnings("ignore")


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    method: str
    score: float            # 0.0 = genuine, 1.0 = forged
    confidence: float       # reliability of this method
    explanation: str
    suspicious_regions: List[Dict] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


@dataclass
class ForensicReport:
    filename: str
    overall_score: float    # final forgery probability 0–1
    verdict: str            # GENUINE / SUSPICIOUS / LIKELY_FORGED / FORGED
    confidence: float
    detections: List[DetectionResult] = field(default_factory=list)
    summary: str = ""
    heatmap_b64: str = ""   # base64-encoded PNG
    ocr_text: str = ""
    language_detected: str = "unknown"


# ─── Method weights & verdict thresholds ──────────────────────────────────────

METHOD_WEIGHTS = {
    "ela":              0.40,   # 🔥 increased (important)
    "noise_analysis":   0.15,
    "font_consistency": 0.10,
    "copy_move":        0.10,
    "layout_anomaly":   0.10,
    "metadata":         0.08,
    "ocr_confidence":   0.07,
}

VERDICT_THRESHOLDS = {
    "GENUINE":       (0.00, 0.25),
    "SUSPICIOUS":    (0.25, 0.50),
    "LIKELY_FORGED": (0.50, 0.75),
    "FORGED":        (0.75, 1.01),
}


# ─── 1. Error Level Analysis ──────────────────────────────────────────────────

def ela_analysis(img_array: np.ndarray, quality: int = 90) -> DetectionResult:
    """
    Re-save image at known JPEG quality; compare pixel-level differences.
    Edited regions retain higher error levels than untouched regions.

    v2: score offset so normal JPEG-compression artifacts (mean_ela ≈ 15-25)
    produce ~0 score. Only regions well above that baseline are flagged.
    """
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).copy()

        ela_img  = ImageChops.difference(img_pil.convert("RGB"), recompressed)
        extrema  = ela_img.getextrema()
        max_diff = max(ex[1] for ex in extrema) or 1
        ela_img  = ImageEnhance.Brightness(ela_img).enhance((255.0 / max_diff) * 10)

        ela_gray = cv2.cvtColor(np.array(ela_img), cv2.COLOR_RGB2GRAY)
        mean_ela = float(np.mean(ela_gray))
        std_ela  = float(np.std(ela_gray))

        # v2: offset baseline — normal compression gives mean≈15-25, std≈20-35
        score = min(1.0, max(0.0,
            (mean_ela - 20.0) / 110.0 * 0.6 +
            (std_ela  - 30.0) / 100.0 * 0.4
        ))

        # Only flag large high-ELA regions (reduces JPEG noise false alarms)
        threshold  = np.percentile(ela_gray, 95)
        suspicious = []
        if threshold > 50:
            contours, _ = cv2.findContours(
                (ela_gray > threshold).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) > 1500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    suspicious.append({"x": int(x), "y": int(y),
                                       "w": int(w), "h": int(h),
                                       "severity": "high"})

        explanation = (
            f"ELA mean={mean_ela:.1f}, std={std_ela:.1f}. "
            + ("High error levels indicate image manipulation. "
               if score > 0.5 else "Error levels within normal range. ")
            + f"Found {len(suspicious)} suspicious region(s)."
        )

        return DetectionResult(
            method="ela", score=round(score, 3), confidence=0.85,
            explanation=explanation, suspicious_regions=suspicious,
            details={"mean_ela": mean_ela, "std_ela": std_ela,
                     "ela_array": ela_gray}
        )

    except Exception as e:
        return DetectionResult("ela", 0.0, 0.0, f"ELA failed: {e}")


# ─── 2. Noise Pattern Analysis ────────────────────────────────────────────────

def noise_analysis(img_array: np.ndarray) -> DetectionResult:
    """
    Measures Laplacian variance across image blocks.
    Spliced images show very inconsistent noise floors between regions.

    v2: offset baseline so natural document variance (CV ≤ 2.0) scores ~0.
    Real documents have text, white margins, stamps — naturally high CV.
    """
    try:
        gray       = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(float)
        h, w       = gray.shape
        noise_map  = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        block_size = max(32, min(h, w) // 16)

        variances = [
            np.var(noise_map[y:y+block_size, x:x+block_size])
            for y in range(0, h - block_size, block_size)
            for x in range(0, w - block_size, block_size)
        ]

        if not variances:
            return DetectionResult("noise_analysis", 0.0, 0.3,
                                   "Insufficient image size for noise analysis.")

        variances  = np.array(variances)
        noise_mean = float(np.mean(variances))
        noise_std  = float(np.std(variances))
        cv_noise   = noise_std / (noise_mean + 1e-6)

        # v2: offset — natural CV for real docs is 1.5-3.0; only flag above 2.0
        score = min(1.0, max(0.0, (cv_noise - 2.0) / 3.0))

        suspicious = []
        if cv_noise > 3.0:
            threshold = np.percentile(variances, 92)
            idx = 0
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    if idx < len(variances) and variances[idx] > threshold:
                        suspicious.append({"x": x, "y": y,
                                           "w": block_size, "h": block_size,
                                           "severity": "medium"})
                    idx += 1

        explanation = (
            f"Noise CV={cv_noise:.3f}. "
            + ("Inconsistent noise patterns suggest image splicing."
               if score > 0.5
               else "Noise patterns within normal range for a real document.")
        )

        return DetectionResult(
            method="noise_analysis", score=round(score, 3), confidence=0.80,
            explanation=explanation, suspicious_regions=suspicious[:10],
            details={"noise_cv": cv_noise, "noise_mean": noise_mean}
        )

    except Exception as e:
        return DetectionResult("noise_analysis", 0.0, 0.0,
                               f"Noise analysis failed: {e}")


# ─── 3. Font Consistency Analysis ─────────────────────────────────────────────

def font_consistency_analysis(img_array: np.ndarray) -> DetectionResult:
    """
    Detects text insertion by measuring character geometry inconsistencies.
    Uses connected-component statistics (height, aspect ratio).

    v2: offset baseline so normal multi-size typography (heading + body)
    scores ~0. Only truly alien pasted text drives the score up.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

        if num_labels < 10:
            return DetectionResult("font_consistency", 0.0, 0.4,
                                   "Insufficient text detected for font analysis.")

        h_img, w_img = gray.shape
        char_stats = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            aspect = w / (h + 1e-6)
            if (3 < h < h_img * 0.15 and 2 < w < w_img * 0.1
                    and 0.1 < aspect < 5.0 and area > 10):
                char_stats.append({"height": h, "width": w, "area": area,
                                   "aspect": aspect, "x": x, "y": y})

        if len(char_stats) < 5:
            return DetectionResult("font_consistency", 0.0, 0.4,
                                   "Too few text elements detected.")

        heights = [c["height"] for c in char_stats]
        aspects = [c["aspect"]  for c in char_stats]

        cv_height = np.std(heights) / (np.mean(heights) + 1e-6)
        cv_aspect = np.std(aspects) / (np.mean(aspects) + 1e-6)
        raw       = cv_height * 0.6 + cv_aspect * 0.4

        # v2: real docs naturally have raw=0.5-0.9; only flag above 0.6
        score = min(1.0, max(0.0, (raw - 0.6) / 0.8))

        mean_h = np.mean(heights)
        std_h  = np.std(heights)
        suspicious = []
        for c in char_stats:
            z = abs(c["height"] - mean_h) / (std_h + 1e-6)
            if z > 3.5:
                suspicious.append({"x": c["x"], "y": c["y"],
                                   "w": c["width"], "h": c["height"],
                                   "severity": "high" if z > 4.5 else "medium"})

        explanation = (
            f"Analyzed {len(char_stats)} text elements. "
            f"Height CV={cv_height:.3f}, Aspect CV={cv_aspect:.3f}. "
            + ("Font size/style inconsistencies detected — possible text insertion."
               if score > 0.4
               else "Font variation within normal range for a real document.")
        )

        return DetectionResult(
            method="font_consistency", score=round(score, 3), confidence=0.75,
            explanation=explanation, suspicious_regions=suspicious[:8],
            details={"n_chars": len(char_stats), "cv_height": cv_height,
                     "mean_height": mean_h}
        )

    except Exception as e:
        return DetectionResult("font_consistency", 0.0, 0.0,
                               f"Font analysis failed: {e}")


# ─── 4. Copy-Move Detection ───────────────────────────────────────────────────

def copy_move_detection(img_array: np.ndarray) -> DetectionResult:
    """
    Detects cloned regions via ORB keypoint self-matching.
    Forged docs often copy-paste stamps, signatures, or text blocks.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        orb  = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 20:
            return DetectionResult("copy_move", 0.0, 0.3,
                                   "Insufficient keypoints for copy-move detection.")

        bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=3)

        suspicious_pairs = []
        for group in matches:
            if len(group) < 2:
                continue
            for m in group[1:]:
                if m.distance < 15:
                    pt1 = keypoints[m.queryIdx].pt
                    pt2 = keypoints[m.trainIdx].pt
                    if np.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1]) > 80:
                        suspicious_pairs.append((pt1, pt2, m.distance))

        n_susp  = len(suspicious_pairs)
        n_kp    = len(keypoints)
        score   = min(1.0, (n_susp / (n_kp + 1e-6)) * 8.0)

        regions = []
        for pt1, pt2, dist in suspicious_pairs[:15]:
            regions.append({
                "x": int(min(pt1[0], pt2[0])),
                "y": int(min(pt1[1], pt2[1])),
                "w": int(abs(pt1[0]-pt2[0]) + 20),
                "h": int(abs(pt1[1]-pt2[1]) + 20),
                "severity": "high" if dist < 15 else "medium"
            })

        explanation = (
            f"Found {n_susp} potential copy-move pairs among {n_kp} keypoints. "
            + ("Cloned regions detected — possible stamp/signature duplication."
               if score > 0.4
               else "No significant copy-move artifacts found.")
        )

        return DetectionResult(
            method="copy_move", score=round(score, 3), confidence=0.70,
            explanation=explanation, suspicious_regions=regions,
            details={"n_keypoints": n_kp, "n_suspicious_pairs": n_susp}
        )

    except Exception as e:
        return DetectionResult("copy_move", 0.0, 0.0,
                               f"Copy-move detection failed: {e}")


# ─── 5. Layout Anomaly Detection ──────────────────────────────────────────────

def layout_anomaly_detection(img_array: np.ndarray) -> DetectionResult:
    """
    Detects misaligned text, inconsistent margins and irregular line spacing
    that are characteristic of text-insertion forgeries.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
        morph = cv2.morphologyEx(
            cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.MORPH_CLOSE, kernel_h
        )

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 3:
            return DetectionResult("layout_anomaly", 0.0, 0.3,
                                   "Insufficient text lines for layout analysis.")

        lines = [{"x": x, "y": y, "w": bw, "h": bh}
                 for cnt in contours
                 for x, y, bw, bh in [cv2.boundingRect(cnt)]
                 if bw > w * 0.1 and bh < h * 0.05]

        if len(lines) < 3:
            return DetectionResult("layout_anomaly", 0.0, 0.4,
                                   "Too few text lines detected.")

        lines.sort(key=lambda l: l["y"])

        left_margins = [l["x"] for l in lines]
        cv_margin    = np.std(left_margins) / (np.mean(left_margins) + 1e-6)

        spacings   = np.diff([l["y"] for l in lines])
        cv_spacing = (np.std(spacings) / (np.mean(spacings) + 1e-6)
                      if len(spacings) > 1 else 0.0)

        heights  = [l["h"] for l in lines]
        cv_lines = np.std(heights) / (np.mean(heights) + 1e-6)

        score = min(1.0, cv_margin * 0.35 + cv_spacing * 0.40 + cv_lines * 0.25)

        suspicious = []
        if len(spacings) > 1:
            mean_sp, std_sp = np.mean(spacings), np.std(spacings)
            for i, sp in enumerate(spacings):
                if abs(sp - mean_sp) > 2 * std_sp:
                    suspicious.append({"x": lines[i]["x"], "y": lines[i]["y"],
                                       "w": lines[i]["w"], "h": int(sp),
                                       "severity": "medium"})

        explanation = (
            f"Analyzed {len(lines)} text lines. "
            f"Margin CV={cv_margin:.3f}, Spacing CV={cv_spacing:.3f}. "
            + ("Irregular layout — text may have been inserted or moved."
               if score > 0.4 else "Layout appears consistent.")
        )

        return DetectionResult(
            method="layout_anomaly", score=round(score, 3), confidence=0.65,
            explanation=explanation, suspicious_regions=suspicious,
            details={"n_lines": len(lines), "cv_margin": cv_margin,
                     "cv_spacing": cv_spacing}
        )

    except Exception as e:
        return DetectionResult("layout_anomaly", 0.0, 0.0,
                               f"Layout analysis failed: {e}")


# ─── 6. Metadata Forensics ────────────────────────────────────────────────────

def metadata_analysis(file_path: str) -> DetectionResult:
    """
    Checks file metadata: missing DPI, mode anomalies,
    JPEG header in non-JPEG file, multiple quantisation tables.
    """
    try:
        score    = 0.0
        findings = []

        size = os.path.getsize(file_path)
        ext  = Path(file_path).suffix.lower()
        img  = Image.open(file_path)
        info = img.info or {}

        if "dpi" not in info:
            score += 0.15
            findings.append("Missing DPI information (common in edited images)")

        with open(file_path, "rb") as f:
            raw = f.read()

        if ext in [".png", ".bmp"] and raw[:3] == b"\xff\xd8\xff":
            score += 0.30
            findings.append("JPEG header in non-JPEG file — possible format manipulation")

        if img.mode not in ["RGB", "RGBA", "L", "CMYK", "P"]:
            score += 0.10
            findings.append(f"Unusual image mode: {img.mode}")

        if ext in [".jpg", ".jpeg"] and hasattr(img, "quantization"):
            if len(img.quantization) > 2:
                score += 0.20
                findings.append("Multiple quantisation tables — possible double JPEG compression")

        score = min(1.0, score)

        explanation = (
            f"File: {Path(file_path).name}, Size: {size // 1024}KB. "
            + (" | ".join(findings) if findings else "No metadata anomalies detected.")
        )

        return DetectionResult(
            method="metadata", score=round(score, 3), confidence=0.60,
            explanation=explanation,
            details={"size_kb": size // 1024, "has_dpi": "dpi" in info,
                     "findings": findings}
        )

    except Exception as e:
        return DetectionResult("metadata", 0.0, 0.0,
                               f"Metadata analysis failed: {e}")


# ─── 7. OCR Confidence Analysis ───────────────────────────────────────────────

def ocr_confidence_analysis(img_array: np.ndarray) -> Tuple[DetectionResult, str, str]:
    """
    Runs Tesseract OCR; tampered/pasted text regions tend to have lower
    word-level confidence than surrounding genuine text.

    v2: baseline lowered from 70% → 45% because real-world document photos
    (camera, scanner) routinely score 30-55% mean confidence on Tesseract.
    Per-word penalty weight reduced 0.30 → 0.15.
    """
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

        data = pytesseract.image_to_data(
            img_pil, output_type=pytesseract.Output.DICT, config="--psm 6"
        )

        confs = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) != -1]
        text  = " ".join(t for t in data["text"] if t.strip())

        if not confs:
            return (DetectionResult("ocr_confidence", 0.0, 0.3,
                                    "No text detected by OCR."),
                    text, "unknown")

        mean_conf = float(np.mean(confs))
        pct_low   = sum(1 for c in confs if c < 40) / len(confs)

        # v2: flag only genuinely poor quality (mean < 45)
        score = max(0.0, min(1.0, (45 - mean_conf) / 45))
        score = min(1.0, score + pct_low * 0.15)

        language  = "english"
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if text and non_ascii > len(text) * 0.2:
            language = "non-english/regional"

        suspicious = []
        for i, conf in enumerate(data["conf"]):
            if str(conf).isdigit() and 0 < int(conf) < 35:
                xd, yd = data["left"][i], data["top"][i]
                wd, hd = data["width"][i], data["height"][i]
                if wd > 5 and hd > 5:
                    suspicious.append({"x": xd, "y": yd, "w": wd, "h": hd,
                                       "severity": "medium"})

        explanation = (
            f"OCR mean confidence={mean_conf:.1f}%, "
            f"{pct_low*100:.1f}% words below 40% confidence. "
            + ("Low OCR confidence suggests degraded or tampered text. "
               if score > 0.4 else "Text quality is acceptable. ")
            + f"Language: {language}."
        )

        return (DetectionResult(
            method="ocr_confidence", score=round(score, 3), confidence=0.70,
            explanation=explanation, suspicious_regions=suspicious[:10],
            details={"mean_conf": mean_conf, "n_words": len(confs),
                     "pct_low_conf": pct_low}
        ), text, language)

    except Exception as e:
        return (DetectionResult("ocr_confidence", 0.0, 0.0,
                                f"OCR failed: {e}"),
                "", "unknown")


# ─── Heatmap Generator ────────────────────────────────────────────────────────

def generate_heatmap(img_array: np.ndarray,
                     detections: List[DetectionResult]) -> str:
    """
    Overlays colour-coded bounding boxes on suspicious regions and blends a
    JET heat map. Returns a base64-encoded PNG string.
    Red=high, Orange=medium, Yellow=low suspicion.
    """
    try:
        overlay    = img_array.copy()
        h, w       = overlay.shape[:2]
        weight_map = np.zeros((h, w), dtype=float)
        colors     = {"high": (0,0,255), "medium": (0,165,255), "low": (0,255,255)}

        for det in detections:
            if det.score < 0.2:
                continue
            wt = det.score * METHOD_WEIGHTS.get(det.method, 0.1)
            for region in det.suspicious_regions:
                rx = max(0, min(int(region.get("x", 0)), w - 1))
                ry = max(0, min(int(region.get("y", 0)), h - 1))
                rw = max(1, min(int(region.get("w", 10)), w - rx))
                rh = max(1, min(int(region.get("h", 10)), h - ry))
                weight_map[ry:ry+rh, rx:rx+rw] += wt
                cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh),
                               colors.get(region.get("severity", "medium"),
                                          colors["medium"]), 2)

        if weight_map.max() > 0:
            norm = (weight_map / weight_map.max() * 255).astype(np.uint8)
            heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            mask = (weight_map > 0.1).astype(np.uint8)
            overlay = cv2.addWeighted(overlay, 0.7, heat * mask[:,:,None], 0.3, 0)

        _, buf = cv2.imencode(".png", overlay)
        return base64.b64encode(buf).decode("utf-8")

    except Exception:
        return ""


# ─── Main Detector ────────────────────────────────────────────────────────────

class DocumentForgeryDetector:
    """
    Main detection engine.

    Usage:
        detector = DocumentForgeryDetector()
        report   = detector.detect("document.jpg")
        result   = report_to_dict(report)   # JSON-ready dict
    """

    def __init__(self):
        self.weights = METHOD_WEIGHTS

    def load_image(self, file_path: str) -> np.ndarray:
        img = cv2.imread(file_path)
        if img is None:
            pil = Image.open(file_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        if img is None:
            raise ValueError(f"Cannot load image: {file_path}")
        return img

    def detect(self, file_path: str) -> ForensicReport:
        """Run full 7-method pipeline. Returns ForensicReport."""
        filename = Path(file_path).name
        img      = self.load_image(file_path)

        h, w = img.shape[:2]
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            img   = cv2.resize(img, (int(w*scale), int(h*scale)))

        detections = []

        print("  [1/7] Running ELA analysis...")
        detections.append(ela_analysis(img))

        print("  [2/7] Running noise analysis...")
        detections.append(noise_analysis(img))

        print("  [3/7] Running font consistency analysis...")
        detections.append(font_consistency_analysis(img))

        print("  [4/7] Running copy-move detection...")
        detections.append(copy_move_detection(img))

        print("  [5/7] Running layout anomaly detection...")
        detections.append(layout_anomaly_detection(img))

        print("  [6/7] Running metadata analysis...")
        detections.append(metadata_analysis(file_path))

        print("  [7/7] Running OCR confidence analysis...")
        ocr_result, ocr_text, language = ocr_confidence_analysis(img)
        detections.append(ocr_result)

        # Confidence-adjusted weighted ensemble
        weighted_sum = sum(d.score * self.weights.get(d.method, 0) * d.confidence
                           for d in detections if d.confidence > 0)
        total_weight = sum(self.weights.get(d.method, 0) * d.confidence
                           for d in detections if d.confidence > 0)
        overall_score = round(min(1.0, weighted_sum / total_weight
                                  if total_weight > 0 else 0.0), 3)

        verdict = "GENUINE"
        for v, (lo, hi) in VERDICT_THRESHOLDS.items():
            if lo <= overall_score < hi:
                verdict = v
                break

        confidence = round(
            sum(d.confidence * self.weights.get(d.method, 0) for d in detections)
            / sum(self.weights.values()), 3
        )

        heatmap_b64 = generate_heatmap(img, detections)
        flagged     = [d for d in detections if d.score > 0.4]
        summary     = self._build_summary(overall_score, verdict, flagged, language)

        return ForensicReport(
            filename=filename, overall_score=overall_score,
            verdict=verdict, confidence=confidence,
            detections=detections, summary=summary,
            heatmap_b64=heatmap_b64, ocr_text=ocr_text[:1000],
            language_detected=language
        )

    def _build_summary(self, score, verdict, flagged, language):
        lines = [
            f"Verdict: {verdict} (score={score:.3f})",
            f"Language detected: {language}",
            "",
            "Flagged issues:" if flagged else "No significant forgery indicators found.",
        ]
        for d in sorted(flagged, key=lambda x: -x.score):
            lines.append(f"  • [{d.method.upper()}] score={d.score:.3f}: {d.explanation}")
        return "\n".join(lines)


# ─── JSON serialiser ──────────────────────────────────────────────────────────

def _safe(v):
    if isinstance(v, np.integer):  return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.ndarray):  return v.tolist()
    return v


def report_to_dict(report: ForensicReport) -> dict:
    return {
        "filename":         report.filename,
        "overall_score":    report.overall_score,
        "verdict":          report.verdict,
        "confidence":       report.confidence,
        "language":         report.language_detected,
        "summary":          report.summary,
        "ocr_text_preview": report.ocr_text[:300],
        "heatmap_b64":      report.heatmap_b64,
        "detections": [
            {
                "method":             d.method,
                "score":              d.score,
                "confidence":         d.confidence,
                "explanation":        d.explanation,
                "suspicious_regions": [{k: _safe(v) for k, v in r.items()}
                                       for r in d.suspicious_regions[:5]],
                "details":            {k: _safe(v) for k, v in d.details.items()
                                       if not isinstance(v, np.ndarray)}
            }
            for d in report.detections
        ]
    }
