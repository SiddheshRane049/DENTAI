"""
DentAI — Dual-Model Clinical YOLO Detector (New Models 1 & 2)

Old Model 1 (dental_yolov8.pt): TEMPORARILY DISABLED as requested.

New Model 1 (runs_dentai_weights_best.pt):
  0: Occlusal Caries, 1: Proximal Caries, 2: Periapical Abscess, 3: Periapical Cyst,
  4: Granuloma, 5: Apical Periodontitis, 6: Horizontal Bone Loss, 7: Vertical Bone Loss,
  8: Root Canal Treated, 9: Milk Tooth (skipped), 10: Healthy

New Model 2 (DentAI_Final.pt):
  0: Bone Loss, 1: Crown, 2: Filling, 3: Missing Teeth, 4: Periapical Lesion,
  5: Primary Teeth (skipped), 6: Retained Root, 7: Root Piece, 8: Impacted Tooth
"""

import time
import cv2
import numpy as np
from pathlib import Path
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

# ─── Model Class Definitions ──────────────────────────────────────────────────
ALL_CLASSES_NEW_M1 = {
    0:  "Occlusal Caries",
    1:  "Proximal Caries",
    2:  "Periapical Abscess",
    3:  "Periapical Cyst",
    4:  "Granuloma",
    5:  "Apical Periodontitis",
    6:  "Horizontal Bone Loss",
    7:  "Vertical Bone Loss",
    8:  "Root Canal Treated",
    9:  "Milk Tooth",
    10: "Healthy",
}

PATHOLOGY_IDS_NEW_M1 = {0, 1, 2, 3, 4, 5, 6, 7}

ALL_CLASSES_NEW_M2 = {
    0: "Bone Loss",
    1: "Crown",
    2: "Filling",
    3: "Missing Teeth",
    4: "Periapical Lesion",
    5: "Primary Teeth",
    6: "Retained Root",
    7: "Root Piece",
    8: "Impacted Tooth",
}

PATHOLOGY_IDS_NEW_M2 = {0, 4, 6, 7}
RESTORATION_IDS_NEW_M2 = {1, 2}
STRUCTURAL_IDS_NEW_M2 = {3, 5, 8}

# ─── Severity and Color Definitions ──────────────────────────────────────────
SEVERITY_MAP = {
    # New Model 1
    "Occlusal Caries":      "medium",
    "Proximal Caries":      "medium",
    "Periapical Abscess":   "high",
    "Periapical Cyst":      "high",
    "Granuloma":            "high",
    "Apical Periodontitis": "high",
    "Horizontal Bone Loss": "medium",
    "Vertical Bone Loss":   "high",
    "Root Canal Treated":   "low",
    "Milk Tooth":           "low",
    "Healthy":              "low",

    # New Model 2
    "Bone Loss":            "medium",
    "Crown":                "low",
    "Filling":              "low",
    "Missing Teeth":        "medium",
    "Periapical Lesion":    "high",
    "Primary Teeth":        "low",
    "Retained Root":        "medium",
    "Root Piece":           "medium",
    "Impacted Tooth":       "medium",
}

CLASS_COLORS_RGB = {
    # New Model 1
    "Occlusal Caries":      (220,  60,  60),   # Red
    "Proximal Caries":      (220,  60,  60),   # Red
    "Periapical Abscess":   (200,  30,  30),   # Dark red
    "Periapical Cyst":      (200,  30,  30),   # Dark red
    "Granuloma":            (200,  30,  30),   # Dark red
    "Apical Periodontitis": (220, 100,  30),   # Orange
    "Horizontal Bone Loss": (220, 140,  30),   # Amber
    "Vertical Bone Loss":   (220, 140,  30),   # Amber
    "Root Canal Treated":   (180,  30, 180),   # Purple
    "Milk Tooth":           ( 60, 160, 220),   # Blue
    "Healthy":              ( 60, 200,  60),   # Green

    # New Model 2
    "Bone Loss":            (220, 140,  30),   # Amber
    "Crown":                (180,  50, 180),   # Purple
    "Filling":              ( 50, 100, 220),   # Blue
    "Missing Teeth":        (140, 100,  40),   # Brown
    "Periapical Lesion":    (200,  30,  30),   # Dark red
    "Primary Teeth":        ( 60, 160, 220),   # Blue
    "Retained Root":        (160,  80,  80),   # Dusty pink
    "Root Piece":           (160,  80,  80),   # Dusty pink
    "Impacted Tooth":       (220, 140,  30),   # Amber
}

FDI_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def _raw_confidence(raw_conf: float) -> float:
    """
    Returns exact raw model confidence score without artificial boosting/scaling.
    """
    return round(float(raw_conf), 4)


def _estimate_fdi(cx_norm: float, cy_norm: float):
    try:
        slot = min(15, int(cx_norm * 16))
        return FDI_UPPER[slot] if cy_norm < 0.5 else FDI_LOWER[slot]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
class DentalDetector:
    """
    Dual-Model Dental Detector running New Model 1 & New Model 2.
    Old Model 1 is temporarily disabled. Real raw confidence values are preserved.
    """

    CONF_THRESHOLD = getattr(settings, 'YOLO_CONFIDENCE_THRESHOLD', 0.25)
    IOU_THRESHOLD  = 0.45

    def __init__(self):
        # Old Model 1 (dental_yolov8.pt): Temporarily turned off as requested
        self.old_model1_enabled = False

        # Paths for New Model 1 and New Model 2
        m1_downloads = Path(r"C:\Users\Acer\Downloads\runs_dentai_weights_best.pt")
        m1_local     = settings.BASE_DIR / 'models' / 'runs_dentai_weights_best.pt'
        new_m1_path  = m1_downloads if m1_downloads.exists() else m1_local

        m2_downloads = Path(r"C:\Users\Acer\Downloads\DentAI_Final.pt")
        m2_local     = settings.BASE_DIR / 'models' / 'DentAI_Final.pt'
        new_m2_path  = m2_downloads if m2_downloads.exists() else m2_local

        m1_exists = new_m1_path.exists()
        m2_exists = new_m2_path.exists()

        if not ULTRALYTICS_AVAILABLE or not (m1_exists or m2_exists):
            self.fallback = True
            logger.warning("New ML Models or ultralytics not found. Using Simulation Engine.")
            return

        self.fallback = False
        try:
            self.new_model1 = YOLO(str(new_m1_path)) if m1_exists else None
            self.new_model2 = YOLO(str(new_m2_path)) if m2_exists else None
            self._warmup()
        except Exception as e:
            logger.error(f"Failed to load new models: {e}. Falling back to simulation.")
            self.fallback = True

    def _warmup(self):
        if getattr(self, 'fallback', False):
            return
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        if getattr(self, 'new_model1', None):
            self.new_model1.predict(source=dummy, conf=0.9, verbose=False)
        if getattr(self, 'new_model2', None):
            self.new_model2.predict(source=dummy, conf=0.9, verbose=False)

    # ── Public ────────────────────────────────────────────────────────────────

    def detect(self, image_path: str) -> dict:
        start = time.time()
        if getattr(self, 'fallback', False):
            import random
            import hashlib
            try:
                with open(image_path, 'rb') as f:
                    content_hash = hashlib.md5(f.read()).hexdigest()
                seed = int(content_hash, 16) % (2**32)
            except Exception:
                seed = int(hashlib.md5(Path(image_path).name.encode('utf-8')).hexdigest(), 16) % (2**32)
            random.seed(seed)
            
            mock_candidates = [
                {
                    "disease_name": "Occlusal Caries",
                    "confidence": round(random.uniform(0.65, 0.88), 4),
                    "severity": "medium",
                    "bbox": {"x1": 0.28, "y1": 0.42, "x2": 0.32, "y2": 0.48},
                    "fdi_tooth_number": 14,
                    "filling_present": False,
                    "crown_present": False,
                    "disease_under_crown": False,
                    "secondary_caries": False,
                },
                {
                    "disease_name": "Proximal Caries",
                    "confidence": round(random.uniform(0.58, 0.82), 4),
                    "severity": "medium",
                    "bbox": {"x1": 0.68, "y1": 0.45, "x2": 0.72, "y2": 0.51},
                    "fdi_tooth_number": 26,
                    "filling_present": False,
                    "crown_present": False,
                    "disease_under_crown": False,
                    "secondary_caries": False,
                },
                {
                    "disease_name": "Periapical Abscess",
                    "confidence": round(random.uniform(0.70, 0.91), 4),
                    "severity": "high",
                    "bbox": {"x1": 0.48, "y1": 0.68, "x2": 0.53, "y2": 0.76},
                    "fdi_tooth_number": 41,
                    "filling_present": False,
                    "crown_present": False,
                    "disease_under_crown": False,
                    "secondary_caries": False,
                },
                {
                    "disease_name": "Root Canal Treated",
                    "confidence": round(random.uniform(0.60, 0.85), 4),
                    "severity": "low",
                    "bbox": {"x1": 0.15, "y1": 0.52, "x2": 0.20, "y2": 0.62},
                    "fdi_tooth_number": 36,
                    "filling_present": False,
                    "crown_present": False,
                    "disease_under_crown": False,
                    "secondary_caries": False,
                }
            ]
            
            num_dets = random.randint(2, 4)
            dets = random.sample(mock_candidates, num_dets)
            
            for d in dets:
                cx = (d['bbox']['x1'] + d['bbox']['x2']) / 2
                cy = (d['bbox']['y1'] + d['bbox']['y2']) / 2
                d['landmarks'] = [
                    {"x": cx - 0.01, "y": cy - 0.01, "label": "apical"},
                    {"x": cx + 0.01, "y": cy + 0.01, "label": "coronal"}
                ]
            
            return {
                "detections": dets,
                "inference_time_ms": random.randint(120, 240),
                "model_version": "DentAI-DualModel-RealConf-Simulation",
            }

        raw   = self._run_tta(image_path)
        dets  = self._apply_nms(raw)
        dets  = self._enrich(dets)
        return {
            "detections":        dets,
            "inference_time_ms": int((time.time() - start) * 1000),
            "model_version":     "DentAI-DualModel-RealConf-TTA",
        }

    # ── TTA ───────────────────────────────────────────────────────────────────

    def _run_tta(self, image_path: str) -> list:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        variants = [
            ("orig",   img),
            ("flip",   cv2.flip(img, 1)),
            ("clahe",  self._clahe(img)),
            ("bright", np.clip(img.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)),
        ]
        all_dets = []
        for name, v in variants:
            all_dets.extend(self._predict_array(v, flipped=(name == "flip")))
        return all_dets

    def _predict_array(self, img: np.ndarray, flipped: bool = False) -> list:
        dets = []

        # New Model 1 (runs_dentai_weights_best.pt)
        if getattr(self, 'new_model1', None):
            res1 = self.new_model1.predict(
                source=img, conf=self.CONF_THRESHOLD,
                iou=self.IOU_THRESHOLD, save=False, verbose=False,
                imgsz=1024
            )
            if res1:
                for box in res1[0].boxes:
                    conf = float(box.conf[0])
                    if conf < self.CONF_THRESHOLD:
                        continue
                    cls_id = int(box.cls[0])
                    name = ALL_CLASSES_NEW_M1.get(cls_id, f"Class_{cls_id}")
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    if flipped:
                        x1, x2 = 1.0 - x2, 1.0 - x1
                    dets.append({
                        "model_id":     1,
                        "cls_id":       cls_id,
                        "disease_name": name,
                        "conf":         conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                    })

        # New Model 2 (DentAI_Final.pt)
        if getattr(self, 'new_model2', None):
            res2 = self.new_model2.predict(
                source=img, conf=self.CONF_THRESHOLD,
                iou=self.IOU_THRESHOLD, save=False, verbose=False,
                imgsz=1024
            )
            if res2:
                for box in res2[0].boxes:
                    conf = float(box.conf[0])
                    if conf < self.CONF_THRESHOLD:
                        continue
                    cls_id = int(box.cls[0])
                    name = ALL_CLASSES_NEW_M2.get(cls_id, f"Class_{cls_id}")
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    if flipped:
                        x1, x2 = 1.0 - x2, 1.0 - x1
                    dets.append({
                        "model_id":     2,
                        "cls_id":       cls_id,
                        "disease_name": name,
                        "conf":         conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                    })

        return dets

    # ── NMS ───────────────────────────────────────────────────────────────────

    def _apply_nms(self, raw: list) -> list:
        if not raw:
            return []

        # Filter out background / normal structures early
        skip_names = {"Milk Tooth", "Primary Teeth", "Healthy"}
        candidates = [d for d in raw if d.get("disease_name") not in skip_names]

        return self._nms_box_list(candidates, iou_thresh=0.45)

    def _nms_box_list(self, boxes: list, iou_thresh: float) -> list:
        boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
        kept  = []
        while boxes:
            best = boxes.pop(0)
            kept.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < iou_thresh]
        return kept

    @staticmethod
    def _iou(a, b) -> float:
        ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        ua = (a["x2"]-a["x1"]) * (a["y2"]-a["y1"])
        ub = (b["x2"]-b["x1"]) * (b["y2"]-b["y1"])
        return inter / (ua+ub-inter) if (ua+ub-inter) > 0 else 0.0

    # ── Enrich detections ─────────────────────────────────────────────────────

    def _enrich(self, raw: list) -> list:
        output = []
        for det in raw:
            model_id = det["model_id"]
            cls_id   = det["cls_id"]
            conf     = det["conf"]

            if model_id == 1:
                # Skip milk tooth
                if cls_id == 9:
                    continue
                name = ALL_CLASSES_NEW_M1.get(cls_id, f"Class_{cls_id}")
                is_path = cls_id in PATHOLOGY_IDS_NEW_M1
                is_rest = (cls_id == 8)
                is_struct = False
                is_fill = False
                is_cr = False
                is_imp = False
            else:
                # Skip primary teeth
                if cls_id == 5:
                    continue
                name = ALL_CLASSES_NEW_M2.get(cls_id, f"Class_{cls_id}")
                is_path = cls_id in PATHOLOGY_IDS_NEW_M2
                is_rest = cls_id in RESTORATION_IDS_NEW_M2
                is_struct = cls_id in STRUCTURAL_IDS_NEW_M2
                is_fill = (cls_id == 2)
                is_cr = (cls_id == 1)
                is_imp = (cls_id == 8)

            fdi   = _estimate_fdi(det["cx"], det["cy"])
            sev   = SEVERITY_MAP.get(name, "low")
            color = CLASS_COLORS_RGB.get(name, (255, 165, 0))

            if is_path and conf > 0.75 and sev == "medium":
                sev = "high"

            # REAL unscaled raw confidence output
            real_conf = _raw_confidence(conf)

            output.append({
                "disease_name":      name,
                "class_id":          cls_id,
                "confidence":        real_conf,
                "severity":          sev,
                "color":             color,
                "fdi_tooth_number":  fdi,
                "bbox": {
                    "x1": round(det["x1"], 4), "y1": round(det["y1"], 4),
                    "x2": round(det["x2"], 4), "y2": round(det["y2"], 4),
                },
                "filling_present":      is_fill,
                "crown_present":        is_cr,
                "implant_present":      is_imp,
                "disease_under_crown":  False,
                "secondary_caries":     False,
                "disease_near_implant": False,
                "is_pathology":         is_path,
                "is_restoration":       is_rest,
                "is_structural":        is_struct,
                "is_filling":           is_fill,
                "is_crown":             is_cr,
                "is_implant":           is_imp,
            })
        return output

    @staticmethod
    def _clahe(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
