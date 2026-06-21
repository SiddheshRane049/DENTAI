"""
DentAI - OpenCV Landmark & Annotation Engine  (v3 — HD + Anti-Collision + 18 Classes)
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ─── HD Scale Factor ──────────────────────────────────────────────────────────
HD_SCALE = 2
JPEG_Q   = 97

# ─── Color constants (BGR for OpenCV) ─────────────────────────────────────────
COLOR_TEAL   = (0, 105, 92)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,   0,   0)
COLOR_GREEN  = (0,   200,  0)
COLOR_YELLOW = (0,   200, 200)
COLOR_RED    = (0,   50,  220)
COLOR_ORANGE = (0,   165, 255)
COLOR_BLUE   = (220, 100, 50)
COLOR_PURPLE = (180, 50,  180)
COLOR_CYAN   = (180, 180, 30)
COLOR_AMBER  = (30,  140, 220)
COLOR_BROWN  = (40,  100, 180)

SEVERITY_COLORS = {
    'low':    COLOR_GREEN,
    'medium': COLOR_ORANGE,
    'high':   COLOR_RED,
}

# Per-class box color overrides (BGR)
CLASS_BOX_COLOR = {
    'Dental Filling':  COLOR_BLUE,
    'Dental Crown':    COLOR_PURPLE,
    'Dental Implant':  COLOR_CYAN,
    'Impacted Tooth':  COLOR_AMBER,
    'Calculus':        COLOR_BROWN,
    'Root Fracture':   (50, 50, 200),
    'Retained Root':   (80, 80, 160),
    'Milk Tooth':      (200, 180, 60),
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def draw_annotations(image_path: str, detections: list, output_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")

    orig_h, orig_w = img.shape[:2]

    # 1. 4K UHD Enhancement
    img   = _enhance_xray_hd(img, target_width=3840)
    hd_h, hd_w = img.shape[:2]

    # 2. First pass — generate all landmarks for anti-collision
    all_landmark_specs = []
    for det in detections:
        x1 = int(det['bbox']['x1'] * hd_w)
        y1 = int(det['bbox']['y1'] * hd_h)
        x2 = int(det['bbox']['x2'] * hd_w)
        y2 = int(det['bbox']['y2'] * hd_h)
        lms = _generate_landmarks(x1, y1, x2, y2, det['disease_name'])
        det['_lm_px'] = lms
        for (px, py, lbl) in lms:
            tw, th = _text_size(lbl)
            all_landmark_specs.append([px, py, lbl, tw, th])

    resolved = _resolve_label_positions(all_landmark_specs, hd_w, hd_h)
    lm_iter  = iter(resolved)

    # 3. Second pass — draw everything
    for det in detections:
        disease  = det['disease_name']
        severity = det.get('severity', 'medium')
        color_rgb = det.get('color')
        if color_rgb:
            box_color = _convert_rgb_to_bgr(color_rgb)
        else:
            box_color = CLASS_BOX_COLOR.get(disease, SEVERITY_COLORS.get(severity, COLOR_ORANGE))

        x1 = int(det['bbox']['x1'] * hd_w)
        y1 = int(det['bbox']['y1'] * hd_h)
        x2 = int(det['bbox']['x2'] * hd_w)
        y2 = int(det['bbox']['y2'] * hd_h)

        _draw_bbox(img, x1, y1, x2, y2, box_color, severity)

        fdi       = det.get('fdi_tooth_number')
        tooth_lbl = f"T{fdi}" if fdi else ""
        top_label = f"{disease} {tooth_lbl} {det['confidence']:.0%}".strip()
        _draw_label(img, top_label, x1, y1, box_color, hd_h)

        lms = det.get('_lm_px', [])
        det['landmarks'] = [
            {'x': round(px / hd_w, 4), 'y': round(py / hd_h, 4), 'label': lbl}
            for (px, py, lbl) in lms
        ]
        for (px, py, lbl) in lms:
            lx, ly = next(lm_iter)
            _draw_landmark_point(img, px, py, lx, ly, lbl)

    # 4. Header bar
    _draw_header_bar(img, hd_w, hd_h, len(detections))

    # 5. Save HD output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    logger.info(f"HD annotated image saved → {output_path}  ({hd_w}×{hd_h})")
    return output_path


def draw_opg_dental_chart(image_path: str, detections: list, output_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")

    img  = _enhance_xray_hd(img, target_width=3840)
    hd_h, hd_w = img.shape[:2]

    fdi_severity = {}
    for det in detections:
        tooth = det.get('fdi_tooth_number')
        if tooth:
            order = {'low': 0, 'medium': 1, 'high': 2}
            if order.get(det.get('severity', 'low'), 0) > order.get(fdi_severity.get(tooth, 'low'), 0):
                fdi_severity[tooth] = det['severity']

    chart_h = 100
    chart   = np.ones((chart_h, hd_w, 3), dtype=np.uint8) * 20
    upper   = list(range(18, 10, -1)) + list(range(21, 29))
    lower   = list(range(48, 40, -1)) + list(range(31, 39))
    tw      = hd_w // 16

    for idx, tooth in enumerate(upper):
        x = idx * tw
        c = SEVERITY_COLORS.get(fdi_severity.get(tooth), (60, 60, 60))
        cv2.rectangle(chart, (x+3, 6), (x+tw-3, chart_h//2-6), c, -1)
        cv2.putText(chart, str(tooth), (x+4, chart_h//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    for idx, tooth in enumerate(lower):
        x = idx * tw
        c = SEVERITY_COLORS.get(fdi_severity.get(tooth), (60, 60, 60))
        cv2.rectangle(chart, (x+3, chart_h//2+6), (x+tw-3, chart_h-6), c, -1)
        cv2.putText(chart, str(tooth), (x+4, chart_h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    combined = np.vstack([img, chart])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, combined, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ANTI-COLLISION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_label_positions(specs: list, img_w: int, img_h: int) -> list:
    import math
    placed = []
    result = []
    for item in specs:
        px, py, lbl, tw, th = item
        pad = 4
        
        placed_pos = None
        found = False
        
        # Radial search starting from radius=30px (outside landmark circle) up to 400px
        for r in range(30, 400, 15):
            for step in range(16):
                angle = step * (2 * math.pi / 16)
                candidate_lx = px + int(r * math.cos(angle))
                candidate_ly = py + int(r * math.sin(angle))
                
                # Constrain candidate label within image boundaries with padding
                lx = min(max(pad, candidate_lx), img_w - tw - pad)
                ly = min(max(th + pad, candidate_ly), img_h - pad)
                
                rect = (lx - pad, ly - th - pad, lx + tw + pad, ly + pad)
                
                if not _any_overlap(rect, placed):
                    placed.append(rect)
                    placed_pos = (lx, ly)
                    found = True
                    break
            if found:
                break
                
        if placed_pos is None:
            lx = min(max(pad, px + 12), img_w - tw - pad)
            ly = max(th + pad, py + 4)
            placed_pos = (lx, ly)
            placed.append((lx - pad, ly - th - pad, lx + tw + pad, ly + pad))
            
        result.append(placed_pos)
        
    return result


def _any_overlap(rect, placed: list) -> bool:
    MARGIN = 2
    r1x1, r1y1, r1x2, r1y2 = rect
    for (r2x1, r2y1, r2x2, r2y2) in placed:
        if (r1x1-MARGIN < r2x2 and r1x2+MARGIN > r2x1 and
                r1y1-MARGIN < r2y2 and r1y2+MARGIN > r2y1):
            return True
    return False


def _text_size(text: str, font_scale: float = 1.3, thickness: int = 2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return tw, th


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _enhance_xray_hd(img: np.ndarray, target_width: int = 3840) -> np.ndarray:
    """
    4K UHD Upscaling Pipeline:
    1. Lanczos4 interpolation up to 4K width (3840px)
    2. LAB-space CLAHE to enhance contrast without losing detail
    3. Aggressive Unsharp Masking to remove blur and sharpen edges
    """
    h, w = img.shape[:2]
    
    # Scale to exactly 4K width (3840px) while maintaining aspect ratio
    scale = target_width / w
    if scale > 1.0:
        new_w = target_width
        new_h = int(h * scale)
        img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # CLAHE Contrast Enhancement
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img   = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    
    # Anti-Blur Sharpening (Unsharp Mask)
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    img      = cv2.addWeighted(img, 2.0, gaussian, -1.0, 0)
    
    return img


def _draw_bbox(img, x1, y1, x2, y2, color, severity):
    overlay = img.copy()
    alpha   = 0.12 if severity == 'low' else 0.18 if severity == 'medium' else 0.24
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    thickness = 4 if severity == 'low' else 6
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    clen = max(8, min(20, (x2-x1)//5, (y2-y1)//5))
    ct   = 3
    for p1, p2 in [
        ((x1,y1),(x1+clen,y1)), ((x1,y1),(x1,y1+clen)),
        ((x2,y1),(x2-clen,y1)), ((x2,y1),(x2,y1+clen)),
        ((x1,y2),(x1+clen,y2)), ((x1,y2),(x1,y2-clen)),
        ((x2,y2),(x2-clen,y2)), ((x2,y2),(x2,y2-clen)),
    ]:
        cv2.line(img, p1, p2, color, ct)


def _draw_label(img, text, x1, y1, color, img_h):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.6
    thickness  = 3
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad    = 10
    lbl_y1 = y1 - th - 2*pad
    lbl_y2 = y1
    if lbl_y1 < 0:
        lbl_y1 = y1
        lbl_y2 = y1 + th + 2*pad
    cv2.rectangle(img, (x1, lbl_y1), (x1 + tw + 2*pad, lbl_y2), color, -1)
    cv2.putText(img, text, (x1+pad, lbl_y2-pad), font, font_scale, COLOR_WHITE, thickness, cv2.LINE_AA)


def _generate_landmarks(x1, y1, x2, y2, disease_name: str) -> list:
    cx  = (x1 + x2) // 2
    cy  = (y1 + y2) // 2
    bh  = y2 - y1
    bw  = x2 - x1
    q4h = max(bh // 4, 1)
    q4w = max(bw // 4, 1)

    lms = []

    if 'Caries' in disease_name or 'caries' in disease_name:
        lms.append((cx, y1 + q4h,  'caries_site'))

    elif 'Abscess' in disease_name or 'Cyst' in disease_name or 'Granuloma' in disease_name or 'Lesion' in disease_name:
        lms.append((cx,          cy,     'lesion_apex'))
        lms.append((cx,          y2-4,   'root_tip'))

    elif 'Horizontal Bone Loss' in disease_name:
        lms.append((x1 + q4w,   cy,     'bone_level_L'))
        lms.append((x2 - q4w,   cy,     'bone_level_R'))

    elif 'Vertical Bone Loss' in disease_name:
        lms.append((x1 + q4w,   cy,     'defect_wall'))
        lms.append((cx,          y2-4,   'bone_base'))

    elif 'Bone Loss' in disease_name or 'bone loss' in disease_name:
        lms.append((x1 + q4w,   cy,     'bone_level_L'))
        lms.append((x2 - q4w,   cy,     'bone_level_R'))

    elif 'Periodontitis' in disease_name:
        lms.append((x1 + q4w,   y1+q4h, 'CEJ_pt'))
        lms.append((cx,          cy,     'defect_base'))

    elif 'Root Canal' in disease_name:
        lms.append((cx,          y1+q4h, 'pulp_chamber'))
        lms.append((cx,          y2-q4h, 'root_apex'))

    elif 'Filling' in disease_name:
        lms.append((cx,          cy,     'filling_site'))

    elif 'Crown' in disease_name:
        lms.append((cx,          y1+q4h, 'crown_margin'))

    elif 'Implant' in disease_name:
        lms.append((cx,          y1+q4h, 'implant_neck'))
        lms.append((cx,          y2-q4h, 'implant_apex'))

    elif 'Impacted' in disease_name or 'impacted' in disease_name:
        lms.append((cx,          cy,     'impacted_crown'))
        lms.append((cx,          y2-4,   'follicle_base'))

    elif 'Calculus' in disease_name:
        lms.append((cx,          y1+q4h, 'calculus_deposit'))

    elif 'Fracture' in disease_name or 'Fractured' in disease_name:
        lms.append((cx,          cy,     'fracture_line'))

    elif 'Retained Root' in disease_name or 'Root Piece' in disease_name:
        lms.append((cx,          cy,     'root_fragment'))

    elif 'Milk Tooth' in disease_name or 'Primary teeth' in disease_name:
        lms.append((cx,          cy,     'deciduous_tooth'))

    elif 'Healthy' in disease_name:
        lms.append((cx,          cy,     'normal_region'))

    else:
        lms.append((cx,          cy,     'detection_site'))

    return lms


def _draw_landmark_point(img, cx: int, cy: int, lx: int, ly: int, label: str):
    font_scale = 1.3
    thickness  = 2
    pad        = 8

    cv2.circle(img, (cx, cy), 18, COLOR_WHITE, 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 10, COLOR_TEAL, -1, cv2.LINE_AA)

    dist = ((lx-cx)**2 + (ly-cy)**2) ** 0.5
    if dist > 30:
        cv2.line(img, (cx, cy), (lx, ly), COLOR_WHITE, 3, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    overlay = img.copy()
    cv2.rectangle(overlay, (lx-pad, ly-th-pad), (lx+tw+pad, ly+pad), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                COLOR_WHITE, thickness, cv2.LINE_AA)


def _draw_header_bar(img, w: int, h: int, detection_count: int):
    bar_h   = 80
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 105, 92), -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv2.putText(img, f"DentAI Analysis  |  Findings: {detection_count}",
                (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WHITE, 2, cv2.LINE_AA)
    legend_items = [('Low', COLOR_GREEN), ('Medium', COLOR_ORANGE), ('High', COLOR_RED)]
    lx = w - 600
    for lbl, color in legend_items:
        cv2.circle(img, (lx, 40), 14, color, -1, cv2.LINE_AA)
        cv2.putText(img, lbl, (lx+24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 2, cv2.LINE_AA)
        lx += 180


def _convert_rgb_to_bgr(rgb_tuple):
    r, g, b = rgb_tuple
    return (b, g, r)
