import os
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.lib.colors import Color, black, white

# ─── Global Constants ────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
ALERT_BG = Color(1, .972549, .882353, 1)
ALERT_TEXT = Color(.901961, .317647, 0, 1)
BLACK_TEXT = Color(.129412, .129412, .129412, 1)
BLUE_FILL = Color(.082353, .396078, .752941, 1)
BLUE_FILL_BG = Color(.890196, .94902, .992157, 1)
GRAY_LIGHT = Color(.960784, .968627, .980392, 1)
GRAY_MID = Color(.933333, .933333, .933333, 1)
GRAY_TEXT = Color(.458824, .458824, .458824, 1)
GREEN_BG = Color(.909804, .960784, .913725, 1)
GREEN_LOW = Color(.219608, .556863, .235294, 1)
ORANGE_BG = Color(1, .952941, .878431, 1)
ORANGE_MED = Color(.960784, .486275, 0, 1)
PURPLE_BG = Color(.952941, .898039, .960784, 1)
PURPLE_CROWN = Color(.415686, .105882, .603922, 1)
RED_BG = Color(1, .921569, .933333, 1)
RED_HIGH = Color(.827451, .184314, .184314, 1)
TEAL = Color(0, .411765, .360784, 1)
TEAL_DARK = Color(0, .301961, .25098, 1)
TEAL_LIGHT = Color(.878431, .94902, .945098, 1)

SEVERITY_COLORS = {
    'high': (RED_HIGH, RED_BG, '🔴 High'),
    'medium': (ORANGE_MED, ORANGE_BG, '🟡 Medium'),
    'low': (GREEN_LOW, GREEN_BG, '🟢 Low')
}

ICD10_MAPPING = {
    "Occlusal Caries": "K02.9",
    "Proximal Caries": "K02.9",
    "Periapical Abscess": "K04.7",
    "Periapical Cyst": "K04.8",
    "Granuloma": "K04.5",
    "Apical Periodontitis": "K04.4",
    "Horizontal Bone Loss": "K05.3",
    "Vertical Bone Loss": "K05.3",
    "Root Canal Required": "K04.0",
    "Caries": "K02.9",
    "Dental Crown": "Restoration",
    "Dental Filling": "Restoration",
    "Dental Implant": "Restoration",
    "Malaligned Tooth": "M26.3",
    "Missing Teeth": "K08.1",
    "Periapical Lesion": "K04.9",
    "Retained Root": "K08.3",
    "Root Canal Treatment": "Restoration",
    "Root Piece": "K08.3",
    "Impacted Tooth": "K00.6",
    "Bone Loss": "K05.3",
    "Fractured Tooth": "S02.5",
    "Bone Defect": "M85.9",
    "Cyst": "K09.9",
    "Root Resorption": "K03.3",
}

# ─── Helper functions ────────────────────────────────────────────────────────
def _build_styles():
    styles = getSampleStyleSheet()
    custom = {
        'SectionHeader': ParagraphStyle(
            'SectionHeader',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            textColor=TEAL_DARK,
            spaceBefore=6,
            spaceAfter=2
        ),
        'CenterBold': ParagraphStyle(
            'CenterBold',
            parent=styles['Normal'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ),
        'RightSmall': ParagraphStyle(
            'RightSmall',
            parent=styles['Normal'],
            alignment=TA_RIGHT,
            fontSize=8
        ),
        'SmallGray': ParagraphStyle(
            'SmallGray',
            parent=styles['Normal'],
            fontSize=8,
            textColor=GRAY_TEXT
        )
    }
    for name, style in custom.items():
        styles.add(style)
    return styles

def _info_table_style():
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), white),
        ('BACKGROUND', (0, 0), (0, -1), GRAY_LIGHT),
        ('BACKGROUND', (2, 0), (2, -1), GRAY_LIGHT),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.3, GRAY_MID),
        ('TEXTCOLOR', (0, 0), (0, -1), GRAY_TEXT),
        ('TEXTCOLOR', (2, 0), (2, -1), GRAY_TEXT),
    ])

def _page_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(0.5)
    canvas.line(1.8 * cm, 1.5 * cm, PAGE_W - 1.8 * cm, 1.5 * cm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GRAY_TEXT)
    canvas.drawCentredString(PAGE_W / 2, 1 * cm, f'Page {doc.page}')
    canvas.drawString(1.8 * cm, 1 * cm, 'DentAI Diagnostic System — Confidential Medical Record')
    canvas.restoreState()

def _image_cell(image_path, max_w, label, styles):
    content = [Paragraph(f"<b>{label}</b>", styles['SmallGray'])]
    if image_path and os.path.exists(str(image_path)):
        img = Image(str(image_path), width=max_w, height=max_w * 0.65)
        content.append(img)
        return content
    else:
        content.append(Paragraph('[Image not available]', styles['SmallGray']))
        return content

# ─── Report Components ───────────────────────────────────────────────────────
def _build_header(scan, styles):
    elements = []
    p1 = Paragraph('<b><font color="#00695C" size="16">🦷 DentAI</font></b><br/><font color="#757575" size="8">AI Dental Diagnostic System</font>', styles['Normal'])
    p2 = Paragraph('<b><font size="14">DENTAL DIAGNOSIS REPORT</font></b>', styles['CenterBold'])
    p3 = Paragraph(f'<font color="#757575" size="8">Report ID: {str(scan.scan_id)[:8].upper()}<br/>Generated: {timezone.now().strftime("%d %b %Y %H:%M")}</font>', styles['RightSmall'])
    
    header_data = [[p1, p2, p3]]
    header_table = Table(header_data, colWidths=['30%', '40%', '30%'])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), TEAL_LIGHT),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROUNDEDCORNERS', [8])
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.15 * cm))
    elements.append(Paragraph('<font size="7" color="#D32F2F"><b>CONFIDENTIAL MEDICAL RECORD</b> — PROTECTED HEALTH INFORMATION (PHI) UNDER HIPAA/GDPR REGULATORY COMPLIANCE</font>', styles['CenterBold']))
    return elements

def _build_patient_info_table(scan, styles):
    elements = []
    elements.append(Paragraph('Patient Information', styles['SectionHeader']))
    elements.append(Spacer(1, 0.2 * cm))
    
    p = scan.patient
    row1 = ['Full Name:', p.full_name, 'Patient ID:', p.patient_id]
    row2 = ['Age:', str(p.age), 'Gender:', p.get_gender_display()]
    row3 = ['Email:', p.email if p.email else '—', 'Phone:', p.phone if p.phone else '—']
    
    data = [row1, row2, row3]
    t = Table(data, colWidths=['18%', '32%', '18%', '32%'])
    t.setStyle(_info_table_style())
    elements.append(t)
    return elements

def _build_scan_info_table(scan, styles):
    elements = []
    elements.append(Paragraph('Scan Information', styles['SectionHeader']))
    elements.append(Spacer(1, 0.2 * cm))
    
    row1 = ['Scan Type:', f'{scan.scan_type} X-Ray', 'Scan Date:', scan.created_at.strftime('%d %b %Y')]
    row2 = ['Status:', scan.get_status_display(), 'Model Version:', scan.model_version]
    row3 = ['Inference:', f'{scan.inference_time_ms:.0f}ms', 'Findings:', str(scan.disease_count)]
    row4 = ['Scan Quality:', 'Satisfactory (Adequate)', 'Clinical Indication:', 'Routine Diagnostic Screening']
    
    data = [row1, row2, row3, row4]
    t = Table(data, colWidths=['18%', '32%', '18%', '32%'])
    t.setStyle(_info_table_style())
    elements.append(t)
    return elements

def _build_xray_section(scan, styles):
    elements = []
    elements.append(Paragraph('X-Ray Analysis Images', styles['SectionHeader']))
    elements.append(Spacer(1, 0.3 * cm))
    
    img_w = (PAGE_W - 5.6 * cm) / 2
    orig_cell = _image_cell(scan.original_image.path, img_w, 'Original X-Ray', styles)
    
    anno_path = os.path.join(settings.MEDIA_ROOT, scan.annotated_image.name) if scan.annotated_image else None
    anno_cell = _image_cell(anno_path, img_w, 'Annotated (AI Analysis)', styles)
    
    img_table = Table([[orig_cell, anno_cell]], colWidths=[img_w + 0.5 * cm, img_w + 0.5 * cm])
    img_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4)
    ]))
    elements.append(img_table)
    return elements

def _build_results_table(scan, styles):
    elements = []
    elements.append(Paragraph('Detection Results', styles['SectionHeader']))
    elements.append(Spacer(1, 0.2 * cm))
    
    results = scan.results.all()
    if not results.exists():
        elements.append(Paragraph('✅ No diseases detected. Scan appears healthy.', styles['Normal']))
        return elements
        
    headers = ['Disease Name', 'Tooth (FDI)', 'Confidence', 'Severity', 'Color Code']
    data = [headers]
    for r in results:
        sev_color, sev_bg, sev_label = SEVERITY_COLORS.get(r.severity, (BLACK_TEXT, GRAY_MID, 'Unknown'))
        tooth = f'T{r.fdi_tooth_number}' if r.fdi_tooth_number else '—'
        icd_code = ICD10_MAPPING.get(r.disease_name)
        display_name = f"<b>{r.disease_name}</b>"
        if icd_code and icd_code != "Restoration":
            display_name += f"<br/><font color='#757575' size='7.5'>ICD-10: {icd_code}</font>"
        data.append([
            Paragraph(display_name, styles['Normal']),
            tooth,
            f'{r.confidence_percent}%',
            sev_label,
            ''
        ])
        
    col_widths = ['35%', '12%', '14%', '18%', '21%']
    t = Table(data, colWidths=col_widths)
    
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), TEAL),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, GRAY_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.3, GRAY_MID),
        ('ROUNDEDCORNERS', [4])
    ]
    
    for row_idx, r in enumerate(results, start=1):
        _, sev_bg, _ = SEVERITY_COLORS.get(r.severity, (BLACK_TEXT, GRAY_LIGHT, ''))
        sev_color, _, _ = SEVERITY_COLORS.get(r.severity, (BLACK_TEXT, GRAY_LIGHT, ''))
        style_cmds.append(('BACKGROUND', (4, row_idx), (4, row_idx), sev_bg))
        style_cmds.append(('TEXTCOLOR', (3, row_idx), (3, row_idx), sev_color))
        
    t.setStyle(TableStyle(style_cmds))
    elements.append(t)
    return elements

def _build_restorations_section(scan, styles):
    elements = []
    elements.append(Paragraph('Restorations Found', styles['SectionHeader']))
    elements.append(Spacer(1, 0.2 * cm))
    
    results = scan.results.filter(disease_name__in=['Dental Filling', 'Dental Crown'])
    if not results.exists():
        elements.append(Paragraph('No restorations (fillings or crowns) detected in this scan.', styles['SmallGray']))
        return elements
        
    headers = ['Restoration Type', 'Tooth (FDI)', 'Confidence', 'Disease Under?', 'Secondary Caries?']
    data = [headers]
    for r in results:
        tooth = f'T{r.fdi_tooth_number}' if r.fdi_tooth_number else '—'
        disease_under = '⚠️ Yes' if r.disease_under_crown else '✔️ No'
        sec_caries = '⚠️ Yes' if r.secondary_caries else '✔️ No'
        data.append([
            r.disease_name,
            tooth,
            f'{r.confidence_percent}%',
            disease_under,
            sec_caries
        ])
        
    col_widths = ['30%', '12%', '14%', '22%', '22%']
    t = Table(data, colWidths=col_widths)
    
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), BLUE_FILL),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, BLUE_FILL_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.3, GRAY_MID)
    ]
    
    for row_idx, r in enumerate(results, start=1):
        if r.disease_under_crown or r.secondary_caries:
            style_cmds.append(('TEXTCOLOR', (3, row_idx), (4, row_idx), RED_HIGH))
            
    t.setStyle(TableStyle(style_cmds))
    elements.append(t)
    return elements

def _build_secondary_findings(scan, styles):
    elements = []
    elements.append(Paragraph('Secondary Findings', styles['SectionHeader']))
    elements.append(Spacer(1, 0.2 * cm))
    
    alerts = []
    for r in scan.results.all():
        if r.disease_under_crown:
            tooth = f' (Tooth T{r.fdi_tooth_number})' if r.fdi_tooth_number else ''
            alerts.append(f'⚠️ Disease detected under crown{tooth} — Crown may need replacement and root canal evaluation.')
        if r.secondary_caries:
            tooth = f' (Tooth T{r.fdi_tooth_number})' if r.fdi_tooth_number else ''
            alerts.append(f'⚠️ Secondary caries detected around filling{tooth} — Remove filling, treat caries, re-restore.')
            
    if not alerts:
        elements.append(Paragraph('✅ No secondary findings. Existing restorations appear intact.', styles['Normal']))
        return elements
        
    for alert_text in alerts:
        alert_box = Table([[Paragraph(alert_text, styles['Normal'])]], colWidths=['100%'])
        alert_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), ALERT_BG),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TEXTCOLOR', (0, 0), (-1, -1), ALERT_TEXT),
        ]))
        elements.append(alert_box)
        elements.append(Spacer(1, 0.15 * cm))
        
    return elements

def _build_recommendations(scan, styles):
    elements = []
    elements.append(Paragraph('Recommended Next Steps', styles['SectionHeader']))
    elements.append(Spacer(1, 0.2 * cm))
    
    recommendations = {
        'Occlusal Caries': 'Schedule composite resin filling. Advise fluoride treatment and dietary changes.',
        'Proximal Caries': 'Interproximal restoration recommended. Consider flossing regimen guidance.',
        'Periapical Abscess': '⚠️ URGENT: Root canal treatment required immediately. Prescribe antibiotics.',
        'Periapical Cyst': 'Surgical enucleation may be required. Refer to oral surgeon. Monitor radiographically.',
        'Granuloma': 'Root canal treatment or apicectomy required. Follow-up in 3 months.',
        'Apical Periodontitis': 'Root canal therapy recommended. Assess restorability of tooth.',
        'Horizontal Bone Loss': 'Periodontal treatment and scaling/root planing required. Refer to periodontist.',
        'Vertical Bone Loss': 'Surgical periodontal intervention likely needed. Immediate consultation.',
        'Root Canal Treated': 'Existing root canal treatment noted. Monitor for long-term stability.',
        'Milk Tooth': 'Monitor for natural exfoliation. Do not extract prematurely unless indicated.',
        'Healthy': 'No immediate intervention required. Routine 6-month follow-up.'
    }
    
    disease_names = scan.results.values_list('disease_name', flat=True).distinct()
    if not disease_names:
        elements.append(Paragraph('No specific recommendations. Routine follow-up advised.', styles['Normal']))
        return elements
        
    for disease in disease_names:
        rec = recommendations.get(disease, 'Consult with specialist for further evaluation.')
        r_obj = scan.results.filter(disease_name=disease).first()
        severity = r_obj.severity if r_obj else 'low'
        _, sev_bg, _ = SEVERITY_COLORS.get(severity, (BLACK_TEXT, GRAY_LIGHT, ''))
        
        row = Table([[Paragraph(f'<b>{disease}</b>', styles['Normal']), Paragraph(rec, styles['SmallGray'])]], colWidths=['30%', '70%'])
        row.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), sev_bg),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        elements.append(row)
        elements.append(Spacer(1, 0.15 * cm))
        
    return elements

def _build_doctor_notes(scan, styles):
    elements = []
    elements.append(Paragraph('Doctor Notes', styles['SectionHeader']))
    elements.append(Spacer(1, 0.2 * cm))
    
    notes_box = Table([[Paragraph(scan.doctor_notes, styles['Normal'])]], colWidths=['100%'])
    notes_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), GRAY_LIGHT),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ROUNDEDCORNERS', [6])
    ]))
    elements.append(notes_box)
    return elements

def _build_footer(scan, styles):
    elements = []
    elements.append(HRFlowable(width='100%', thickness=0.5, color=TEAL_LIGHT))
    elements.append(Spacer(1, 0.3 * cm))
    
    p1 = Paragraph('<font color="#757575" size="8">This report is generated by DentAI AI Diagnostic System.<br/>AI analysis is intended to assist, not replace, clinical judgment.<br/>All findings should be verified by a licensed dental professional.</font>', styles['Normal'])
    p2 = Paragraph(f'<b>Verified by:</b><br/><br/><br/>________________________<br/><font size="8">Dr. ________________<br/>License / NPI: ________________<br/>Date: {timezone.now().strftime("%d/%m/%Y")}</font>', styles['RightSmall'])
    
    footer_table = Table([[p1, p2]], colWidths=['60%', '40%'])
    footer_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    elements.append(footer_table)
    return elements

# ─── Main Generation Entry Point ─────────────────────────────────────────────
def generate_pdf_report(scan, request=None) -> str:
    report_dir = Path(settings.MEDIA_ROOT) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = str(report_dir / f'report_{scan.patient.patient_id}_{scan.scan_id}.pdf')
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2 * cm,
        title=f'DentAI Report - {scan.patient.full_name}',
        author='DentAI Diagnostic System'
    )
    
    styles = _build_styles()
    story = []
    
    story.extend(_build_header(scan, styles))
    story.append(Spacer(1, 0.5 * cm))
    
    story.extend(_build_patient_info_table(scan, styles))
    story.append(Spacer(1, 0.4 * cm))
    
    story.extend(_build_scan_info_table(scan, styles))
    story.append(Spacer(1, 0.6 * cm))
    
    story.extend(_build_xray_section(scan, styles))
    story.append(Spacer(1, 0.6 * cm))
    
    story.extend(_build_results_table(scan, styles))
    story.append(Spacer(1, 0.6 * cm))
    
    story.extend(_build_restorations_section(scan, styles))
    story.append(Spacer(1, 0.6 * cm))
    
    story.extend(_build_secondary_findings(scan, styles))
    story.append(Spacer(1, 0.6 * cm))
    
    story.extend(_build_recommendations(scan, styles))
    story.append(Spacer(1, 0.6 * cm))
    
    if scan.doctor_notes:
        story.extend(_build_doctor_notes(scan, styles))
        story.append(Spacer(1, 0.6 * cm))
        
    story.extend(_build_footer(scan, styles))
    
    doc.build(story, onFirstPage=_page_header_footer, onLaterPages=_page_header_footer)
    return pdf_path
