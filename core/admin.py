"""
DentAI - Django Admin Registration
Registers all models with a customized admin interface.
"""

from django.contrib import admin
from django.utils.html import format_html
from .models import Patient, Scan, DetectionResult, ScanComparison


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display  = ('patient_id', 'full_name', 'age', 'gender', 'email', 'scan_count', 'created_at')
    search_fields = ('patient_id', 'first_name', 'last_name', 'email')
    list_filter   = ('gender',)
    readonly_fields = ('patient_id', 'created_at', 'updated_at')
    ordering      = ('-created_at',)

    def scan_count(self, obj):
        return obj.scans.count()
    scan_count.short_description = 'Scans'


class DetectionResultInline(admin.TabularInline):
    model       = DetectionResult
    extra       = 0
    readonly_fields = ('disease_name', 'confidence_percent', 'severity', 'fdi_tooth_number')
    fields      = ('disease_name', 'confidence', 'severity', 'fdi_tooth_number', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2')


@admin.register(Scan)
class ScanAdmin(admin.ModelAdmin):
    list_display  = ('scan_id_short', 'patient', 'scan_type', 'status', 'disease_count', 'highest_severity', 'created_at')
    list_filter   = ('scan_type', 'status')
    search_fields = ('patient__first_name', 'patient__last_name', 'patient__patient_id')
    readonly_fields = ('scan_id', 'inference_time_ms', 'model_version', 'created_at', 'updated_at', 'annotated_preview')
    inlines       = [DetectionResultInline]
    ordering      = ('-created_at',)

    def scan_id_short(self, obj):
        return str(obj.scan_id)[:8].upper()
    scan_id_short.short_description = 'Scan ID'

    def annotated_preview(self, obj):
        if obj.annotated_image:
            return format_html('<img src="{}" style="max-width:300px;"/>', obj.annotated_image.url)
        return '—'
    annotated_preview.short_description = 'Annotated Preview'


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display  = ('disease_name', 'confidence_percent', 'severity', 'fdi_tooth_number', 'scan')
    list_filter   = ('severity', 'disease_name')
    search_fields = ('disease_name', 'scan__patient__first_name')
    ordering      = ('-confidence',)


@admin.register(ScanComparison)
class ScanComparisonAdmin(admin.ModelAdmin):
    list_display = ('patient', 'scan_before', 'scan_after', 'created_at')
    ordering     = ('-created_at',)


# Customize admin site header
admin.site.site_header  = "🦷 DentAI Administration"
admin.site.site_title   = "DentAI Admin"
admin.site.index_title  = "Dental Disease Detection System"
