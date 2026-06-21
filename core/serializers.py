"""
DentAI - Django REST Framework Serializers
Converts Django model instances to/from JSON for the API.
"""

from rest_framework import serializers
from .models import Patient, Scan, DetectionResult, ScanComparison


class DetectionResultSerializer(serializers.ModelSerializer):
    """Serializes a single disease detection finding."""

    confidence_percent = serializers.ReadOnlyField()
    severity_emoji     = serializers.ReadOnlyField()

    class Meta:
        model  = DetectionResult
        fields = [
            'id', 'disease_name', 'confidence', 'confidence_percent',
            'severity', 'severity_emoji', 'bbox_x1', 'bbox_y1',
            'bbox_x2', 'bbox_y2', 'fdi_tooth_number', 'landmarks',
            'filling_present', 'crown_present',
            'disease_under_crown', 'secondary_caries',
            'created_at',
        ]


class ScanSerializer(serializers.ModelSerializer):
    """Full scan details with nested detection results."""

    results          = DetectionResultSerializer(many=True, read_only=True)
    disease_count    = serializers.ReadOnlyField()
    highest_severity = serializers.ReadOnlyField()
    patient_name     = serializers.SerializerMethodField()
    original_image_url   = serializers.SerializerMethodField()
    annotated_image_url  = serializers.SerializerMethodField()

    class Meta:
        model  = Scan
        fields = [
            'id', 'scan_id', 'patient', 'patient_name',
            'scan_type', 'status', 'original_image', 'original_image_url',
            'annotated_image', 'annotated_image_url',
            'inference_time_ms', 'model_version', 'doctor_name', 'doctor_notes',
            'report_pdf', 'disease_count', 'highest_severity',
            'results', 'created_at', 'updated_at',
        ]
        read_only_fields = [
            'scan_id', 'annotated_image', 'inference_time_ms',
            'model_version', 'status', 'report_pdf',
        ]

    def get_patient_name(self, obj):
        return obj.patient.full_name

    def get_original_image_url(self, obj):
        request = self.context.get('request')
        if obj.original_image and request:
            return request.build_absolute_uri(obj.original_image.url)
        return None

    def get_annotated_image_url(self, obj):
        request = self.context.get('request')
        if obj.annotated_image and request:
            return request.build_absolute_uri(obj.annotated_image.url)
        return None


class ScanListSerializer(serializers.ModelSerializer):
    """Lightweight scan data for list/table views (no nested results)."""

    disease_count    = serializers.ReadOnlyField()
    highest_severity = serializers.ReadOnlyField()
    patient_name     = serializers.SerializerMethodField()
    patient_id_code  = serializers.SerializerMethodField()

    class Meta:
        model  = Scan
        fields = [
            'id', 'scan_id', 'patient', 'patient_name', 'patient_id_code',
            'scan_type', 'status', 'disease_count', 'highest_severity',
            'inference_time_ms', 'created_at',
        ]

    def get_patient_name(self, obj):
        return obj.patient.full_name

    def get_patient_id_code(self, obj):
        return obj.patient.patient_id


class PatientSerializer(serializers.ModelSerializer):
    """Full patient record with scan summary stats."""

    scan_count   = serializers.SerializerMethodField()
    last_scan_at = serializers.SerializerMethodField()
    full_name    = serializers.ReadOnlyField()

    class Meta:
        model  = Patient
        fields = [
            'id', 'patient_id', 'full_name', 'first_name', 'last_name',
            'age', 'gender', 'email', 'phone',
            'scan_count', 'last_scan_at', 'created_at',
        ]
        read_only_fields = ['patient_id']

    def get_scan_count(self, obj):
        return obj.scans.count()

    def get_last_scan_at(self, obj):
        last = obj.scans.first()
        return last.created_at if last else None


class ScanUploadSerializer(serializers.Serializer):
    """
    Validates the scan upload request from the Upload & Scan page.
    Accepts patient details + image file for new patients,
    or patient_id for existing patients.
    """

    # Patient can be existing (by patient_id) or new (by name fields)
    patient_id_code = serializers.CharField(required=False, allow_blank=True,
                                            help_text="Existing patient ID (PT-YYYY-NNNN)")
    first_name      = serializers.CharField(required=False, max_length=100)
    last_name       = serializers.CharField(required=False, max_length=100)
    age             = serializers.IntegerField(required=False, min_value=1, max_value=120)
    gender          = serializers.ChoiceField(choices=['M', 'F', 'O'], required=False)

    scan_type       = serializers.ChoiceField(choices=['OPG', 'IOPA'])
    image           = serializers.ImageField()
    doctor_name     = serializers.CharField(required=False, allow_blank=True, max_length=150)
    doctor_notes    = serializers.CharField(required=False, allow_blank=True)

    def validate(self, data):
        """Ensure we have either an existing patient ID or full new patient details."""
        has_existing = bool(data.get('patient_id_code'))
        has_new      = bool(data.get('first_name') and data.get('last_name') and data.get('age'))

        if not has_existing and not has_new:
            raise serializers.ValidationError(
                "Provide either 'patient_id_code' for an existing patient, "
                "or 'first_name', 'last_name', and 'age' for a new patient."
            )
        return data


class DashboardStatsSerializer(serializers.Serializer):
    """Stats data for the Dashboard screen."""
    total_patients   = serializers.IntegerField()
    total_scans      = serializers.IntegerField()
    scans_today      = serializers.IntegerField()
    scans_this_month = serializers.IntegerField()
    pending_scans    = serializers.IntegerField()
    high_severity    = serializers.IntegerField()
    recent_scans     = ScanListSerializer(many=True)
