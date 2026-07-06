"""
DentAI - Core Database Models
Defines: Patient, Scan, DetectionResult, ScanComparison
"""

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
import uuid

class DoctorProfile(models.Model):
    """Profile model to store additional information about a Doctor user."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    designation = models.CharField(max_length=100, default='Dental Surgeon')

    def __str__(self):
        return f"{self.user.username} - {self.designation}"

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        DoctorProfile.objects.get_or_create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'profile'):
        instance.profile.save()


class Patient(models.Model):
    """Represents a dental patient in the system."""

    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    # Unique patient identifier (e.g., PT-2024-0001)
    patient_id = models.CharField(max_length=20, unique=True, editable=False)
    doctor      = models.ForeignKey(User, on_delete=models.CASCADE, related_name='patients', null=True, blank=True)
    first_name  = models.CharField(max_length=100)
    last_name   = models.CharField(max_length=100)
    age         = models.PositiveIntegerField()
    gender      = models.CharField(max_length=1, choices=GENDER_CHOICES, default='M')
    email       = models.EmailField(blank=True, null=True)
    phone       = models.CharField(max_length=20, blank=True, null=True)
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'

    def save(self, *args, **kwargs):
        """Auto-generate patient_id before first save."""
        if not self.patient_id:
            from django.db.models import Max
            year = timezone.now().year
            prefix = f"PT-{year}-"
            last = Patient.objects.filter(
                patient_id__startswith=prefix
            ).aggregate(max_id=Max('patient_id'))['max_id']
            if last:
                seq = int(last.split('-')[-1]) + 1
            else:
                seq = 1
            self.patient_id = f"{prefix}{seq:04d}"
        super().save(*args, **kwargs)

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def __str__(self):
        return f"{self.patient_id} - {self.full_name}"


def scan_image_upload_path(instance, filename):
    """Organizes uploads by patient ID and date."""
    date_str = timezone.now().strftime('%Y/%m/%d')
    return f"scans/{instance.patient.patient_id}/{date_str}/{filename}"


def annotated_image_upload_path(instance, filename):
    """Organizes annotated images by patient ID."""
    date_str = timezone.now().strftime('%Y/%m/%d')
    return f"annotated/{instance.patient.patient_id}/{date_str}/{filename}"


class Scan(models.Model):
    """Represents an X-ray scan uploaded for analysis."""

    SCAN_TYPE_CHOICES = [
        ('OPG',  'OPG (Orthopantomogram)'),
        ('IOPA', 'IOPA (Intraoral Periapical)'),
    ]

    STATUS_CHOICES = [
        ('pending',    'Pending'),
        ('processing', 'Processing'),
        ('completed',  'Completed'),
        ('failed',     'Failed'),
        ('reviewed',   'Reviewed'),
        ('flagged',    'Flagged'),
    ]

    # Unique scan identifier
    scan_id          = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    patient          = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='scans')
    scan_type        = models.CharField(max_length=4, choices=SCAN_TYPE_CHOICES)
    original_image   = models.ImageField(upload_to=scan_image_upload_path)
    annotated_image  = models.ImageField(upload_to=annotated_image_upload_path, blank=True, null=True)
    status           = models.CharField(max_length=12, choices=STATUS_CHOICES, default='pending', db_index=True)

    # Inference metadata
    inference_time_ms = models.FloatField(null=True, blank=True, help_text="Model inference time in milliseconds")
    model_version     = models.CharField(max_length=50, default='YOLOv8-dental-v1')
    doctor_name       = models.CharField(max_length=150, blank=True, null=True)
    doctor_notes      = models.TextField(blank=True, null=True)
    report_pdf        = models.FileField(upload_to='reports/', blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Scan'
        verbose_name_plural = 'Scans'

    def __str__(self):
        return f"Scan [{self.scan_type}] - {self.patient.full_name} - {self.created_at.strftime('%Y-%m-%d')}"

    @property
    def disease_count(self):
        """Count of detected diseases (excluding 'Healthy' findings)."""
        # Compute in-memory if results are prefetched to avoid N+1 queries
        return sum(1 for r in self.results.all() if r.disease_name != 'Healthy')

    @property
    def highest_severity(self):
        """Returns the highest severity found across all results."""
        # Compute in-memory if results are prefetched to avoid N+1 queries
        severities = [r.severity for r in self.results.all()]
        if 'high' in severities:
            return 'high'
        if 'medium' in severities:
            return 'medium'
        return 'low'


class DetectionResult(models.Model):
    """Stores a single disease detection finding from YOLOv8 inference."""

    SEVERITY_CHOICES = [
        ('low',    'Low'),
        ('medium', 'Medium'),
        ('high',   'High'),
    ]

    scan            = models.ForeignKey(Scan, on_delete=models.CASCADE, related_name='results')
    disease_name    = models.CharField(max_length=100)
    confidence      = models.FloatField(help_text="Confidence score 0.0 to 1.0")
    severity        = models.CharField(max_length=6, choices=SEVERITY_CHOICES, default='low', db_index=True)

    # Bounding box coordinates (normalized 0-1)
    bbox_x1 = models.FloatField(default=0)
    bbox_y1 = models.FloatField(default=0)
    bbox_x2 = models.FloatField(default=0)
    bbox_y2 = models.FloatField(default=0)

    # FDI tooth number (e.g., 11, 22, 36) — null if region-level detection
    fdi_tooth_number = models.IntegerField(null=True, blank=True, help_text="FDI notation tooth number")

    # Landmark points as JSON string: [{"x": 0.5, "y": 0.3, "label": "root_tip"}, ...]
    landmarks = models.JSONField(default=list, blank=True)

    # ── Restoration cross-verification flags (added v2) ────────────────────
    filling_present     = models.BooleanField(default=False, help_text="Filling detected on same tooth region")
    crown_present       = models.BooleanField(default=False, help_text="Crown detected on same tooth region")
    disease_under_crown = models.BooleanField(default=False, help_text="Disease detected underneath a crown")
    secondary_caries    = models.BooleanField(default=False, help_text="Secondary caries detected around a filling")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-confidence']
        verbose_name = 'Detection Result'
        verbose_name_plural = 'Detection Results'

    def __str__(self):
        tooth = f" (T{self.fdi_tooth_number})" if self.fdi_tooth_number else ""
        return f"{self.disease_name}{tooth} - {self.confidence:.0%} confidence"

    @property
    def confidence_percent(self):
        return round(self.confidence * 100, 1)

    @property
    def severity_emoji(self):
        return {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(self.severity, '⚪')


class ScanComparison(models.Model):
    """Links two scans for side-by-side historical comparison."""

    patient     = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='comparisons')
    scan_before = models.ForeignKey(Scan, on_delete=models.CASCADE, related_name='compared_as_before')
    scan_after  = models.ForeignKey(Scan, on_delete=models.CASCADE, related_name='compared_as_after')
    notes       = models.TextField(blank=True, null=True)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Scan Comparison'

    def __str__(self):
        return f"Comparison: {self.scan_before} vs {self.scan_after}"
