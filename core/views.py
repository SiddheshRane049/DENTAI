"""
DentAI - Django REST API Views
All API endpoints for the frontend to consume.

Endpoints summary:
  GET  /api/dashboard/            → Dashboard stats + recent scans
  POST /api/scans/upload/         → Upload X-ray, run YOLOv8 detection
  GET  /api/scans/                → List all scans (paginated)
  GET  /api/scans/<id>/           → Single scan detail with results
  GET  /api/patients/             → List all patients
  POST /api/patients/             → Create patient
  GET  /api/patients/<id>/scans/  → Patient scan history
  GET  /api/scans/<id>/report/    → Download PDF report
  POST /api/scans/<id>/report/    → Generate PDF report
"""

import os
import logging
import threading
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from django.http import FileResponse, Http404
from django.db.models import Count, Q
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from .models import Patient, Scan, DetectionResult
from .serializers import (
    PatientSerializer, ScanSerializer, ScanListSerializer,
    ScanUploadSerializer, DashboardStatsSerializer,
)

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.core.mail import send_mail
import random
import time

logger = logging.getLogger(__name__)


# ─── Doctor Auth Views ────────────────────────────────────────────────────────

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard-page')

    if request.method == 'POST':
        full_name = request.POST.get('full_name', '').strip()
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')

        if not full_name or not username or not email or not password:
            return render(request, 'register.html', {'error': 'All fields are required.', 'data': request.POST})

        if password != confirm_password:
            return render(request, 'register.html', {'error': 'Passwords do not match.', 'data': request.POST})

        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error': 'Username already exists.', 'data': request.POST})

        if User.objects.filter(email=email).exists():
            return render(request, 'register.html', {'error': 'Email address already registered.', 'data': request.POST})

        try:
            parts = full_name.split(' ', 1)
            first_name = parts[0]
            last_name = parts[1] if len(parts) > 1 else ''

            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )

            try:
                subject = 'Welcome to DentAI!'
                message = f"Hello Dr. {full_name},\n\nWelcome to DentAI Diagnostic System! Your doctor account has been successfully created.\n\nUsername: {username}\nEmail: {email}\n\nYou can now log in to upload scans, run diagnostics, and generate patient reports.\n\nBest Regards,\nDentAI Team"
                send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [email], fail_silently=True)
            except Exception as e:
                logger.error(f"Failed to send welcome email to {email}: {e}")

            login(request, user)
            return redirect('dashboard-page')
        except Exception as e:
            return render(request, 'register.html', {'error': f'Registration failed: {e}', 'data': request.POST})

    return render(request, 'register.html')
# ─── Dashboard ────────────────────────────────────────────────────────────────

class DashboardView(APIView):
    """
    GET /api/dashboard/
    Returns aggregate stats and recent scans for the Dashboard screen.
    """

    def get(self, request):
        today = timezone.now().date()
        month_start = today.replace(day=1)

        # Get querysets isolated to the logged-in doctor
        patients_qs = Patient.objects.filter(doctor=request.user)
        scans_qs = Scan.objects.filter(patient__doctor=request.user)

        stats = {
            'total_patients':   patients_qs.count(),
            'total_scans':      scans_qs.count(),
            'scans_today':      scans_qs.filter(created_at__date=today).count(),
            'scans_this_month': scans_qs.filter(created_at__date__gte=month_start).count(),
            'pending_scans':    scans_qs.filter(status='pending').count(),
            'high_severity':    scans_qs.filter(results__severity='high').distinct().count(),
            'recent_scans': ScanListSerializer(
                scans_qs.select_related('patient').order_by('-created_at')[:8],
                many=True,
                context={'request': request},
            ).data,
        }

        return Response(stats, status=status.HTTP_200_OK)


# ─── Scan Upload & Detection ──────────────────────────────────────────────────

class ScanUploadView(APIView):
    """
    POST /api/scans/upload/
    Accepts X-ray image + patient info → runs YOLOv8 detection asynchronously.

    Form fields:
      image          (file)  - X-ray image (JPEG/PNG)
      scan_type      (str)   - 'OPG' or 'IOPA'
      patient_id_code(str)   - Existing patient ID   -OR-
      first_name     (str)   - New patient first name
      last_name      (str)   - New patient last name
      age            (int)   - Patient age
      gender         (str)   - 'M'/'F'/'O'
      doctor_notes   (str)   - Optional doctor notes
    """

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = ScanUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        # ── Resolve or create Patient ──────────────────────────────────────
        patient = self._get_or_create_patient(data, request)
        if isinstance(patient, Response):   # Error response
            return patient

        # ── Create Scan record ─────────────────────────────────────────────
        doctor_name_default = f"Dr. {request.user.first_name} {request.user.last_name}"
        scan = Scan.objects.create(
            patient       = patient,
            scan_type     = data['scan_type'],
            original_image= data['image'],
            doctor_name   = data.get('doctor_name') or doctor_name_default,
            doctor_notes  = data.get('doctor_notes', ''),
            status        = 'processing',
        )

        # ── Run detection asynchronously (or synchronously on Vercel) ──────
        if os.environ.get('VERCEL') == '1':
            self._run_detection_async(scan.id)
            scan.refresh_from_db()
        else:
            thread = threading.Thread(
                target=self._run_detection_async,
                args=(scan.id,),
                daemon=True,
            )
            thread.start()

        return Response(
            {
                'scan_id':  str(scan.scan_id),
                'id':       scan.id,
                'status':   scan.status,
                'message':  'Scan uploaded. Detection completed.' if os.environ.get('VERCEL') == '1' else 'Scan uploaded. Detection running...',
                'patient':  PatientSerializer(patient).data,
            },
            status=status.HTTP_202_ACCEPTED,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_or_create_patient(self, data, request):
        """Resolve existing patient by ID or create a new one."""
        pid_code = data.get('patient_id_code')
        if pid_code:
            try:
                return Patient.objects.get(patient_id=pid_code, doctor=request.user)
            except Patient.DoesNotExist:
                return Response(
                    {'error': f"Patient '{pid_code}' not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

        # Create new patient
        return Patient.objects.create(
            first_name=data['first_name'],
            last_name =data['last_name'],
            age       =data['age'],
            gender    =data.get('gender', 'M'),
            doctor    =request.user,
        )

    def _run_detection_async(self, scan_id: int):
        """
        Background thread: runs YOLOv8 inference, draws annotations,
        saves DetectionResult records, updates Scan status.
        """
        from .ml.detector  import DentalDetector
        from .ml.landmarks import draw_annotations, draw_opg_dental_chart

        try:
            scan = Scan.objects.get(id=scan_id)

            # ── Run inference ──────────────────────────────────────────────
            detector = DentalDetector()
            image_path = os.path.join(settings.MEDIA_ROOT, scan.original_image.name)
            result = detector.detect(image_path)

            detections = result['detections']

            # ── Draw annotations ──────────────────────────────────────────
            annotated_name = f"annotated/{scan.patient.patient_id}/{scan.scan_id}_annotated.jpg"
            annotated_path = os.path.join(settings.MEDIA_ROOT, annotated_name)

            draw_annotations(image_path, detections, annotated_path)

            # For OPG: additionally draw the FDI dental chart overlay
            if scan.scan_type == 'OPG' and detections:
                chart_name = f"annotated/{scan.patient.patient_id}/{scan.scan_id}_chart.jpg"
                chart_path = os.path.join(settings.MEDIA_ROOT, chart_name)
                draw_opg_dental_chart(image_path, detections, chart_path)

            # ── Save DetectionResult records ──────────────────────────────
            for det in detections:
                DetectionResult.objects.create(
                    scan                = scan,
                    disease_name        = det['disease_name'],
                    confidence          = det['confidence'],
                    severity            = det['severity'],
                    bbox_x1             = det['bbox']['x1'],
                    bbox_y1             = det['bbox']['y1'],
                    bbox_x2             = det['bbox']['x2'],
                    bbox_y2             = det['bbox']['y2'],
                    fdi_tooth_number    = det.get('fdi_tooth_number'),
                    landmarks           = det.get('landmarks', []),
                    filling_present     = det.get('filling_present', False),
                    crown_present       = det.get('crown_present', False),
                    disease_under_crown = det.get('disease_under_crown', False),
                    secondary_caries    = det.get('secondary_caries', False),
                )

            # ── Update scan record ────────────────────────────────────────
            scan.annotated_image  = annotated_name
            scan.inference_time_ms= result['inference_time_ms']
            scan.model_version    = result['model_version']
            scan.status           = 'completed'
            scan.save()

            logger.info(f"Scan {scan_id} detection completed. {len(detections)} findings.")

        except Exception as e:
            logger.error(f"Detection failed for scan {scan_id}: {e}", exc_info=True)
            Scan.objects.filter(id=scan_id).update(status='failed')

def _get_scan_by_pk_or_uuid(pk, request, select_related_patient=False, prefetch_results=False):
    import uuid
    from django.http import Http404
    from .models import Scan

    qs = Scan.objects.filter(patient__doctor=request.user)
    if select_related_patient:
        qs = qs.select_related('patient')
    if prefetch_results:
        qs = qs.prefetch_related('results')

    try:
        val = uuid.UUID(str(pk))
        return qs.get(scan_id=val)
    except (ValueError, TypeError, Scan.DoesNotExist):
        try:
            return qs.get(pk=int(pk))
        except (ValueError, TypeError, Scan.DoesNotExist):
            raise Http404("Scan not found")


# ─── Scan Status Polling ──────────────────────────────────────────────────────

class ScanStatusView(APIView):
    """
    GET /api/scans/<id>/status/
    Frontend polls this endpoint to know when detection is done.
    """

    def get(self, request, pk):
        scan = _get_scan_by_pk_or_uuid(pk, request)
        return Response({'status': scan.status, 'scan_id': str(scan.scan_id)})


# ─── Scan Detail ─────────────────────────────────────────────────────────────

class ScanDetailView(generics.RetrieveAPIView):
    """
    GET /api/scans/<id>/
    Full scan details with all detection results (for Results page).
    """
    serializer_class = ScanSerializer

    def get_object(self):
        pk = self.kwargs.get('pk')
        return _get_scan_by_pk_or_uuid(pk, self.request, select_related_patient=True, prefetch_results=True)

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        ctx['request'] = self.request
        return ctx


class ScanListView(generics.ListAPIView):
    """
    GET /api/scans/
    Paginated list of all scans (for Patient History table).
    Query params: ?patient_id=&scan_type=&status=&search=
    """
    serializer_class = ScanListSerializer

    def get_queryset(self):
        qs = Scan.objects.filter(patient__doctor=self.request.user).select_related('patient').order_by('-created_at')

        # Filter by patient ID code
        pid = self.request.query_params.get('patient_id')
        if pid:
            qs = qs.filter(patient__patient_id__icontains=pid)

        # Filter by scan type
        scan_type = self.request.query_params.get('scan_type')
        if scan_type in ('OPG', 'IOPA'):
            qs = qs.filter(scan_type=scan_type)

        # Filter by status
        scan_status = self.request.query_params.get('status')
        if scan_status:
            qs = qs.filter(status=scan_status)

        # Search by patient name
        search = self.request.query_params.get('search')
        if search:
            qs = qs.filter(
                Q(patient__first_name__icontains=search) |
                Q(patient__last_name__icontains=search)  |
                Q(patient__patient_id__icontains=search)
            )

        return qs


# ─── Patient Views ────────────────────────────────────────────────────────────

class PatientListCreateView(generics.ListCreateAPIView):
    """
    GET  /api/patients/  → List all patients
    POST /api/patients/  → Create new patient
    """
    serializer_class = PatientSerializer

    def get_queryset(self):
        return Patient.objects.filter(doctor=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        serializer.save(doctor=self.request.user)


class PatientScanHistoryView(generics.ListAPIView):
    """
    GET /api/patients/<patient_id>/scans/
    All scans for a specific patient (Patient History screen).
    """
    serializer_class = ScanListSerializer

    def get_queryset(self):
        return Scan.objects.filter(
            patient__doctor=self.request.user,
            patient__patient_id=self.kwargs['patient_id']
        ).order_by('-created_at')


# ─── PDF Report Generation ────────────────────────────────────────────────────

class ReportGenerateView(APIView):
    """
    POST /api/scans/<id>/report/
    Generate a PDF diagnosis report for a scan.

    Body (JSON):
      include_annotated  (bool, default true)
      include_chart      (bool, default true)
      doctor_notes       (str, optional override)

    Returns:
      JSON with report_url once generated.
    """

    def post(self, request, pk):
        scan = _get_scan_by_pk_or_uuid(pk, request, prefetch_results=True)

        if scan.status != 'completed':
            return Response(
                {'error': 'Scan detection must be completed before generating a report.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Override doctor notes if provided
        notes = request.data.get('doctor_notes')
        if notes is not None:
            scan.doctor_notes = notes
            scan.save(update_fields=['doctor_notes'])

        try:
            from .report import generate_pdf_report
            pdf_path = generate_pdf_report(scan, request)

            # Save report path on scan
            report_rel = os.path.relpath(pdf_path, settings.MEDIA_ROOT)
            scan.report_pdf = report_rel
            scan.save(update_fields=['report_pdf'])

            report_url = request.build_absolute_uri(
                settings.MEDIA_URL + report_rel.replace('\\', '/')
            )
            return Response({'report_url': report_url}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Report generation failed for scan {pk}: {e}", exc_info=True)
            return Response(
                {'error': f"Report generation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ReportDownloadView(APIView):
    """
    GET /api/scans/<id>/report/download/
    Stream the PDF report file to the browser.
    """

    def get(self, request, pk):
        scan = _get_scan_by_pk_or_uuid(pk, request, select_related_patient=True)

        if not scan.report_pdf:
            return Response(
                {'error': 'No report generated for this scan yet.'},
                status=status.HTTP_404_NOT_FOUND,
            )

        pdf_path = os.path.join(settings.MEDIA_ROOT, scan.report_pdf.name)
        if not os.path.exists(pdf_path):
            return Response(
                {'error': 'Report file not found on disk.'},
                status=status.HTTP_404_NOT_FOUND,
            )

        filename = f"DentAI_Report_{scan.patient.patient_id}_{scan.scan_id}.pdf"
        response = FileResponse(
            open(pdf_path, 'rb'),
            content_type='application/pdf',
        )
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

