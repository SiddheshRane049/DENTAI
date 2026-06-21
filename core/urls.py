"""
DentAI - Core App URL Configuration
Maps all /api/ endpoints to their respective views.
"""

from django.urls import path
from .views import (
    DashboardView,
    ScanUploadView,
    ScanStatusView,
    ScanDetailView,
    ScanListView,
    PatientListCreateView,
    PatientScanHistoryView,
    ReportGenerateView,
    ReportDownloadView,
)

urlpatterns = [
    # ── Dashboard ─────────────────────────────────────────────────────
    path('dashboard/',                              DashboardView.as_view(),           name='api-dashboard'),

    # ── Scans ─────────────────────────────────────────────────────────
    path('scans/upload/',                           ScanUploadView.as_view(),          name='api-scan-upload'),
    path('scans/',                                  ScanListView.as_view(),            name='api-scan-list'),
    path('scans/<int:pk>/',                         ScanDetailView.as_view(),          name='api-scan-detail'),
    path('scans/<int:pk>/status/',                  ScanStatusView.as_view(),          name='api-scan-status'),

    # ── Reports ───────────────────────────────────────────────────────
    path('scans/<int:pk>/report/',                  ReportGenerateView.as_view(),      name='api-report-generate'),
    path('scans/<int:pk>/report/download/',         ReportDownloadView.as_view(),      name='api-report-download'),

    # ── Patients ──────────────────────────────────────────────────────
    path('patients/',                               PatientListCreateView.as_view(),   name='api-patient-list'),
    path('patients/<str:patient_id>/scans/',        PatientScanHistoryView.as_view(),  name='api-patient-scans'),
]
