"""
DentAI - Main URL Configuration
Routes API calls and serves Stitch MCP frontend screens.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    # ── Django Admin ────────────────────────────────────────────────────────
    path('admin/', admin.site.urls),

    # ── REST API routes ─────────────────────────────────────────────────────
    path('api/', include('core.urls')),

    # ── Stitch MCP Frontend Screens (served as Django templates) ───────────
    path('',            TemplateView.as_view(template_name='home.html'),           name='home-page'),
    path('dashboard/',  TemplateView.as_view(template_name='dashboard.html'),      name='dashboard-page'),
    path('upload/',     TemplateView.as_view(template_name='upload_scan.html'),    name='upload-page'),
    path('results/',    TemplateView.as_view(template_name='results.html'),        name='results-page'),
    path('history/',    TemplateView.as_view(template_name='patient_history.html'),name='history-page'),
    path('reports/',    TemplateView.as_view(template_name='report.html'),         name='report-page'),
]

# Serve media files in all environments (for serverless/ephemeral /tmp uploads)
from django.views.static import serve
from django.urls import re_path

urlpatterns += [
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
