"""
DentAI - Main URL Configuration
Routes API calls and serves Stitch MCP frontend screens.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required

# Wrap template views with login_required so unauthenticated users are redirected to /login/
def protected_template(template_name):
    return login_required(
        TemplateView.as_view(template_name=template_name),
        login_url='/login/'
    )

from django.contrib.auth import logout
from django.shortcuts import redirect
from core import views as core_views

def logout_view(request):
    logout(request)
    return redirect('login')

urlpatterns = [
    # ── Django Admin ────────────────────────────────────────────────────────
    path('admin/', admin.site.urls),

    # ── Auth ────────────────────────────────────────────────────────────────
    path('login/',      auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('register/',   core_views.register_view,                                 name='register'),
    path('logout/',     logout_view,                                              name='logout'),

    # ── REST API routes ─────────────────────────────────────────────────────
    path('api/', include('core.urls')),

    # ── Frontend Pages (protected — require login) ──────────────────────────
    path('',           TemplateView.as_view(template_name='home.html'),           name='home-page'),
    path('dashboard/', protected_template('dashboard.html'),      name='dashboard-page'),
    path('upload/',    protected_template('upload_scan.html'),    name='upload-page'),
    path('results/',   protected_template('results.html'),        name='results-page'),
    path('history/',   protected_template('patient_history.html'),name='history-page'),
    path('reports/',   protected_template('report.html'),         name='report-page'),
]

# Serve media files in all environments (for serverless/ephemeral /tmp uploads)
from django.views.static import serve
from django.urls import re_path

urlpatterns += [
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])

