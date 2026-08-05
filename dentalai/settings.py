"""
DentAI - Dental Disease Detection System
Django Settings Configuration
"""

from pathlib import Path
import os
import dj_database_url

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local .env file if it exists for local environment testing
_env_path = BASE_DIR / '.env'
if _env_path.exists():
    with open(_env_path, 'r', encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ[_k.strip()] = _v.strip()

# SECURITY
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-dentai-secret-key-change-in-production-2024')
DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 'yes')
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '*').split(',')

# ─── Installed Applications ───────────────────────────────────────────────────
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third-party
    'rest_framework',
    'corsheaders',
    # Local
    'core',
]

# ─── Middleware ────────────────────────────────────────────────────────────────
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',           # Must be first
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.gzip.GZipMiddleware',
    'django.middleware.http.ConditionalGetMiddleware',
]

ROOT_URLCONF = 'dentalai.urls'

# ─── Templates ────────────────────────────────────────────────────────────────
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],       # Stitch MCP generated HTML goes here
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'dentalai.wsgi.application'

# ─── Database ───────────────────────────────────────────────────────────────────
# Uses DATABASE_URL env var (Neon PostgreSQL on Vercel) with SQLite fallback for local dev.
_DATABASE_URL = os.environ.get('DATABASE_URL')
if _DATABASE_URL:
    DATABASES = {'default': dj_database_url.config(default=_DATABASE_URL, conn_max_age=600, ssl_require=True)}
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# ─── Password Validators ─────────────────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ─── Authentication Redirects ──────────────────────────────────────────────────
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/dashboard/'
LOGOUT_REDIRECT_URL = '/login/'

# ─── Internationalization ─────────────────────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

# ─── Static & Media Files ─────────────────────────────────────────────────────
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
if os.environ.get('VERCEL') == '1':
    STATIC_ROOT = '/tmp/staticfiles'
else:
    STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedStaticFilesStorage'

MEDIA_URL = '/media/'
if os.environ.get('VERCEL') == '1':
    MEDIA_ROOT = '/tmp/media'
else:
    MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ─── CORS Configuration ───────────────────────────────────────────────────────
CORS_ALLOW_ALL_ORIGINS = DEBUG  # Only allow all origins in development
CORS_ALLOW_CREDENTIALS = True
if not DEBUG:
    CORS_ALLOWED_ORIGINS = [
        origin.strip() for origin in
        os.environ.get('CORS_ALLOWED_ORIGINS', 'https://dentai.vercel.app').split(',')
    ]

# ─── Django REST Framework ────────────────────────────────────────────────────
REST_FRAMEWORK = {
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',   # For image uploads
        'rest_framework.parsers.FormParser',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

# ─── ML Model Configuration ───────────────────────────────────────────────────
USE_NEW_MODEL_ONLY = False          # Run concurrent multi-model inference
ENABLE_SIMULTANEOUS_MODELS = True  # Enable concurrent execution of all active YOLO models
YOLO_MODEL_PATH = BASE_DIR / 'models' / 'runs_dentai_weights_best.pt'
YOLO_FINAL_MODEL_PATH = BASE_DIR / 'models' / 'DentAI_Final.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.38   # Minimum confidence for high-accuracy detections

# ─── Dental Disease Classes (FDI + Disease Mapping) ──────────────────────────
DENTAL_DISEASE_CLASSES = {
    0:  {'name': 'Occlusal Caries',           'severity': 'medium', 'color': (255, 165, 0)},
    1:  {'name': 'Proximal Caries',           'severity': 'medium', 'color': (255, 140, 0)},
    2:  {'name': 'Periapical Abscess',        'severity': 'high',   'color': (220, 50,  50)},
    3:  {'name': 'Periapical Cyst',           'severity': 'high',   'color': (180, 0,   0)},
    4:  {'name': 'Granuloma',                 'severity': 'medium', 'color': (255, 100, 0)},
    5:  {'name': 'Apical Periodontitis',      'severity': 'medium', 'color': (200, 80,  0)},
    6:  {'name': 'Horizontal Bone Loss',      'severity': 'high',   'color': (150, 0,  150)},
    7:  {'name': 'Vertical Bone Loss',        'severity': 'high',   'color': (130, 0,  200)},
    8:  {'name': 'Root Canal Treated',        'severity': 'low',    'color': (200, 0,   50)},
    9:  {'name': 'Milk Tooth',                'severity': 'low',    'color': (0,  180, 100)},
    10: {'name': 'Healthy',                   'severity': 'low',    'color': (0,  200,   0)},
}

# ─── Email Configuration ──────────────────────────────────────────────────────
# Uses SMTP if user credentials are provided (on Vercel or locally via .env), otherwise falls back to console.
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', 'DentAI System <no-reply@dentalai.com>')

if not EMAIL_HOST_USER or not EMAIL_HOST_PASSWORD:
    EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
    DEFAULT_FROM_EMAIL = 'DentAI System <no-reply@dentalai.com>'

# ─── Production Security Headers ──────────────────────────────────────────────
if not DEBUG and os.environ.get('VERCEL'):
    SECURE_SSL_REDIRECT = False  # Vercel edge handles SSL redirection
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
else:
    SECURE_SSL_REDIRECT = False
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SECURE = False

    CSRF_TRUSTED_ORIGINS = [
        origin.strip() for origin in
        os.environ.get('CSRF_TRUSTED_ORIGINS', 'https://dentai.vercel.app').split(',')
    ]

