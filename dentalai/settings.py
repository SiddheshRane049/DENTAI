"""
DentAI - Dental Disease Detection System
Django Settings Configuration
"""

from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY - Change in production!
SECRET_KEY = 'django-insecure-dentai-secret-key-change-in-production-2024'
DEBUG = True
ALLOWED_HOSTS = ['*']

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

# ─── Database: SQLite3 ────────────────────────────────────────────────────────
if os.environ.get('VERCEL') == '1':
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': '/tmp/db.sqlite3',
        }
    }
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
CORS_ALLOW_ALL_ORIGINS = True       # Tighten in production
CORS_ALLOW_CREDENTIALS = True

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
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

# ─── ML Model Configuration ───────────────────────────────────────────────────
# Path to the YOLOv8 weights file (fine-tuned on dental dataset)
# Place your trained model at: models/dental_yolov8.pt
# If not available, the app falls back to YOLOv8 nano pretrained (for demo)
YOLO_MODEL_PATH = BASE_DIR / 'models' / 'dental_yolov8.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.25   # Minimum confidence for detections

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
    8:  {'name': 'Root Canal Required',       'severity': 'high',   'color': (200, 0,   50)},
    9:  {'name': 'Milk Tooth',                'severity': 'low',    'color': (0,  180, 100)},
    10: {'name': 'Healthy',                   'severity': 'low',    'color': (0,  200,   0)},
}
