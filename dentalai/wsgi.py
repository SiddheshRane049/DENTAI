"""
WSGI config for dentalai project.
Exposes the WSGI callable as a module-level variable named ``application``.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dentalai.settings')
application = get_wsgi_application()

# Auto-migrate and collectstatic on Vercel cold starts only
if os.environ.get('VERCEL') == '1':
    from django.core.management import call_command
    try:
        call_command('migrate', interactive=False)
    except Exception as e:
        print(f"Auto-migration error: {e}")
    try:
        call_command('collectstatic', interactive=False, clear=True)
    except Exception as e:
        print(f"Collectstatic error: {e}")

app = application
