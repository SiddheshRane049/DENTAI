from rest_framework.authentication import SessionAuthentication

class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    Custom SessionAuthentication that bypasses CSRF checks.
    Useful for simplified AJAX calls in SPAs / dynamic HTML frontends.
    """
    def enforce_csrf(self, request):
        return  # Bypass CSRF validation
