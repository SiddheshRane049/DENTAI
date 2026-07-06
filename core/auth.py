# This module previously contained CsrfExemptSessionAuthentication.
# CSRF protection is now properly enforced via session cookies.
# Frontend fetch() calls include the X-CSRFToken header.
