import re

def sanitize_text(text: str) -> str:
    """
    Strips HTML tags and removes javascript protocol injections.
    """
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]*>', '', text)
    # Remove potential script injections
    clean = re.sub(r'javascript:', '', clean, flags=re.IGNORECASE)
    return clean.strip()
