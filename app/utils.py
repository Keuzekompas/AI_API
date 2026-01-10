import bleach

# Allowed HTML tags
ALLOWED_TAGS = []

def sanitize_text(text: str) -> str:
    """
    Cleans text using Bleach (removes HTML/Scripts).
    """
    if not text:
        return ""
    
    # If it's not a string (e.g., int), convert to string or return as is
    if not isinstance(text, str):
        return str(text)

    # Bleach clean removes dangerous tags
    clean = bleach.clean(text, tags=ALLOWED_TAGS, strip=True)
    return clean.strip()

def sanitize_recursive(data):
    """
    Recursively loops through data (dicts/lists) 
    and cleans all strings.
    """
    if isinstance(data, str):
        return sanitize_text(data)
    elif isinstance(data, list):
        return [sanitize_recursive(item) for item in data]
    elif isinstance(data, dict):
        return {key: sanitize_recursive(value) for key, value in data.items()}
    return data