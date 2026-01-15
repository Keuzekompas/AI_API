from app.utils import sanitize_text, sanitize_recursive

def test_sanitize_text_strips_html():
    """Unit test: Control if HTML and scripts are removed correctly."""
    dirty_html = "<script>alert('xss')</script>Hello <b>World</b>"
    clean_text = sanitize_text(dirty_html)
    assert clean_text == "alert('xss')Hello World"
    assert "<script>" not in clean_text

def test_sanitize_text_handles_non_strings():
    """Unit test: Control if numbers are correctly converted to strings."""
    assert sanitize_text(123) == "123"
    assert sanitize_text(None) == ""

def test_sanitize_recursive_list():
    """Unit test: Control if a list with dirty strings is recursively cleaned."""
    dirty_data = ["<b>Gevaar</b>", {"key": "<i>Schuin</i>"}]
    clean_data = sanitize_recursive(dirty_data)
    assert clean_data == ["Gevaar", {"key": "Schuin"}]