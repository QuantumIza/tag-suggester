import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # supprime les balises HTML
    text = re.sub(r"[^a-z0-9\s]", "", text)  # supprime la ponctuation
    return text
