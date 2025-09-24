import re
import string
from src.logger import logging

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        logging.warning("Non-string input to clean_text.")
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())
