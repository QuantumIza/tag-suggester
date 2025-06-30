import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.cleaning import clean_text

def test_clean_text():
    assert clean_text("Bonjour <b>le</b> monde !") == "bonjour le monde "
