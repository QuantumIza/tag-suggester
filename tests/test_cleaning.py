import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.text_cleaning import clean_text_spacy_custom_2

def test_clean_text():
    assert clean_text_spacy_custom_2("Bonjour <b>le</b> monde !") == "bonjour le monde"
