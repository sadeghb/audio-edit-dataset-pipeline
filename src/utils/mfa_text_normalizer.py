# src/utils/mfa_text_normalizer.py
import re

def normalize_text_for_mfa(text: str) -> str:
    """
    Normalizes a text string to be compatible with the Montreal Forced Aligner.

    This pre-processing step is crucial for ensuring the highest possible
    alignment accuracy by cleaning the transcript to match the lexicon and
    acoustic model expectations.

    The process involves:
    1. Removing parenthetical content.
    2. Converting the text to uppercase.
    3. Removing all punctuation except apostrophes and digits.
    4. Collapsing multiple whitespace characters into a single space.

    Args:
        text: The input string, typically from a transcription service.

    Returns:
        A cleaned, MFA-compatible version of the text.
    """
    # Remove parenthetical content (e.g., [Music], (laughs))
    text = re.sub(r'\[[^\]]*\]|\([^)]*\)', '', text)

    # Convert to uppercase to match standard MFA dictionaries
    text = text.upper()

    # Remove all characters that are not letters, digits, apostrophes, or whitespace
    text = re.sub(r"[^A-Z0-9'\s]", '', text)

    # Collapse consecutive whitespace characters into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text
