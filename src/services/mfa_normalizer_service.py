# src/services/mfa_normalizer_service.py
import logging
from pathlib import Path
from typing import Any, Dict, List

import textgrid

from src.utils.mfa_text_normalizer import normalize_text_for_mfa


def _edit_distance_leq(a: str, b: str, max_dist: int = 2) -> bool:
    """
    An efficient implementation to check if the Levenshtein distance between
    two strings is less than or equal to a max distance. It uses a banded
    dynamic programming matrix for an early exit if the distance exceeds the max.
    """
    if abs(len(a) - len(b)) > max_dist: return False
    if len(a) > len(b): a, b = b, a

    prev_row = list(range(len(a) + 1))
    for i, char_b in enumerate(b, 1):
        curr_row = [i] + [0] * len(a)
        for j in range(1, len(a) + 1):
            cost = 0 if a[j - 1] == char_b else 1
            curr_row[j] = min(prev_row[j] + 1, curr_row[j - 1] + 1, prev_row[j - 1] + cost)
        if min(curr_row) > max_dist: return False
        prev_row = curr_row
    return prev_row[len(a)] <= max_dist


class MfaNormalizerService:
    """
    Parses TextGrid files from an MFA run and normalizes the results into a
    single, structured JSON list with rich diagnostic information.
    """
    def __init__(self):
        logging.info("MfaNormalizerService initialized.")

    def run(self, mfa_output_dir: Path, mfa_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalizes all TextGrids and returns a flat list of aligned-word dictionaries.

        Args:
            mfa_output_dir: The directory containing the .TextGrid files from MFA.
            mfa_chunks: The list of chunk dictionaries, containing the original
                        Scribe words and validator info.

        Returns:
            A single, flat list of word dictionaries with detailed alignment and
            diagnostic data.
        """
        logging.info("Normalizing MFA TextGrid results...")
        all_words: List[Dict[str, Any]] = []
        chunk_map = {c["id"]: c for c in mfa_chunks}
        tg_files = sorted(mfa_output_dir.glob("*.TextGrid"), key=lambda p: int(p.stem.split("_")[-1]))

        for tg_file in tg_files:
            try:
                chunk_id = int(tg_file.stem.split("_")[-1])
                chunk_info = chunk_map.get(chunk_id)
                if not chunk_info: continue

                words = self._parse_textgrid(
                    tg_file,
                    offset_s=chunk_info["start_s"],
                    original_words=chunk_info["scribe_words"],
                    chunk_info=chunk_info,
                )
                all_words.extend(words)
            except (ValueError, IndexError):
                logging.warning(f"Could not parse chunk ID from filename: {tg_file.name}")
                continue
        
        logging.info("Successfully normalized %d words from MFA output.", len(all_words))
        return all_words

    def _parse_textgrid(self, tg_path: Path, offset_s: float, original_words: List[Dict], chunk_info: Dict) -> List[Dict]:
        """Parses a single TextGrid and maps aligned words to original Scribe words."""
        aligned_words = []
        try:
            tg = textgrid.TextGrid.fromFile(str(tg_path))
            word_tier = tg.getFirst("words")
            phone_tier = tg.getFirst("phones")
            if not word_tier: return []

            mfa_input_words = [w for w in original_words if w.get("type") == "word" and w.get("text") != "..."]
            original_idx, mismatch_found = 0, False
            mismatched_pairs = []

            for interval in word_tier:
                if not interval.mark or interval.mark.lower() in ("sp", "spn", "sil"): continue
                if original_idx >= len(mfa_input_words): break

                orig_word = mfa_input_words[original_idx]
                if not _edit_distance_leq(normalize_text_for_mfa(orig_word["text"]), normalize_text_for_mfa(interval.mark)):
                    mismatch_found = True
                    mismatched_pairs.append([orig_word["text"], interval.mark])

                word_data: Dict[str, Any] = {
                    "id": orig_word["id"], "word": orig_word["text"], "mfa_word": interval.mark,
                    "start": round(interval.minTime + offset_s, 4), "end": round(interval.maxTime + offset_s, 4), "phonemes": []
                }
                
                if phone_tier:
                    for p in phone_tier:
                        if p.minTime >= interval.minTime and p.maxTime <= interval.maxTime and p.mark:
                            word_data["phonemes"].append({"text": p.mark, "start": round(p.minTime + offset_s, 4), "end": round(p.maxTime + offset_s, 4)})
                
                aligned_words.append(word_data)
                original_idx += 1

            # Inject chunk-level diagnostic info into every word from that chunk.
            chunk_diag = { "id": chunk_info["id"], "contains_audio_event": chunk_info.get("contains_audio_event", False),
                           "contains_oov_words": chunk_info.get("contains_oov", False), "contains_mismatched_words": mismatch_found,
                           "oov_words": chunk_info.get("oov_words", []), "mismatched_pairs": mismatched_pairs }
            for w in aligned_words: w["chunk_info"] = chunk_diag
        
        except Exception:
            logging.error("Could not parse TextGrid file: %s", tg_path, exc_info=True)

        return aligned_words
