# src/services/mfa_chunker_service.py
import logging
from typing import Any, Dict, List

import pandas as pd

class MfaChunkerService:
    """
    Creates audio chunks specifically for processing by Montreal Forced Aligner (MFA).
    
    This service uses scribe's word timestamps and VAD-based split points to
    create chunks that end in silences, which is optimal for alignment.
    """
    def __init__(self):
        logging.info("MfaChunkerService initialized.")

    def _find_word_at_time(self, scribe_data: Dict[str, Any], time_s: float) -> Dict[str, Any] | None:
        """Helper to find the word/spacing object active at a specific timestamp."""
        for w in scribe_data.get("words", []):
            if w["start"] <= time_s <= w["end"]:
                return w
        return None

    def run(self, split_points_df: pd.DataFrame, scribe_data: Dict[str, Any], total_duration_s: float, min_duration_ms: int = 1000) -> List[Dict[str, Any]]:
        """
        Generates MFA-ready chunks by splitting the audio at silent points.

        Args:
            split_points_df: DataFrame of eligible split points from SplitPointService.
            scribe_data: The full, normalized Scribe transcription data.
            total_duration_s: The total duration of the source audio in seconds.
            min_duration_ms: The minimum duration for a chunk to be considered valid.

        Returns:
            A list of chunk dictionaries, each ready for MFA processing.
        """
        logging.info("Executing MFA Chunker Service...")
        mfa_chunks: List[Dict[str, Any]] = []
        eligible_split_points_s = (split_points_df["split_point_ms"] / 1000.0).tolist()
        if 0 < total_duration_s and total_duration_s not in eligible_split_points_s:
            eligible_split_points_s.append(total_duration_s)
        eligible_split_points_s.sort()

        current_start_s = 0.0
        while current_start_s < total_duration_s:
            found_chunk_end = False
            for split_point_s in eligible_split_points_s:
                if split_point_s <= current_start_s: continue
                if (split_point_s - current_start_s) * 1000 < min_duration_ms and split_point_s != total_duration_s: continue

                word_at_split = self._find_word_at_time(scribe_data, split_point_s)
                is_last_point = split_point_s >= total_duration_s

                # A valid chunk must end on a silent (spacing) segment or at the very end of the file.
                if (word_at_split and word_at_split.get("type") == "spacing") or is_last_point:
                    current_end_s = split_point_s
                    chunk_scribe_words = [w for w in scribe_data["words"] if current_start_s <= w["start"] < current_end_s]
                    transcript_parts = [w["text"] for w in chunk_scribe_words if w.get("type") == "word"]

                    if not transcript_parts:
                        current_start_s = current_end_s
                        found_chunk_end = True
                        break

                    mfa_chunks.append({
                        "id": len(mfa_chunks),
                        "start_s": current_start_s, "end_s": current_end_s,
                        "transcript": " ".join(transcript_parts),
                        "scribe_words": chunk_scribe_words,
                        "contains_audio_event": any(w.get("type") == "audio_event" for w in chunk_scribe_words),
                    })
                    current_start_s = current_end_s
                    found_chunk_end = True
                    break
            if not found_chunk_end: break

        logging.info("Defined %d chunks for MFA.", len(mfa_chunks))
        return mfa_chunks
