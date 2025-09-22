# src/services/scribe_normalizer_service.py
import logging
from typing import Any, Dict, List

import pandas as pd

class ScribeNormalizerService:
    """
    Normalizes and combines multiple Scribe JSON results from audio chunks
    into a single, coherent transcript object.
    """
    def __init__(self):
        logging.info("ScribeNormalizerService initialized.")

    def run(self, scribe_results: List[Dict], chunk_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Normalizes timestamps and combines text from a list of chunked Scribe results.

        This is a crucial post-processing step. It takes the transcription results
        from individual audio chunks and correctly adjusts their word-level timestamps
        to be relative to the start of the original, full-length audio file.

        Args:
            scribe_results: A list of raw JSON dictionaries from the Scribe API.
            chunk_df: DataFrame with the 'chunk_start_ms' of each corresponding chunk.

        Returns:
            A single dictionary mirroring the Scribe output format, but with
            normalized timestamps and a complete text transcript.
        """
        master_transcript = {"words": []}
        full_text_parts = []

        for i, result in enumerate(scribe_results):
            # Get the start time of the chunk to use as an offset for all timestamps within it.
            chunk_start_offset_s = chunk_df.iloc[i]['chunk_start_ms'] / 1000.0
            
            full_text_parts.append(result.get('text', ''))

            for item in result.get('words', []):
                normalized_item = item.copy()
                
                # Adjust timestamps by adding the chunk's start time offset.
                normalized_item['start'] = round(item.get('start', 0) + chunk_start_offset_s, 3)
                normalized_item['end'] = round(item.get('end', 0) + chunk_start_offset_s, 3)
                
                master_transcript['words'].append(normalized_item)

        master_transcript['text'] = " ".join(full_text_parts)

        # Add a unique ID to each word for easier downstream processing.
        for i, word in enumerate(master_transcript['words']):
            word['id'] = i

        return master_transcript
