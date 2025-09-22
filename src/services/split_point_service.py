# src/services/split_point_service.py
import logging

import pandas as pd

class SplitPointService:
    """
    Analyzes VAD timestamps to identify all points where the audio can be safely split.

    This service finds the midpoint of each silence between detected speech segments,
    creating a list of optimal split points for chunking.
    """
    def __init__(self):
        logging.info("SplitPointService initialized.")

    def run(self, vad_timestamps_df: pd.DataFrame, total_duration_ms: int) -> pd.DataFrame:
        """
        Generates a DataFrame of eligible split points from VAD timestamps.

        The logic is a three-step process:
        1.  The first split point is always at the beginning of the audio (0ms).
        2.  Intermediate points are the calculated midpoints of each silent gap
            between speech segments.
        3.  The last split point is always at the very end of the audio.

        Args:
            vad_timestamps_df: A DataFrame from VADService with 'start_ms' and 'end_ms'.
            total_duration_ms: The total duration of the source audio in milliseconds.

        Returns:
            A DataFrame with 'split_point_ms', 'silence_start_ms', and
            'silence_end_ms' columns.
        """
        if vad_timestamps_df.empty:
            return pd.DataFrame(columns=['split_point_ms', 'silence_start_ms', 'silence_end_ms'])

        split_points = []

        # 1. Add the first split point at the start of the audio.
        split_points.append({
            'split_point_ms': 0,
            'silence_start_ms': 0,
            'silence_end_ms': vad_timestamps_df.iloc[0]['start_ms']
        })

        # 2. Find the midpoint of each silence between speech segments.
        for i in range(len(vad_timestamps_df) - 1):
            silence_start = vad_timestamps_df.iloc[i]['end_ms']
            silence_end = vad_timestamps_df.iloc[i+1]['start_ms']
            mid_point = silence_start + (silence_end - silence_start) / 2
            
            split_points.append({
                'split_point_ms': int(mid_point),
                'silence_start_ms': silence_start,
                'silence_end_ms': silence_end
            })

        # 3. Add the final split point at the end of the audio.
        split_points.append({
            'split_point_ms': total_duration_ms,
            'silence_start_ms': vad_timestamps_df.iloc[-1]['end_ms'],
            'silence_end_ms': total_duration_ms
        })

        return pd.DataFrame(split_points)
