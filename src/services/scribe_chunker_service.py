# src/services/scribe_chunker_service.py
import logging

import pandas as pd

class ScribeChunkerService:
    """
    Creates audio chunks for transcription that are as long as possible
    without exceeding a specified maximum duration.
    """
    def __init__(self, max_duration_ms: int = 475000):
        """
        Initializes the chunker with a maximum duration.

        Args:
            max_duration_ms: The maximum duration of any single chunk in milliseconds,
                             often determined by transcription API limits.
        """
        logging.info("ScribeChunkerService initialized.")
        self.max_duration_ms = max_duration_ms

    def run(self, split_points_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates chunk timestamps using a greedy algorithm.

        This method iterates through the available split points (silences) and
        creates the longest possible chunks that adhere to the max_duration_ms
        constraint. This is an efficient way to minimize the number of API calls
        while respecting service limits.

        Args:
            split_points_df: A DataFrame from SplitPointService.

        Returns:
            A DataFrame with 'chunk_start_ms' and 'chunk_end_ms' for each chunk.
        """
        if split_points_df.empty:
            return pd.DataFrame(columns=['chunk_start_ms', 'chunk_end_ms'])

        chunks = []
        split_points = split_points_df['split_point_ms'].tolist()
        
        current_chunk_start_index = 0
        while current_chunk_start_index < len(split_points) - 1:
            start_time = split_points[current_chunk_start_index]
            end_index = current_chunk_start_index
            
            # Look ahead to find the furthest split point within the max duration.
            for i in range(current_chunk_start_index + 1, len(split_points)):
                if split_points[i] - start_time <= self.max_duration_ms:
                    end_index = i
                else:
                    break  # This split point is too far; stop looking.

            # If a single VAD segment is longer than the max duration, force progress.
            if end_index == current_chunk_start_index:
                end_index += 1

            end_time = split_points[end_index]
            chunks.append({
                'chunk_start_ms': start_time,
                'chunk_end_ms': end_time
            })
            
            # The next chunk begins where the last one ended.
            current_chunk_start_index = end_index

        return pd.DataFrame(chunks)
