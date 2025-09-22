# src/services/audio_splitter_service.py
import logging
from pathlib import Path

import pandas as pd
from pydub import AudioSegment

class AudioSplitterService:
    """
    Splits a large audio file into multiple smaller chunks based on a
    DataFrame of start and end times.
    """
    def __init__(self):
        logging.info("AudioSplitterService initialized.")

    def run(self, audio: AudioSegment, chunks_df: pd.DataFrame, chunks_dir: Path, audio_name: str) -> list[Path]:
        """
        Splits the main audio into smaller chunks based on the provided DataFrame.

        Args:
            audio: The full audio file as a pydub AudioSegment.
            chunks_df: A pandas DataFrame with 'start_ms' and 'end_ms' columns
                       defining the boundaries of each chunk.
            chunks_dir: The directory where the output chunk files will be saved.
            audio_name: The base name for the output chunk files.

        Returns:
            A list of Path objects for each created audio chunk.
        """
        chunk_paths = []
        for i, row in chunks_df.iterrows():
            chunk_path = chunks_dir / f"{audio_name}_chunk_{i + 1}.wav"
            self.split_and_save_chunk(audio, row['start_ms'], row['end_ms'], chunk_path)
            chunk_paths.append(chunk_path)
            
        return chunk_paths

    def split_and_save_chunk(self, audio: AudioSegment, start_ms: float, end_ms: float, output_path: Path):
        """
        Extracts a single segment from an AudioSegment and saves it as a WAV file.

        Args:
            audio: The source pydub AudioSegment.
            start_ms: The start time of the chunk in milliseconds.
            end_ms: The end time of the chunk in milliseconds.
            output_path: The full path to save the output WAV file.
        """
        chunk = audio[start_ms:end_ms]
        chunk.export(output_path, format="wav")
