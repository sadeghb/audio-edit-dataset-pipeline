# src/services/vad_service.py
from pathlib import Path

import pandas as pd

from ..vad_processor import process_audio

class VADService:
    """
    A specialist service for performing Voice Activity Detection (VAD).

    This class acts as a clean, injectable wrapper around the core VAD processing
    logic. It receives the pre-loaded Silero VAD model as a dependency and provides
    a simple `run` method for the pipeline orchestrator to call.
    """
    def __init__(self, model, utils):
        """
        Initializes the VADService with a loaded Silero model and its utils.

        Args:
            model: The pre-loaded Silero VAD model object.
            utils: The utility functions loaded alongside the model from torch.hub.
        """
        self.model = model
        # Unpack the specific function we need from the utils tuple
        self.get_speech_timestamps = utils[0]

    def run(self, audio_path: Path) -> pd.DataFrame:
        """
        Runs the VAD processing on a given audio file.

        Args:
            audio_path: The path to the audio file to process.

        Returns:
            A pandas DataFrame with 'start_ms' and 'end_ms' for each speech segment.
        """
        return process_audio(
            audio_path=audio_path,
            model=self.model,
            get_speech_timestamps=self.get_speech_timestamps
        )
