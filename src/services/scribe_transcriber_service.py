# src/services/scribe_transcriber_service.py
import logging
from pathlib import Path
from typing import Any, Dict

import requests

class ScribeTranscriberService:
    """A dedicated service for transcribing audio using the ElevenLabs Scribe API."""
    def __init__(self, api_key: str):
        logging.info("ScribeTranscriberService initialized.")
        if not api_key or "YOUR_ELEVENLABS_API_KEY_HERE" in api_key:
            raise ValueError("ElevenLabs API key is not configured.")
        self.api_key = api_key
        self.url = 'https://api.elevenlabs.io/v1/speech-to-text'

    def run(self, audio_chunk_path: Path) -> Dict[str, Any]:
        """
        Transcribes a single audio chunk and returns the full JSON response.
        
        Args:
            audio_chunk_path: The local path to the audio chunk to be transcribed.
            
        Returns:
            The raw JSON response from the Scribe API as a dictionary.
        
        Raises:
            requests.exceptions.RequestException: If the API call fails.
        """
        logging.info(f"Requesting Scribe transcription for '{audio_chunk_path.name}'")
        headers = {'xi-api-key': self.api_key}
        data = {'model_id': 'eleven_scribe_v1', 'diarize': 'true'}

        try:
            with open(audio_chunk_path, 'rb') as audio_file:
                files = {'file': (audio_chunk_path.name, audio_file, 'audio/wav')}
                response = requests.post(self.url, headers=headers, data=data, files=files, timeout=300)
                response.raise_for_status()
            
            logging.info(f"Successfully received transcription for '{audio_chunk_path.name}'.")
            return response.json()
        
        except requests.exceptions.RequestException as e:
            if e.response is not None:
                logging.error(f"Scribe API error. Status: {e.response.status_code}, Body: {e.response.text}")
            logging.error(f"Scribe API request failed for '{audio_chunk_path.name}': {e}", exc_info=False)
            raise
