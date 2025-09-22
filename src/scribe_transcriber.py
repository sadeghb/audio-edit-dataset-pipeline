# src/scribe_transcriber.py
import logging
from typing import Any, Dict

import requests

from .utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_scribe_results(audio_path: str) -> Dict[str, Any]:
    """
    Transcribes an audio file using the ElevenLabs Scribe API.

    This function handles the entire API interaction, including loading credentials,
    constructing the multipart/form-data request, and handling potential errors.

    Args:
        audio_path: The local path to the audio file (e.g., .mp3, .wav).

    Returns:
        The full, unprocessed JSON response from Scribe as a dictionary.
    
    Raises:
        ValueError: If the API key is not configured.
        requests.exceptions.RequestException: For network or HTTP errors.
    """
    logging.info(f"Requesting transcription from ElevenLabs Scribe for '{audio_path}'")
    try:
        app_config = load_config()
        scribe_api_key = app_config.get('api_keys', {}).get('elevenlabs')
        if not scribe_api_key or "YOUR_ELEVENLABS_API_KEY_HERE" in scribe_api_key:
            raise ValueError("ElevenLabs API key is not configured in config.yaml.")

        url = 'https://api.elevenlabs.io/v1/speech-to-text'
        headers = {'xi-api-key': scribe_api_key}
        
        # The data payload specifies the model and requests rich details like
        # word-level timestamps and speaker labels (diarization).
        data = {
            'model_id': 'eleven_scribe_v1',
            'timestamps': 'word',
            'diarize': 'true'
        }
        
        with open(audio_path, 'rb') as audio_file:
            files = {'file': (audio_path, audio_file, 'audio/mpeg')}
            response = requests.post(url, headers=headers, data=data, files=files, timeout=300)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            logging.info("Successfully received raw response from Scribe.")
            return response.json()

    except (ValueError, requests.exceptions.RequestException) as e:
        logging.error(f"An error occurred while calling ElevenLabs Scribe API: {e}", exc_info=True)
        raise
