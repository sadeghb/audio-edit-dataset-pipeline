# src/vad_processor.py
from pathlib import Path

import numpy as np
import pandas as pd
from pydub import AudioSegment

# The Silero VAD model expects audio to be in a specific format.
EXPECTED_SAMPLE_RATE = 16000

def process_audio(audio_path: Path, model, get_speech_timestamps) -> pd.DataFrame:
    """
    Pre-processes an audio file and runs Silero VAD to find speech segments.

    This function handles the critical steps of loading an audio file in any
    format, converting it to the precise format required by the Silero VAD model
    (16kHz, 16-bit, mono PCM), and then running the model to get timestamps.

    Args:
        audio_path: The path to the source audio file.
        model: The loaded Silero VAD model.
        get_speech_timestamps: The utility function from the Silero VAD package.

    Returns:
        A pandas DataFrame with 'start_ms' and 'end_ms' for each detected speech segment.
    """
    try:
        # 1. Load and pre-process the audio using pydub.
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2) # 16-bit
        
        # 2. Convert to the float32 NumPy array format the model expects.
        raw_samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        audio_float32 = raw_samples.astype(np.float32) / 32768.0
    except Exception as e:
        raise RuntimeError(f"Error loading or preprocessing audio file {audio_path.name}: {e}")

    # 3. Run the VAD model on the processed audio.
    try:
        speech_timestamps = get_speech_timestamps(
            audio_float32, model, sampling_rate=EXPECTED_SAMPLE_RATE
        )
    except Exception as e:
        raise RuntimeError(f"Error during VAD processing for {audio_path.name}: {e}")

    if not speech_timestamps:
        return pd.DataFrame(columns=['start_ms', 'end_ms'])
    
    # 4. Format the output from samples to milliseconds in a DataFrame.
    df = pd.DataFrame(speech_timestamps)
    df['start_ms'] = (df['start'] / EXPECTED_SAMPLE_RATE * 1000).astype(int)
    df['end_ms'] = (df['end'] / EXPECTED_SAMPLE_RATE * 1000).astype(int)
    
    return df[['start_ms', 'end_ms']]
