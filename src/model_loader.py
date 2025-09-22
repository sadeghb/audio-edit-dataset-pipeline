# src/model_loader.py
import logging

import torch

def load_silero_model():
    """
    Loads the pre-trained Silero VAD model and its utility functions from Torch Hub.

    This function is called once at the module level to ensure the model is
    loaded into memory only a single time when the application starts. This
    singleton-like pattern is efficient and prevents slow model loading during
    request processing.

    Returns:
        A tuple containing the loaded model and a dictionary of utility functions.
    
    Raises:
        Exception: If the model cannot be downloaded or loaded from Torch Hub.
    """

    logging.info("Initializing Silero VAD model... (This should only happen once)")
    try:
        # Load the model from the official Silero repository.
        # onnx=True is used for a faster runtime.
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=True)
        return model, utils
    except Exception as e:
        logging.error(f"❌ Fatal: Error loading Silero VAD model: {e}")
        raise

# Load the model once when this module is first imported.
SILERO_MODEL, SILERO_UTILS = load_silero_model()
