# main.py
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src import model_loader
from src.pipeline_orchestrator import PipelineOrchestrator
from src.services.audio_splitter_service import AudioSplitterService
from src.services.mfa_aligner_service import MfaAlignerService
from src.services.mfa_chunk_validator_service import MfaChunkValidatorService
from src.services.mfa_chunker_service import MfaChunkerService
from src.services.mfa_normalizer_service import MfaNormalizerService
from src.services.scribe_chunker_service import ScribeChunkerService
from src.services.scribe_normalizer_service import ScribeNormalizerService
from src.services.scribe_transcriber_service import ScribeTranscriberService
from src.services.split_point_service import SplitPointService
from src.services.vad_service import VADService
from src.utils.config_loader import load_config


def main():
    """
    Main entry point for running the dataset generation pipeline as a batch job.

    This script acts as the "driver" for the application. Its responsibilities include:
    1. Parsing command-line arguments.
    2. Loading configuration and models.
    3. Building all the necessary service objects (the "Composition Root").
    4. Instantiating the main PipelineOrchestrator.
    5. Reading a manifest of audio files from a metadata CSV.
    6. Looping through each audio file and executing the pipeline.
    7. Updating the metadata CSV with the results.
    """
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the VAD-Scribe-MFA dataset generation pipeline.")
    parser.add_argument(
        "--metadata-csv",
        type=str,
        required=True,
        help="Path to the metadata CSV file containing the 'converted_file_path' for audios to process."
    )
    args = parser.parse_args()

    # --- 2. Configuration and Model Loading ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    
    config = load_config()
    logging.info("Configuration and models loaded.")

    # --- 3. Service Instantiation (Composition Root) ---
    # Build all specialist services and place them in a dictionary for injection.
    services = {
        'vad': VADService(model=model_loader.SILERO_MODEL, utils=model_loader.SILERO_UTILS),
        'split_point': SplitPointService(),
        'audio_splitter': AudioSplitterService(),
        'scribe_chunker': ScribeChunkerService(),
        'scribe_trascriber': ScribeTranscriberService(config['api_keys']['elevenlabs']),
        'scribe_normalizer': ScribeNormalizerService(),
        'mfa_chunker': MfaChunkerService(),
        'mfa_chunk_validator': MfaChunkValidatorService(config),
        'mfa_aligner': MfaAlignerService(config),
        'mfa_normalizer': MfaNormalizerService(),
    }
    logging.info("All services initialized.")

    # --- 4. Orchestrator Instantiation ---
    # The main "engine" is created here, with all its dependencies injected.
    orchestrator = PipelineOrchestrator(services=services, config=config)

    # --- 5. Data Loading ---
    metadata_csv_path = Path(args.metadata_csv)
    if not metadata_csv_path.exists():
        logging.error(f"FATAL: Metadata CSV not found at '{metadata_csv_path}'")
        return

    df = pd.read_csv(metadata_csv_path)
    # Ensure the output columns exist in the DataFrame.
    for col in ['vad_path', 'scribe_path', 'mfa_path']:
        if col not in df.columns:
            df[col] = pd.NA
            
    audio_files = [Path(p) for p in df['converted_file_path'].dropna().tolist()]
    if not audio_files:
        logging.warning(f"No audio files found in '{metadata_csv_path}'.")
        return

    # --- 6. Main Processing Loop ---
    logging.info(f"Found {len(audio_files)} audio file(s) to process from metadata CSV.")
    for index, audio_path in enumerate(tqdm(audio_files, desc="Processing audio files")):
        try:
            logging.info(f"--- Starting processing for: {audio_path.name} ---")
            
            # Run the entire pipeline for one audio file.
            orchestrator.run(audio_path=audio_path)

            # --- 7. Update Metadata ---
            # Record the paths to the generated output files in the main CSV.
            output_dir = audio_path.parent
            df.loc[index, 'vad_path'] = str(output_dir / (audio_path.stem + "_vad.csv"))
            df.loc[index, 'scribe_path'] = str(output_dir / (audio_path.stem + "_scribe.json"))
            df.loc[index, 'mfa_path'] = str(output_dir / (audio_path.stem + "_mfa.json"))

        except Exception as e:
            logging.error(f"❌ An unhandled error occurred for {audio_path.name}: {e}", exc_info=True)

    # Save the updated DataFrame with the new paths back to the CSV file.
    df.to_csv(metadata_csv_path, index=False)
    logging.info(f"--- All files processed. Metadata CSV updated at {metadata_csv_path} ---")

if __name__ == '__main__':
    main()
