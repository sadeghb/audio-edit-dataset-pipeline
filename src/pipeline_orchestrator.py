# src/pipeline_orchestrator.py

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from pydub import AudioSegment

from .utils.mfa_text_normalizer import normalize_text_for_mfa


class PipelineOrchestrator:
    """
    Orchestrates the multi-stage VAD-Scribe-MFA pipeline for a single audio file.

    This class is the core "engine" of the dataset generation process. It is
    designed to be a reusable component that accepts a path to an audio file
    and processes it through a sequence of specialized services. It uses
    dependency injection to receive the services it needs, making it flexible
    and testable.
    """

    def __init__(self, services: Dict, config: Dict[str, Any]):
        """
        Initializes the orchestrator with its dependencies.

        Args:
            services: A dictionary of instantiated service objects.
            config: The application's configuration dictionary.
        """
        logging.info("PipelineOrchestrator initialized.")
        self.services = services
        self.config = config
        self.use_cache = self.config.get('use_cache', False)

    def run(self, audio_path: Path):
        """
        Executes the full VAD-Scribe-MFA pipeline for a single audio file.

        The pipeline is resumable; if `use_cache` is True and an output file for a
        stage already exists, that stage will be skipped.

        Args:
            audio_path: The path to the source audio file to process.
        """
        logging.info(f"\n--- Starting pipeline for: {audio_path.name} ---")

        # Define final output paths next to the source file.
        output_dir = audio_path.parent
        vad_output_path = output_dir / (audio_path.stem + "_vad.csv")
        scribe_output_path = output_dir / (audio_path.stem + "_scribe.json")
        mfa_output_path = output_dir / (audio_path.stem + "_mfa.json")

        # Create a single temporary directory for all intermediate files (e.g., audio chunks).
        temp_work_dir = Path(tempfile.mkdtemp(prefix="pipeline_"))
        logging.info(f"Created temporary working directory: {temp_work_dir}")
        
        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration_s = len(audio) / 1000.0
            
            # --- Stage 1: Voice Activity Detection (VAD) ---
            logging.info("Executing VAD stage...")
            if self.use_cache and vad_output_path.exists():
                logging.info(f"Using cache for VAD: {vad_output_path}")
                vad_df = pd.read_csv(vad_output_path)
            else:
                vad_df = self.services['vad'].run(audio_path)
                vad_df.to_csv(vad_output_path, index=False)
            logging.info(f"VAD stage complete. Output: {vad_output_path}")

            if vad_df.empty:
                logging.warning("VAD returned no speech segments. Aborting pipeline for this file.")
                return

            # --- Stage 2: Scribe Transcription ---
            logging.info("Executing Scribe Transcription stage...")
            if self.use_cache and scribe_output_path.exists():
                logging.info(f"Using cache for Scribe: {scribe_output_path}")
                with open(scribe_output_path, 'r') as f:
                    final_transcript = json.load(f)
            else:
                split_points_df = self.services['split_point'].run(vad_df, len(audio))
                scribe_chunks_df = self.services['scribe_chunker'].run(split_points_df)
                splitter_df = scribe_chunks_df.rename(columns={'chunk_start_ms': 'start_ms', 'chunk_end_ms': 'end_ms'})
                
                chunk_paths = self.services['audio_splitter'].run(audio, splitter_df, temp_work_dir, audio_path.stem)
                
                raw_scribe_results = [self.services['scribe_trascriber'].run(p) for p in chunk_paths]
                
                final_transcript = self.services['scribe_normalizer'].run(raw_scribe_results, scribe_chunks_df)
                with open(scribe_output_path, 'w') as f:
                    json.dump(final_transcript, f, indent=2)
            logging.info(f"Scribe Transcription stage complete. Output: {scribe_output_path}")

            # --- Stage 3: MFA Alignment ---
            logging.info("Executing MFA Alignment stage...")
            if self.use_cache and mfa_output_path.exists():
                logging.info(f"Using cache for MFA: {mfa_output_path}")
            else:
                split_points_df = self.services['split_point'].run(vad_df, len(audio))
                mfa_chunks = self.services['mfa_chunker'].run(split_points_df, final_transcript, total_duration_s=total_duration_s)
                
                # Prepare .lab and .wav files for MFA in the temporary directory.
                for chunk in mfa_chunks:
                    lab_path = temp_work_dir / f"mfa_chunk_{chunk['id']}.lab"
                    normalized_text = normalize_text_for_mfa(chunk['transcript'])
                    with open(lab_path, 'w') as f:
                        f.write(normalized_text)
                    self.services['audio_splitter'].split_and_save_chunk(audio, chunk['start_s'] * 1000, chunk['end_s'] * 1000, temp_work_dir / f"mfa_chunk_{chunk['id']}.wav")
                
                validated_chunks = self.services['mfa_chunk_validator'].run(temp_work_dir, mfa_chunks)
                mfa_output_dir = self.services['mfa_aligner'].run(temp_work_dir)
                final_mfa_data = self.services['mfa_normalizer'].run(mfa_output_dir, validated_chunks)
                
                with open(mfa_output_path, 'w') as f:
                    json.dump(final_mfa_data, f, indent=4)
            logging.info(f"MFA Alignment stage complete. Output: {mfa_output_path}")

            logging.info(f"\n--- Pipeline finished for: {audio_path.name} ---")
        
        finally:
            # This block ensures the temporary directory and all its contents are always cleaned up.
            logging.info(f"Cleaning up temporary working directory: {temp_work_dir}")
            shutil.rmtree(temp_work_dir)
