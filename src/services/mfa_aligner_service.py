# src/services/mfa_aligner_service.py
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

class MfaAlignerService:
    """
    A service to run the core Montreal Forced Aligner command on a directory
    of prepared audio chunks and their corresponding transcript files.
    """
    def __init__(self, config: Dict[str, Any]):
        logging.info("MfaAlignerService initialized.")
        self.mfa_config = config.get('mfa', {})
        self.num_jobs = self.mfa_config.get('num_jobs', 1)
        self.dictionary_name = self.mfa_config.get('dictionary_name')
        self.acoustic_model_name = self.mfa_config.get('acoustic_model_name')

    def run(self, mfa_chunks_dir: Path) -> Path:
        """
        Runs the 'mfa align' process on a directory of prepared chunks.

        Args:
            mfa_chunks_dir: The directory containing the .lab and .wav files.

        Returns:
            The path to the directory containing the output .TextGrid files.
            
        Raises:
            FileNotFoundError: If the 'mfa' command is not found.
            subprocess.CalledProcessError: If the MFA process returns a non-zero exit code.
        """
        logging.info(f"Starting MFA alignment for files in {mfa_chunks_dir}...")
        output_dir = mfa_chunks_dir / "mfa_output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        mfa_command = [ "mfa", "align", str(mfa_chunks_dir),
                        self.dictionary_name, self.acoustic_model_name, str(output_dir),
                        "--clean", "--overwrite", "--num_jobs", str(self.num_jobs) ]
        logging.info(f"Executing MFA command: {' '.join(mfa_command)}")

        try:
            process = subprocess.run(mfa_command, check=True, capture_output=True, text=True, encoding='utf-8')
            logging.info("MFA alignment completed successfully.")
            return output_dir
        except FileNotFoundError:
            logging.error("MFA command not found. Is MFA installed and in your system's PATH?")
            raise
        except subprocess.CalledProcessError as e:
            logging.error(f"MFA process failed with exit code {e.returncode}.")
            logging.error("MFA Stderr:\n" + e.stderr)
            logging.error("MFA Stdout:\n" + e.stdout)
            raise
