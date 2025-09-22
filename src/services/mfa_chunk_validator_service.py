# src/services/mfa_chunk_validator_service.py
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

class MfaChunkValidatorService:
    """
    Uses 'mfa validate' to detect Out-of-Vocabulary (OOV) words in chunks
    before running the full, computationally expensive alignment.
    """
    def __init__(self, config: Dict[str, Any]):
        self.mfa_config: Dict[str, Any] = config.get("mfa", {})
        self.num_jobs: int = self.mfa_config.get("num_jobs", 1)
        self.dictionary_name: str = self.mfa_config.get("dictionary_name", "english_us_arpa")
        logging.info("MfaChunkValidatorService initialized (dictionary=%s)", self.dictionary_name)

    def run(self, mfa_chunks_dir: Path, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs the MFA validator and injects OOV information into each chunk dictionary.

        Args:
            mfa_chunks_dir: The directory containing the .lab and .wav files for MFA.
            chunks: The list of chunk dictionaries produced by MfaChunkerService.

        Returns:
            The same list of chunks, now annotated with 'oov_words' and 'contains_oov'.
        """
        validate_out_dir = mfa_chunks_dir / "mfa_validate_output"
        if validate_out_dir.exists():
            shutil.rmtree(validate_out_dir)

        cmd = [ "mfa", "validate", str(mfa_chunks_dir), self.dictionary_name,
                "--ignore_acoustics", "--clean", "--overwrite",
                "--num_jobs", str(self.num_jobs),
                "--output_directory", str(validate_out_dir) ]
        
        logging.info("Running MFA validator...")
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")

        # Parse the raw text output from the MFA validator to extract OOV words.
        oov_map: Dict[int, List[str]] = {}
        utt_oov_file = validate_out_dir / "utterance_oovs.txt"
        if utt_oov_file.is_file():
            with utt_oov_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    match = re.match(r"mfa_chunk_(\d+).*?:.*?:\s*(.*)", line.strip())
                    if not match: continue
                    chunk_id = int(match.group(1))
                    # MFA outputs characters; this helper groups them back into words.
                    oov_map[chunk_id] = self._collect_words_from_chars(match.group(2).split(","))
        
        # Annotate each chunk with the OOV data.
        for chunk in chunks:
            chunk["contains_oov"] = chunk["id"] in oov_map
            chunk["oov_words"] = oov_map.get(chunk["id"], [])

        logging.info(f"Validator found OOV words in {len(oov_map)}/{len(chunks)} chunks.")
        return chunks

    def _collect_words_from_chars(self, char_tokens: List[str]) -> List[str]:
        """Helper to group MFA's character-based OOV output into words."""
        words, current_word = [], []
        for token in char_tokens:
            token = token.strip()
            if not token: # An empty token indicates a word boundary.
                if current_word:
                    words.append("".join(current_word))
                    current_word = []
            else:
                current_word.append(token)
        if current_word:
            words.append("".join(current_word))
        return words
