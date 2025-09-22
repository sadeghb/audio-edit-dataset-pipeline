# src/services/audio_editor_service.py
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class AudioEditorService:
    """
    Handles the programmatic creation of 'natural' and 'unnatural' audio edits.

    This service is the core of the dataset generation pipeline. It uses precise
    timestamp data to perform surgical audio cuts, generating labeled examples
    of both high-quality and low-quality edits.
    """
    def __init__(self, config: Dict[str, Any]):
        logging.info("AudioEditorService initialized.")
        self.config = config.get('editing', {})
        self.backward_invasion_interval = self.config.get('backward_phoneme_invasion_interval', [0.7, 0.9])
        self.forward_invasion_interval = self.config.get('forward_phoneme_invasion_interval', [0.7, 0.9])
        self.context_duration_ms = self.config.get('context_duration_ms', 3000)

    def _find_outward_zero_crossing(self, signal: np.ndarray, sample_index: int, direction: str) -> int:
        """
        Finds the nearest zero-crossing to prevent audible clicks at edit points.

        This is a key audio processing step. It searches "outward" from a given
        cut point to find the last sample before the signal crosses the zero
        axis, which is a perceptually ideal place to splice audio.
        """
        if not (0 <= sample_index < len(signal)):
            return max(0, min(len(signal) - 1, sample_index))

        start_sign = np.sign(signal[sample_index])
        if start_sign == 0: return sample_index

        search_range = range(sample_index + 1, len(signal)) if direction == 'forward' else range(sample_index - 1, -1, -1)
        for i in search_range:
            if np.sign(signal[i]) != start_sign:
                return i if direction == 'forward' else i + 1
        
        return len(signal) - 1 if direction == 'forward' else 0

    def _get_cut_boundaries(self, word_ids_to_cut: List[int], word_id_map: Dict, all_words: List[Dict], bwd_inv: float, fwd_inv: float) -> Tuple[float, float]:
        """
        Calculates the precise start and end time for a cut segment in seconds.

        This method contains two distinct logic paths:
        - For a "natural" cut (invasion factor = 0), it finds the silent midpoint between words.
        - For an "unnatural" cut (invasion factor > 0), it uses phoneme data to "invade" the adjacent word.
        """
        first_word_id = word_ids_to_cut[0]
        last_word_id = word_ids_to_cut[-1]

        first_word_idx = next((i for i, w in enumerate(all_words) if w["id"] == first_word_id), -1)
        last_word_idx = next((i for i, w in enumerate(all_words) if w["id"] == last_word_id), -1)

        # --- Calculate Start Time ---
        if bwd_inv > 0: # Unnatural cut: Invade the previous word's last phoneme.
            prev_word = all_words[first_word_idx - 1] if first_word_idx > 0 else None
            if prev_word and prev_word.get('phonemes'):
                last_phoneme = prev_word['phonemes'][-1]
                duration = last_phoneme['end'] - last_phoneme['start']
                start_time = last_phoneme['end'] - (duration * bwd_inv)
            else: # Fallback if no phoneme data
                start_time = word_id_map[first_word_id]['start']
        else: # Natural cut: Find the silent midpoint between words.
            prev_word = all_words[first_word_idx - 1] if first_word_idx > 0 else None
            start_time = (prev_word['end'] + word_id_map[first_word_id]['start']) / 2 if prev_word else 0.0

        # --- Calculate End Time ---
        if fwd_inv > 0: # Unnatural cut: Invade the next word's first phoneme.
            next_word = all_words[last_word_idx + 1] if last_word_idx < len(all_words) - 1 else None
            if next_word and next_word.get('phonemes'):
                first_phoneme = next_word['phonemes'][0]
                duration = first_phoneme['end'] - first_phoneme['start']
                end_time = first_phoneme['start'] + (duration * fwd_inv)
            else: # Fallback if no phoneme data
                end_time = word_id_map[last_word_id]['end']
        else: # Natural cut: Find the silent midpoint between words.
            next_word = all_words[last_word_idx + 1] if last_word_idx < len(all_words) - 1 else None
            end_time = (word_id_map[last_word_id]['end'] + next_word['start']) / 2 if next_word else word_id_map[last_word_id]['end']

        return start_time, end_time

    def run(self, cut_word_ids: List[int], full_audio: AudioSegment, y_full: np.ndarray, sr: int, mfa_data: List[Dict]) -> Optional[Dict]:
        """
        Generates a "natural" and an "unnatural" audio clip for a given cut.
        
        Args:
            cut_word_ids: List of integer word IDs to be removed.
            full_audio: The full pydub AudioSegment object.
            y_full: The full audio waveform as a NumPy array.
            sr: The sample rate of the audio.
            mfa_data: The list of all word dictionaries from the MFA alignment.

        Returns:
            A dictionary containing the pydub AudioSegment for the natural and unnatural cuts.
        """
        word_id_map = {word['id']: word for word in mfa_data}
        if not all(word_id in word_id_map for word_id in cut_word_ids):
            logging.error(f"One or more word IDs in {cut_word_ids} not found in MFA data. Skipping cut.")
            return None

        # --- 1. Generate the "Natural" Cut ---
        # Calculate boundaries at the silent midpoints between words.
        nat_start_s, nat_end_s = self._get_cut_boundaries(cut_word_ids, word_id_map, mfa_data, 0.0, 0.0)
        # Nudge the cut points to the nearest zero-crossing to prevent clicks.
        nat_splice_before_s = self._find_outward_zero_crossing(y_full, int(nat_start_s * sr), 'backward') / sr
        nat_splice_after_s = self._find_outward_zero_crossing(y_full, int(nat_end_s * sr), 'forward') / sr

        # Splice the audio, keeping some context around the cut.
        nat_before_start_ms = max(0, int(nat_splice_before_s * 1000) - self.context_duration_ms)
        nat_before_segment = full_audio[nat_before_start_ms : int(nat_splice_before_s * 1000)]
        nat_after_end_ms = min(len(full_audio), int(nat_splice_after_s * 1000) + self.context_duration_ms)
        nat_after_segment = full_audio[int(nat_splice_after_s * 1000) : nat_after_end_ms]
        natural_cut_audio = (nat_before_segment + nat_after_segment).set_channels(1).set_frame_rate(16000)

        # --- 2. Generate the "Unnatural" (Phoneme Invasion) Cut ---
        # Calculate boundaries that cut into the adjacent words' phonemes.
        bwd_factor = random.uniform(self.backward_invasion_interval[0], self.backward_invasion_interval[1])
        fwd_factor = random.uniform(self.forward_invasion_interval[0], self.forward_invasion_interval[1])
        unn_start_s, unn_end_s = self._get_cut_boundaries(cut_word_ids, word_id_map, mfa_data, bwd_factor, fwd_factor)
        # Nudge to zero-crossings.
        unn_splice_before_s = self._find_outward_zero_crossing(y_full, int(unn_start_s * sr), 'backward') / sr
        unn_splice_after_s = self._find_outward_zero_crossing(y_full, int(unn_end_s * sr), 'forward') / sr
        
        # Splice the audio.
        unn_before_start_ms = max(0, int(unn_splice_before_s * 1000) - self.context_duration_ms)
        unn_before_segment = full_audio[unn_before_start_ms : int(unn_splice_before_s * 1000)]
        unn_after_end_ms = min(len(full_audio), int(unn_splice_after_s * 1000) + self.context_duration_ms)
        unn_after_segment = full_audio[int(unn_splice_after_s * 1000) : unn_after_end_ms]
        unnatural_cut_audio = (unn_before_segment + unn_after_segment).set_channels(1).set_frame_rate(16000)

        return { "natural_cut": natural_cut_audio, "unnatural_cut": unnatural_cut_audio }
