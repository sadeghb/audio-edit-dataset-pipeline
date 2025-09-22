# Automated Audio Edit Quality Dataset Pipeline

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)
![PyDub](https://img.shields.io/badge/PyDub-Audio-4B4D4B?style=for-the-badge)

## Overview

This project is the culmination of a multi-stage research and development effort to solve the "Quality Problem" in automated audio editing. Rather than relying on simple heuristics, this project takes a modern, data-driven approach. Its purpose is to build an **automated pipeline that generates a large-scale, labeled dataset** of "good" and "bad" audio edits. This dataset is designed to enable the future training of a state-of-the-art deep learning model that can automatically classify the perceptual quality of an audio edit.

This pipeline is a capstone project that **integrates previously developed services**, including a high-precision Montreal Forced Aligner (MFA) service, to create its foundational data.

---

## ⚠️ Portfolio Version Notice

This is a polished, portfolio-ready version of a project developed during a professional internship. The code and architecture are presented as is to showcase the engineering and problem-solving skills involved.

---

## The Data Generation Pipeline

The system is architected as a modular, multi-stage pipeline that processes raw audio into a structured, labeled dataset. The flow is orchestrated by the `PipelineOrchestrator` class and proceeds as follows:

1.  **VAD (Voice Activity Detection)**: The pipeline first uses Silero VAD to identify all voiced segments in the source audio, creating a map of speech vs. silence.
2.  **Transcription**: The audio is then transcribed using a service like ElevenLabs Scribe to generate an initial transcript with word-level timestamps.
3.  **Chunking**: The audio and transcript are intelligently chunked into smaller segments suitable for processing, using silences as the primary split points.
4.  **Validation**: Before alignment, each chunk is validated using `mfa validate` to identify and flag any Out-of-Vocabulary (OOV) words, ensuring data quality.
5.  **Alignment**: The validated chunks are then processed by the Montreal Forced Aligner (MFA) to generate hyper-precise, phoneme-level timestamps.
6.  **Normalization**: The raw `.TextGrid` outputs from MFA are parsed and normalized into a clean, unified JSON format with rich diagnostic data.

---

## The Core Innovation: `AudioEditorService`

The final and most innovative stage of the pipeline is the `AudioEditorService`, which uses the high-precision aligned data to programmatically generate a triplet of audio clips for each potential edit point:

* **1. Original Clip**: An unedited segment of the source audio for reference.
* **2. Natural Cut**: A "good" edit, created using a sophisticated heuristic that places the splice in the **exact midpoint of the silent gap** between words.
* **3. Unnatural Cut**: A "bad" edit, created by intentionally **"invading" the phonemes** of the adjacent words by a randomized, configurable amount.

Furthermore, all edit points are finely adjusted to the nearest **outward zero-crossing** in the audio waveform to prevent audible "clicks," a key detail from professional audio engineering.

---

## Professional Architecture

This project was built using modern software design patterns to ensure the code is modular, reusable, and testable.

* **Dependency Injection & Composition Root**: The `main.py` script acts as a "Composition Root," where it builds all the individual service objects and "injects" them into the main orchestrator.
* **Reusable Pipeline "Engine"**: The `src/pipeline_orchestrator.py` file contains a self-contained `PipelineOrchestrator` class that acts as the reusable "engine" for processing a single file.
* **Modular, Service-Oriented Design**: Each distinct task in the pipeline (VAD, chunking, aligning, editing, etc.) is encapsulated in its own class within the `src/services/` directory, promoting a clean separation of concerns.

---

## 🚀 Usage

This pipeline is designed to be run as a command-line batch processing tool.

**1. Prepare the Input**
The script requires a single manifest file: a CSV containing a `converted_file_path` column that points to the local audio files you wish to process.

**2. Run the Pipeline**
Execute the `main.py` script from the project root, providing the path to your metadata file.

```bash
python main.py --metadata-csv path/to/your/metadata.csv