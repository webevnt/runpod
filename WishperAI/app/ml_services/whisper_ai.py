from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import math
import torch
import scipy.io.wavfile
from typing import Optional, List, Dict, Any
from app.config import settings
import io


# from whisperx import load_align_model, align
# from whisperx.diarize import DiarizationPipeline, assign_word_speakers


print(settings.HF_AUTH_TOKEN,"-----------------------------")


def prepare_diarize_audio(contents):
    import whisper
    from transformers.pipelines.audio_utils import ffmpeg_read
    inputs = ffmpeg_read(contents, 16000)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=DEVICE)
    audio = whisper.pad_or_trim(inputs)
    return audio


def transcribe(contents):
    import whisper
    from transformers.pipelines.audio_utils import ffmpeg_read
    inputs = ffmpeg_read(contents, 16000)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=DEVICE)
    audio = whisper.pad_or_trim(inputs)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
    result = whisper.decode(model, mel, options)
    
    
    return {
        "text": result.text,
        "language_code": result.language
    }
    
def transcribe_audio(audio) -> Dict[str, Any]:
    """
    Transcribe an audio file using a speech-to-text model.

    Args:
        audio_file: Path to the audio file to transcribe.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the transcript, including the segments, the language code, and the duration of the audio file.
    """
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device)
    result = model.transcribe(audio, fp16=False)
    
    language_code = result["language"]
    return {
        "segments": result["segments"],
        "language_code": language_code,
    }

def align_segments(
    segments: List[Dict[str, Any]],
    language_code: str,
    audio
) -> Dict[str, Any]:
    """
    Align the transcript segments using a pretrained alignment model.

    Args:
        segments: List of transcript segments to align.
        language_code: Language code of the audio file.
        contents: Path to the audio file containing the audio data.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the aligned transcript segments.
    """
    from whisperx import load_align_model, align
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_a, metadata = load_align_model(language_code=language_code, device=device)
    result_aligned = align(segments, model_a, metadata, audio, device)
    return result_aligned

def diarize(audio) -> Dict[str, Any]:
    """
    Perform speaker diarization on an audio file.

    Args:
        contents: Path to the audio file to diarize.
        hf_token: Authentication token for accessing the Hugging Face API.

    Returns:
        A dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
    """
    # from whisperx.diarize import DiarizationPipeline
    from pyannote.audio import Pipeline
    # diarization_pipeline = DiarizationPipeline(use_auth_token=settings.HF_AUTH_TOKEN)
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",use_auth_token=settings.HF_AUTH_TOKEN)
    diarization_result = diarization_pipeline(audio)
    return diarization_result


def assign_speakers(
    diarization_result: Dict[str, Any], aligned_segments: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Assign speakers to each transcript segment based on the speaker diarization result.

    Args:
        diarization_result: Dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
        aligned_segments: Dictionary representing the aligned transcript segments.

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    result_segments, word_seg = assign_word_speakers(
        diarization_result, aligned_segments["segments"]
    )
    results_segments_w_speakers: List[Dict[str, Any]] = []
    for result_segment in result_segments:
        results_segments_w_speakers.append(
            {
                "start": result_segment["start"],
                "end": result_segment["end"],
                "text": result_segment["text"],
                "speaker": result_segment["speaker"],
            }
        )
    return results_segments_w_speakers

def transcribe_and_diarize(contents, FILE_PATH) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file and perform speaker diarization to determine which words were spoken by each speaker.

    Args:
        audio_file: Path to the audio file to transcribe and diarize.
        hf_token: Authentication token for accessing the Hugging Face API.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers.pipelines.audio_utils import ffmpeg_read
    from pydub import AudioSegment
    import torchaudio
    audio = ffmpeg_read(contents, 16000)
    transcript = transcribe_audio(audio)
    aligned_segments = align_segments(
        transcript["segments"], transcript["language_code"], audio
    )
    # sound = AudioSegment.from_file(FILE_PATH)
    # print("===========================================")
    # # # this is an array
    # samples = sound.get_array_of_samples()
    # sample_rate = sound.get_frame()
    # samples_np = np.array(samples)
    
    # samples_tensor = torch.tensor(samples)
    # inputs = {"waveform": samples_tensor, "sample_rate": 44100}
    waveform, sample_rate = torchaudio.load(FILE_PATH, normalize=True)
    inputs = {"waveform": waveform, "sample_rate": sample_rate}
    
    diarization_result = diarize(inputs)
    results_segments_w_speakers = assign_speakers(diarization_result, aligned_segments)

    # Print the results in a user-friendly way
    for i, segment in enumerate(results_segments_w_speakers):
        print(f"Segment {i + 1}:")
        print(f"Start time: {segment['start']:.2f}")
        print(f"End time: {segment['end']:.2f}")
        print(f"Speaker: {segment['speaker']}")
        print(f"Transcript: {segment['text']}")
        print("")

    return results_segments_w_speakers