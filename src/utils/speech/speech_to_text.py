from abc import ABC, abstractmethod
import os
import wave
from pathlib import Path
from openai import OpenAI
from typing import Optional
import threading

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

class SpeechToTextBase(ABC):
    """Base class for speech-to-text implementations"""
    
    @abstractmethod
    def record_audio(self, output_path: str, duration: int = None):
        """Record audio from microphone"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Convert speech to text"""
        pass

class OpenAISTT(SpeechToTextBase):
    """OpenAI's speech-to-text implementation using Whisper"""

    def __init__(self):
        self.client = OpenAI()
        self.chunk = 1024
        self.channels = 1
        self.rate = 44100
        self.recording = False
        self.audio_available = PYAUDIO_AVAILABLE
        if self.audio_available:
            self.format = pyaudio.paInt16
            self.p = pyaudio.PyAudio()
        
    def record_audio(self, output_path: str, duration: Optional[int] = None):
        """
        Record audio from microphone
        
        Args:
            output_path: Path to save the recorded audio
            duration: Recording duration in seconds. If None, will record until enter is pressed
        """
        if not self.audio_available:
            print("Error: Voice input unavailable - PyAudio not installed")
            return None
        
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("\n🎤 Recording... Press Enter in a new line to stop.")
        frames = []
        self.recording = True
        
        def stop_recording():
            input()  # Wait for Enter key
            self.recording = False
            print("\n⏹️ Recording stopped.")
        
        # Start the input listener in a separate thread
        stop_thread = threading.Thread(target=stop_recording)
        stop_thread.daemon = True
        stop_thread.start()
        
        # Record audio until enter is pressed or duration is reached
        while self.recording:
            try:
                data = stream.read(self.chunk)
                frames.append(data)
                if duration and len(frames) > int(duration * self.rate / self.chunk):
                    self.recording = False
                    print("\n⏹️ Recording stopped (duration reached).")
                    break
            except Exception as e:
                print(f"\nError during recording: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        # self.p.terminate()
        
        # Save the recorded audio
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using OpenAI's Whisper API

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text
        """
        path = Path(audio_path)
        if not path.exists() or path.stat().st_size < 100:
            raise ValueError(f"Audio file is empty or too small: {audio_path}")

        print(f"[STT] Transcribing {path.name} ({path.stat().st_size} bytes, {path.suffix})")

        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        print(f"[STT] Result: '{transcript.text}'")
        return transcript.text

def create_stt_engine() -> Optional[SpeechToTextBase]:
    """Factory function to create STT engine.

    Transcription (Whisper API) works regardless of PyAudio availability.
    PyAudio is only required for local microphone recording.
    """
    if not PYAUDIO_AVAILABLE:
        print("Warning: Local microphone recording unavailable - PyAudio not installed.")
        print("Whisper transcription of uploaded audio will still work.")
        print("To enable microphone recording on macOS:")
        print("1. brew install portaudio")
        print("2. pip install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio")
    return OpenAISTT()