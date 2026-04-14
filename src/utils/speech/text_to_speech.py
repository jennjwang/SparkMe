from abc import ABC, abstractmethod
from typing import Optional
import os
import re
from openai import OpenAI

_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)



class TextToSpeechBase(ABC):
    """Base class for text-to-speech implementations"""
    
    @abstractmethod
    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> str:
        """Convert text to speech and save to file"""
        pass

class OpenAITTS(TextToSpeechBase):
    """OpenAI's text-to-speech implementation using Whisper"""
    
    def __init__(self, voice: str = "alloy"):
        """
        Initialize OpenAI TTS
        
        Args:
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        """
        self.client = OpenAI()
        self.voice = voice
        
    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Convert text to speech using OpenAI's API
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file. If None, will generate one
            
        Returns:
            Path to the saved audio file
        """
        if output_path is None:
            output_path = f"{os.getenv('DATA_DIR', 'data')}/temp_audio_{hash(text)}.mp3"
            
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=text,
            speed=1.0
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Write the binary content directly to file
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        return output_path

class CartesiaTTS(TextToSpeechBase):
    """Cartesia's text-to-speech implementation using Sonic models"""

    def __init__(self, voice: str = None, model_id: str = "sonic-3"):
        """
        Initialize Cartesia TTS

        Args:
            voice: Voice ID to use. Defaults to CARTESIA_VOICE_ID env var.
            model_id: Model to use (default: sonic-3)
        """
        from cartesia import Cartesia
        self.client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))
        valid_voice = voice if (voice and _UUID_RE.match(voice)) else None
        self.voice_id = valid_voice or os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")
        self.model_id = model_id

    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Convert text to speech using Cartesia's API

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file. If None, will generate one

        Returns:
            Path to the saved audio file
        """
        if output_path is None:
            output_path = f"{os.getenv('DATA_DIR', 'data')}/temp_audio_{hash(text)}.mp3"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Use mp3 for .mp3 paths, wav otherwise
        if output_path.endswith(".mp3"):
            output_format = {"container": "mp3", "bit_rate": 128000, "sample_rate": 44100}
        else:
            output_format = {"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100}

        response = self.client.tts.generate(
            model_id=self.model_id,
            transcript=text,
            voice={"mode": "id", "id": self.voice_id},
            output_format=output_format,
        )
        response.write_to_file(output_path)
        return output_path

    def stream_tts_chunks(self, text: str):
        """Stream TTS as MP3 byte chunks via Cartesia /tts/bytes endpoint."""
        output_format = {"container": "mp3", "bit_rate": 128000, "sample_rate": 44100}
        response = self.client.tts.generate(
            model_id=self.model_id,
            transcript=text,
            voice={"mode": "id", "id": self.voice_id},
            output_format=output_format,
        )
        yield from response.iter_bytes()


class GoogleTTS(TextToSpeechBase):
    """Placeholder for Google TTS implementation"""
    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> str:
        raise NotImplementedError("Google TTS not implemented yet")

def create_tts_engine(provider: str = "openai", **kwargs) -> TextToSpeechBase:
    """Factory function to create TTS engine"""
    providers = {
        "openai": OpenAITTS,
        "cartesia": CartesiaTTS,
        "google": GoogleTTS
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available providers: {list(providers.keys())}")

    return providers[provider](**kwargs)