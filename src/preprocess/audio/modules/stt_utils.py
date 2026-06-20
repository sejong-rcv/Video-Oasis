import torch
from transformers import pipeline

class WhisperTranscriber:
    def __init__(self, model_id="openai/whisper-large-v3", device=0):
        if isinstance(device, int) or (isinstance(device, str) and "cuda" in device):
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device, 
            torch_dtype=dtype,
            model_kwargs={"attn_implementation": "sdpa"} 
        )

    def transcribe(self, audio_path, batch_size=16):
        try:
            res = self.pipe(
                audio_path, 
                chunk_length_s=30, 
                batch_size=batch_size, 
                return_timestamps=True
            )
            return res
        except Exception as e:
            return {"text": f"Error: {str(e)}", "chunks": []}