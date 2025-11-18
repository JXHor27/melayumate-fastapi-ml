import malaya_speech
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
import io
import logging.config
import uuid
import soundfile as sf
from contextvars import ContextVar
from logging_config import LOGGING_CONFIG
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
from model import TTSRequest, STTResponse

# --- Context Variable and Logging Setup ---
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="NO_ID")
logging.config.dictConfig(LOGGING_CONFIG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speech_service")

# --- Initialize FastAPI app ---
app = FastAPI(
    title="Malaya Speech Service",
    description="A service for Text-to-Speech and Speech-to-Text using Malaya-Speech.",
)


# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Malaya Speech Service is running"}


# --- Global Dictionary to Hold Models ---
tts_models = {}
vocoder_models = {}
stt_models = {}


@app.on_event("startup")
async def load_model():
    """
    Load STT and TTS models on startup
    """
    logger.info("Loading Whisper model and processor...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Load the processor (tokenizer and feature extractor)
    stt_models['processor'] = AutoProcessor.from_pretrained("mesolitica/malaysian-whisper-base")
    stt_models['model'] = AutoModelForSpeechSeq2Seq.from_pretrained("mesolitica/malaysian-whisper-base")
    stt_models['model'].to(device)  # Move model to GPU if available
    logger.info("Whisper model and processor loaded successfully.")

    # --- Load TTS FastSpeech2 models ---
    logger.info("Loading TTS models...")
    tts_models['female'] = malaya_speech.tts.fastspeech2(model='yasmin')
    tts_models['male'] = malaya_speech.tts.fastspeech2(model='osman')
    logger.info("TTS models loaded successfully.")

    logger.info("Loading Vocoder models...")
    vocoder_models['female'] = malaya_speech.vocoder.melgan(model='yasmin')
    vocoder_models['male'] = malaya_speech.vocoder.melgan(model='osman')
    logger.info("Vocoder models loaded successfully.")


# --- Request Interceptor Middleware Setup ---
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id")
    if not trace_id:
        trace_id = str(uuid.uuid4())
    token = trace_id_var.set(trace_id)

    logger.info(f"Request started: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        logger.info(f"Request finished with status: {response.status_code}")
        return response
    finally:
        trace_id_var.reset(token)


# --- Text-to-Speech Endpoint ---
@app.post("/tts", response_class=StreamingResponse)
async def text_to_speech(request: TTSRequest):
    """
    Converts text to speech using a gender-specific model and returns the audio as a WAV file.
    """
    try:
        logger.info(f"Received TTS request for gender: '{request.gender}' with text: '{request.text[:30]}'")
        tts = tts_models[request.gender]
        vocoder = vocoder_models[request.gender]
        mel_spectrogram = tts.predict(request.text)
        y_ = vocoder(mel_spectrogram['mel-output'])

        # Write audio data to an in-memory buffer as a WAV file
        buffer = io.BytesIO()
        sf.write(buffer, y_, samplerate=22050, format="WAV")

        # Move buffer's cursor back to beginning to be read by response
        buffer.seek(0)

        # Stream content back to client
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        logger.error(f"An error occurred during TTS processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process text-to-speech request.")


# --- Speech-to-Text Endpoint ---
@app.post("/stt", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...)):
    """
    Converts speech from an audio file to text using the Whisper model.
    """
    try:
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

        logger.info(f"Received STT request for file: {file.filename}")

        # Read uploaded file content into memory
        audio_bytes = await file.read()

        # Load audio data and its original sampling rate
        # 'y' will be audio waveform (numpy array), 'sr' will be sample rate
        y, sr = sf.read(io.BytesIO(audio_bytes))

        # Resample audio to 16,000 Hz, which is what Whisper expects.
        if sr != 16000:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=16000)

        # Ensure audio is mono (single channel)
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        processor = stt_models['processor']
        model = stt_models['model']

        # Process the audio array to get the input features
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features
        attention_mask = torch.ones(input_features.shape, dtype=torch.long)
        if torch.cuda.is_available():
            input_features = input_features.to("cuda:0")
            attention_mask = attention_mask.to("cuda:0")
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            language='ms'
        )

        # Decode the token IDs to text
        # Using batch_decode is preferred as it handles batches and cleans up special tokens.
        transcribed_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        print(f"Whisper transcription result: '{transcribed_text}'")
        return STTResponse(text=transcribed_text)

    except Exception as e:
        print(f"Error during STT with Whisper: {e}")
        # Be careful not to expose too much detail in production errors
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {e}")

# --- Speech-to-Text Endpoint ---
# @app.post("/stt", response_model=STTResponse)
# async def speech_to_text(file: UploadFile = File(...)):
#     """
#     Converts speech from an audio file to text.
#     """
#     try:
#         if not file.content_type.startswith("audio/"):
#             raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
#
#         print(f"Received STT request for file: {file.filename}")
#         # Read the audio file content
#         audio_bytes = await file.read()
#
#         y, sr = sf.read(io.BytesIO(audio_bytes))
#         transcribed_text = small_model.beam_decoder([y])[0]
#
#         print(f"Transcription result: '{transcribed_text}'")
#         return STTResponse(text=transcribed_text)
#     except Exception as e:
#         print(f"Error during STT: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # --- Text-to-Speech Endpoint ---
# @app.post("/tts", response_class=StreamingResponse)
# async def text_to_speech(request: TTSRequest):
#     """
#     Converts text to speech and returns the audio as a WAV file.
#     """
#     try:
#         logger.info(f"Received TTS request: '{request}'")
#
#         mel_spectrogram = tts_model.predict(request.text)
#         y_ = vocoder_female(mel_spectrogram['mel-output'])
#         buffer = io.BytesIO()
#         sf.write(buffer, y_, samplerate=22050, format="WAV")
#         # Move the cursor to the beginning
#         buffer.seek(0)
#         return StreamingResponse(buffer, media_type="audio/wav")
#
#     except Exception as e:
#         print(f"Error during TTS: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
