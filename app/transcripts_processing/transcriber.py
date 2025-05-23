from groq import Groq
import tempfile
import os
import logging
from fastapi import HTTPException, UploadFile

# Set up logging for debugging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq()

async def transcribe_audio(audio_file: UploadFile):
    try:
        # Debug: Log information about the incoming file
        logger.debug(f"Received file: {audio_file.filename}, size: {audio_file.size} bytes")
        logger.debug(f"Content type: {audio_file.content_type}")

        # Check if the file is indeed a valid WAV file
        if not audio_file.filename.endswith(".wav"):
            logger.error("Invalid file format. Expected .wav file.")
            raise HTTPException(status_code=422, detail="Invalid file format. Only .wav files are allowed.")

        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio_file.read()
            logger.debug(f"Read {len(content)} bytes of data from the file")

            # Write content to the temporary file
            temp_file.write(content)
            temp_path = temp_file.name

        # Debug: Log the path of the temporary file
        logger.debug(f"Temporary file created at: {temp_path}")

        try:
            # Open the temporary file and transcribe
            with open(temp_path, "rb") as file:
                logger.debug("Sending file to transcription service...")
                transcription = client.audio.transcriptions.create(
                    file=(temp_path, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )
                transcription_text = transcription.text

            # Return transcription result
            logger.debug(f"Transcription result: {transcription_text}")
            return {"transcription": transcription_text}

        finally:
            # Clean up: remove the temporary file
            os.unlink(temp_path)
            logger.debug(f"Temporary file {temp_path} removed.")

    except Exception as e:
        # Log the error before raising it
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")