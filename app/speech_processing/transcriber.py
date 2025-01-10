from google.cloud import speech
from google.protobuf import wrappers_pb2

# Initialize Speech-to-Text client
client = speech.SpeechClient()

def transcribe_audio(gcs_uri: str):
    try:
        # Create RecognitionAudio and RecognitionConfig objects
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            language_code="en-US",
            model="latest_long",
            audio_channel_count=1,
            enable_word_confidence=True,
            enable_word_time_offsets=True,
        )

        # Start long-running transcription
        operation = client.long_running_recognize(config=config, audio=audio)
        print("Waiting for operation to complete...")

        # Wait for the transcription operation to finish
        response = operation.result(timeout=90)

        # Process and return the transcription results
        transcripts = [
            {
                "transcript": result.alternatives[0].transcript,
                "confidence": result.alternatives[0].confidence,
                "words": [
                    {
                        "word": word_info.word,
                        "start_time": word_info.start_time.total_seconds(),
                        "end_time": word_info.end_time.total_seconds(),
                    }
                    for word_info in result.alternatives[0].words
                ],
            }
            for result in response.results
        ]

        return {"message": "Transcription completed successfully!", "transcripts": transcripts}

    except Exception as e:
        raise Exception(f"Error during transcription: {str(e)}")