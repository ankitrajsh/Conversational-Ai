import assemblyai as aai
from elevenlabs.client import ElevenLabs
import ollama
import constants

class AI_Assistant:
    def __init__(self) -> None:
        aai.settings.api_key = constants.assemblyai_api_key
        self.client = ElevenLabs(api_key=constants.elevenlabs_api_key)
        self.transcriber = None
        self.full_transcript = [
            {"role": "system", "content": "You are a language model called Llama 3 created by Meta."}
        ]

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        pass

    def on_data(self, transcript: aai.RealtimeTranscript):
        if transcript.text:
            if isinstance(transcript, aai.RealtimeFinalTranscript):
                print(transcript.text, end="\r\n")
                self.generate_ai_response(transcript)
            else:
                print(transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        pass

    def on_close(self):
        pass

    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16_000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
        )
        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
        self.transcriber.stream(microphone_stream)

    def close_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def generate_ai_response(self, transcript):
        self.close_transcription()
        self.full_transcript.append({'role': "user", "content": transcript.text})
        print(f"\nUser: {transcript.text}", end="\r\n")
        ollama_stream = ollama.chat(
            model="llama3",
            messages=self.full_transcript,
            stream=True
        )
        text_buffer = ''
        full_text = ''
        for chunk in ollama_stream:
            text_buffer += chunk['messages']['content']
            if text_buffer.endswith('.'):
                audio_stream = self.client.generate(text=text_buffer, model="eleven_turbo_v2", stream=True)
                print(text_buffer, end="\n", flush=True)
                self.stream(audio_stream)
                full_text += text_buffer
                text_buffer = ''
        if text_buffer:
            audio_stream = self.client.generate(text=text_buffer, model="eleven_turbo_v2", stream=True)
            print(text_buffer, end="\n", flush=True)
            self.stream(audio_stream)
            full_text += text_buffer
        self.full_transcript.append({"role": "assistant", "content": full_text})
        self.start_transcription()

    def stream(self, audio_stream):
        for chunk in audio_stream:
            print(chunk['audio'], flush=True)  # Assuming 'audio' contains audio data

ai_assistant = AI_Assistant()
ai_assistant.start_transcription()
