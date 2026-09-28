from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from RealtimeSTT import AudioToTextRecorder
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

MAX_OUTPUT_TOKENS = 100

def main():
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    if not elevenlabs_api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set.")    

   
    print("API KEY IS FINEEE")
    
    client = genai.Client(api_key=gemini_api_key)
    elevenlabs=ElevenLabs(
        api_key=elevenlabs_api_key,
    )
    
    chat = client.chats.create(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="My name is Lorena",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )

    recorder = AudioToTextRecorder(
        model="tiny.en",
        language="en",
        spinner=False,
        sample_rate=16000,
        silero_backend="pytorch_cpu",
    )

    while True:
        print("You:", end=" ",flush=True)
        user_input = recorder.text()
        print(user_input)
        if user_input.lower() == "exit":
            break

        print("🤖 Sending to Gemini...")
        response = chat.send_message_stream(user_input)
        full_response = []

        for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response.append(chunk.text)
        print()
        if full_response:
            audio = elevenlabs.text_to_speech(
                text="".join(full_response),
                voice_id="pqHfZKP75CvOlQylNhV4",
                model_id="eleven_flash_2_5",
                output_format="mp3_44100_128"
            )
            play(audio)

    recorder.shutdown()    

if __name__ == "__main__":
    main()