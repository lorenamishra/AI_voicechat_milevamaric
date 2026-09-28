import json
import os
from io import BytesIO
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from faster_whisper import WhisperModel

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)
chat = client.chats.create(
    model="gemini-3.1-flash-lite",
    config=types.GenerateContentConfig(
        system_instruction="My name is Lorena",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)
transcriber = None


class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return

        page = Path(__file__).with_name("index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        if self.path == "/transcribe":
            self.transcribe_audio()
            return
        if self.path != "/chat":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            message = body.get("message", "").strip()
            if not message:
                self.send_json({"error": "Please type a message."}, 400)
                return

            response = chat.send_message(message)
            self.send_json({"reply": response.text})
        except Exception as error:
            self.send_json({"error": str(error)}, 500)

    def transcribe_audio(self):
        global transcriber
        try:
            if transcriber is None:
                transcriber = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            length = int(self.headers.get("Content-Length", 0))
            body = BytesIO(self.rfile.read(length))
            content_type = self.headers.get("Content-Type", "")
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body.read()
            )
            audio = next(
                part.get_payload(decode=True)
                for part in message.iter_parts()
                if part.get_param("name", header="content-disposition") == "audio"
            )
            audio_path = Path(__file__).with_name(".voice-input.webm")
            audio_path.write_bytes(audio)
            segments, _ = transcriber.transcribe(str(audio_path), language="en")
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            audio_path.unlink(missing_ok=True)
            self.send_json({"text": transcript})
        except Exception as error:
            self.send_json({"error": str(error)}, 500)

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", 8000), ChatHandler)
    print("Lorena is ready at http://localhost:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        server.server_close()