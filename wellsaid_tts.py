import aiohttp
import asyncio
import json
import base64
from aiohttp import MultipartReader
URL = "https://api.wellsaidlabs.com/v1/tts/stream"
headers = {"x-api-key": "3dc733a2-874f-4a0e-b239-201c06dc89f2",
           "Accept": "multipart/mixed"}


def generate_tts_with_wordtimings_sync(speaker_id:int, text:str):
    return asyncio.run(generate_tts_with_wordtimings(speaker_id, text))

async def generate_tts_with_wordtimings(speaker_id: int, text: str):
    async with aiohttp.ClientSession() as session:
        """Request multipart/mixed stream and extract audio + word timing parts."""
        async with session.post(URL,
                                headers=headers,
                                json={"speaker_id": speaker_id, "text": text, "include_word_timing": True}) as resp:
            print(f"[multipart] Status: {resp.status}, Content-Type: {resp.headers.get('Content-Type')}")
            reader = MultipartReader.from_response(resp)

            audio_path = "tts_multipart.mp3"
            timing_path = "tts_multipart_word_timing.json"
            timings = []

            with open(audio_path, "wb") as f:
                async for part in reader:
                    ctype = part.headers.get("Content-Type", "")
                    data = await part.read()
                    if ctype.startswith("audio/"):
                        f.write(data)
                    elif ctype.startswith("application/json"):
                        try:
                            timings.append(json.loads(data.decode()))
                        except Exception as e:
                            print(f"[multipart] Error parsing timing JSON: {e}")

            if timings:
                with open(timing_path, "w") as tf:
                    json.dump(timings, tf, indent=2)

            print(f"[multipart] Saved {audio_path} and {timing_path}")


def generate_tts_sync(speaker_id:int, text:str) -> bytes:
    return asyncio.run(generate_tts(speaker_id, text))

async def generate_tts(speaker_id: int, text: str) -> bytes:
    async with aiohttp.ClientSession() as session:

        headers = {"x-api-key": "3dc733a2-874f-4a0e-b239-201c06dc89f2"}

        async with session.post(URL,
                                headers=headers,
                                json={"speaker_id": speaker_id, "text": text, "model":"caruso"}) as resp:
           if resp.status != 200:
                body = await resp.read()
                print("Error response body:", body.decode())
                raise Exception(f"TTS request failed with status {resp.status}. {body.decode()}")
           bytes = await resp.read()

        return bytes
