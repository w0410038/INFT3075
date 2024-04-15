from gtts import gTTS
from pydub import AudioSegment
import os

def text_to_speech(text, mp3_path, wav_path):
    # Generate speech
    tts = gTTS(text=text, lang='en')
    tts.save(mp3_path)
    
    # Convert to WAV
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    
    print(f"Audio file saved as {wav_path}")

# Example usage
text = "Hello, my name is John Doe."
mp3_path = 'audioFile/hello_john_doe.mp3'  # Temp path for MP3 file
wav_path = 'audioFile/hello_john_doe.wav'  # Final WAV file path
text_to_speech(text, mp3_path, wav_path)