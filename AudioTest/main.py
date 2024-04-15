# //https://realpython.com/python-speech-recognition/



import speech_recognition as sr
import librosa

def convert_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    return {
        "mfccs": mfccs,
        "tempo": tempo,
        "spectral_centroid": spectral_centroids,
        "duration": librosa.get_duration(y=y, sr=sr),
        "sample_rate": sr,
        "chroma_stft": chroma_stft,
        "spectral_contrast": spectral_contrast,
        "tonnetz": tonnetz
    }

def analyze_features(features):
    analysis_result = "Analysis Results:\n"
    
    # Tempo analysis
    analysis_result += f"- Tempo: {'Fast' if features['tempo'] > 120 else 'Slow'}\n"
    
    # Brightness based on spectral centroid
    bright = features['spectral_centroid'].mean() > 3000
    analysis_result += f"- Sound: {'Bright' if bright else 'Dark'}\n"
    
    # Harmony based on chroma
    harmony = features['chroma_stft'].mean(axis=1).argmax()
    analysis_result += f"- Predominant Pitch Class (Harmony): {harmony}\n"
    
    # Contrast
    highest_contrast_band = features['spectral_contrast'].mean(axis=1).argmax()
    analysis_result += f"- Band with Highest Spectral Contrast: {highest_contrast_band}\n"
    
    # Tonal Centroid (Tonnetz)
    tonal_centroid_means = features['tonnetz'].mean(axis=1)
    analysis_result += "- Tonal Centroids (Tonnetz): " + ', '.join([f"{x:.2f}" for x in tonal_centroid_means]) + "\n"
    
    return analysis_result

def main():
    audio_path = 'audioFile/hello_john_doe.wav' # Update this to the path of your audio file
    text = convert_audio_to_text(audio_path)
    print(f"Converted Text: {text}")
    
    features = extract_audio_features(audio_path)
    analysis_result = analyze_features(features)
    result = {
        "text": text,
        "features": analysis_result
    }
    
    print(result)
    print(analysis_result)

if __name__ == '__main__':
    main()