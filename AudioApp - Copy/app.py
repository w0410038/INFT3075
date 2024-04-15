from flask import Flask, request, render_template, redirect, url_for
import speech_recognition as sr
import librosa
from gtts import gTTS
from pydub import AudioSegment
import os
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import logging
from flask import flash

# Import your analysis functions here

app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET"])
def home():
    # Render the home page with forms directly instead of redirecting
    return render_template("index.html", converted=False, uploaded=False)


@app.route("/", methods=["POST"])
def upload_files():
    # 파일이 제대로 업로드되었는지 확인

    if request.method == "POST":
        # 파일 업로드 처리
        if "audio1" not in request.files or "audio2" not in request.files:
            flash("Missing files")  # 사용자에게 파일 누락 알림
            return redirect(url_for("home"))

        audio1 = request.files["audio1"]
        audio2 = request.files["audio2"]
        if audio1.filename == "" or audio2.filename == "":
            flash("No selected file")  # Notify user no file was selected
            return redirect(url_for("home"))

            # 파일 저장
        audio_path1 = "./temp_audio1.wav"
        audio_path2 = "./temp_audio2.wav"
        # audio_path2 = os.path.join('./uploads', secure_filename(audio2.filename))
        audio1.save(audio_path1)
        audio2.save(audio_path2)

        # 파일 분석
        text1 = convert_audio_to_text(audio_path1)
        features1 = extract_audio_features(audio_path1)
        analysis_result1 = analyze_features(features1)

        text2 = convert_audio_to_text(audio_path2)
        features2 = extract_audio_features(audio_path2)
        analysis_result2 = analyze_features(features2)

        # 분석 결과와 함께 템플릿 렌더링
        return render_template(
            "index.html",
            uploaded=True,
            analysis1=analysis_result1,
            analysis2=analysis_result2,
        )


@app.route("/convert", methods=["POST"])
def convert():
    text = request.form.get("text")
    if text:
        wav_path = "./static/converted_audio_google.wav"
        text_to_speech(text, wav_path)
        return render_template("index.html", converted_google=True, audio_path=wav_path)
    return redirect("/")


@app.route("/analyze", methods=["POST"])
def analyze():
    audio_path = request.form.get("audio_path")
    if audio_path:
        # Check if the file exists
        if os.path.exists(audio_path):
            # Perform the audio analysis
            text = convert_audio_to_text(audio_path)
            features = extract_audio_features(audio_path)
            analysis_result = analyze_features(features)
            return render_template(
                "index.html",
                text=text,
                analysis=analysis_result,
                uploaded=False,
                converted=True,
                singleAnalyze=True,
            )
        else:
            # Log an error if the file does not exist
            flash("Error: Audio file does not exist.")
            return redirect("/")
    else:
        # Log an error if the audio path is not provided
        flash("Error: No audio path provided.")
        return redirect("/")


@app.route("/convert_vits", methods=["POST"])
def convert_vits():
    text = request.form.get("text")
    if text:
        output_path = "./static/converted_audio_vits.wav"
        text_to_speech_vits(text, output_path)
        # 변환된 오디오 파일 경로와 함께 템플릿 렌더링
        return render_template(
            "index.html", converted_vist=True, audio_path=output_path
        )
    return redirect(url_for("upload_audio"))


def text_to_speech_vits(text, output_path):
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    # 변환된 오디오를 파일로 저장
    sf.write(output_path, output.squeeze().numpy(), 22050)


def text_to_speech(text, wav_path):
    # 임시 MP3 파일 경로 설정
    mp3_path = wav_path.replace(".wav", ".mp3")

    # 텍스트를 오디오(MP3)로 변환하고 저장
    tts = gTTS(text=text, lang="en", tld="com", slow=False)
    tts.save(mp3_path)

    # MP3를 WAV로 변환
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

    # 임시 MP3 파일 삭제
    os.remove(mp3_path)

    print(f"Audio file saved as {wav_path}")


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
            return (
                f"Could not request results from Google Speech Recognition service; {e}"
            )
    pass


def extract_audio_features(audio_path):

    y, sr = librosa.load(audio_path, sr=None)
    n_fft = min(len(y), 1024)

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
        "tonnetz": tonnetz,
    }
    pass


def analyze_features(features):
    analysis_results = [
        f"Tempo: {'Fast' if features['tempo'] > 120 else 'Slow'}",
        f"Sound: {'Bright' if features['spectral_centroid'].mean() > 3000 else 'Dark'}",
        f"Predominant Pitch Class (Harmony): {features['chroma_stft'].mean(axis=1).argmax()}",
        f"Band with Highest Spectral Contrast: {features['spectral_contrast'].mean(axis=1).argmax()}",
        "Tonal Centroids (Tonnetz): "
        + ", ".join([f"{x:.2f}" for x in features["tonnetz"].mean(axis=1)]),
    ]
    return analysis_results


if __name__ == "__main__":
    app.run(debug=True)
