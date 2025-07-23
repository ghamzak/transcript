import streamlit as st
import torch
from transformers.pipelines import pipeline
from datetime import datetime


st.title("Audio2Text: Audio Transcription App")
st.write("Upload an MP3 or M4A file and get the transcription as a downloadable text file.")


# File uploader
uploaded_file = st.file_uploader("Choose an MP3 or M4A file", type=["mp3", "m4a"])


# Language and task options
language = st.selectbox("Language of the audio", options=["English"])  # You can add more languages if needed
task = st.selectbox("Task", options=["transcribe", "translate"])


if uploaded_file is not None:
    import os
    from m4a_to_mp3 import m4a_to_mp3
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension == ".mp3":
        temp_mp3 = "temp_uploaded_audio.mp3"
        with open(temp_mp3, "wb") as f:
            f.write(uploaded_file.read())
    elif file_extension == ".m4a":
        temp_m4a = "temp_uploaded_audio.m4a"
        temp_mp3 = "temp_uploaded_audio.mp3"
        with open(temp_m4a, "wb") as f:
            f.write(uploaded_file.read())
        # Convert m4a to mp3
        m4a_to_mp3(temp_m4a, temp_mp3)
    else:
        st.error("Unsupported file type. Please upload an MP3 or M4A file.")
        st.stop()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipeline(
        "automatic-speech-recognition",
        "openai/whisper-small",
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
        device=device,
        generate_kwargs={"language": language, "task": task}
    )

    st.info("Transcribing audio... This may take a while for long files.")
    transcription = pipe(temp_mp3)


    formatted_lyrics = ""
    chunks = None
    text = None
    # Try to get 'chunks' and 'text' from dict or object
    if isinstance(transcription, dict):
        chunks = transcription.get('chunks', None)
        text = transcription.get('text', None)
    else:
        chunks = getattr(transcription, 'chunks', None)
        text = getattr(transcription, 'text', None)

    if chunks:
        for line in chunks:
            line_text = line.get("text", "") if isinstance(line, dict) else str(line)
            ts = line.get("timestamp", "") if isinstance(line, dict) else ""
            formatted_lyrics += f"{ts}-->{line_text}\n"
    elif text:
        formatted_lyrics = text if isinstance(text, str) else str(text)
    else:
        formatted_lyrics = str(transcription)

    st.success("Transcription complete!")
    st.text_area("Transcription", formatted_lyrics.strip(), height=300)

    st.download_button(
        label="Download Transcription as .txt",
        data=formatted_lyrics.strip(),
        file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
