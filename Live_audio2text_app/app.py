import streamlit as st
import pyaudio
import torch
import numpy as np
import os
import threading
import queue
import time
from datetime import datetime
from scipy.signal import resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration



# Initialize PyAudio
p = pyaudio.PyAudio()


# Function to list all input devices
def list_audio_devices():
    device_list = []
    for idx in range(p.get_device_count()):
        device_list.append(p.get_device_info_by_index(idx)['name'])
    return device_list

# Function to cache the model and processor
@st.cache_resource          
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to('cuda')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model, device

# Load Whisper model and processor
processor, model, device = load_whisper_model()

# Function to process audio and generate transcription
def transcribe_audio(audio_chunk, processor, model, language, task):
    input_features = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to(device), language=language, task=task)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Initialize session state to store transcriptions
if "transcriptions" not in st.session_state:
    st.session_state["transcriptions"] = ""

# Function to append transcription to session state

# Thread-safe buffer for transcriptions (global, not in session_state)
global transcription_buffer
transcription_buffer = None
def update_transcription(new_text, time_taken):
    if transcription_buffer is not None:
        transcription_buffer.append(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {new_text}----{time_taken:.2f}s\n")


# Streamlit UI
st.title(":studio_microphone: Live Audio Transcription App")
st.info("After you finish talking, please wait at least 5 seconds before pressing the 'Stop Transcription' button to ensure all audio is processed.")

# Allow user to select input device
devices = list_audio_devices()
input_device = st.selectbox("Select Microphone", devices)

# Select language and task
languages = ['English', 'French', 'Spanish'] # Choose the source language # , 'Persian' (removing Persian because of low accuracy)
tasks = ['transcribe', 'translate'] # when you chose translate -> it means translation to english

language = st.selectbox("Choose the language of the audio", options=languages)
st.write("**When you choose 'translate', it translates the audio to English**.")
task = st.selectbox("Choose the task", options=tasks)

# Add slider for TRANSCRIPTION_INTERVAL (max 30, default 20)
TRANSCRIPTION_INTERVAL = st.slider("Set Transcription Interval (in seconds)", min_value=5, max_value=30, value=20)



# Create button states
col1, col2, col3 = st.columns(3)
with col1:
    new_recording_button = st.button('New Recording')
with col2:
    start_button = st.button('Start Transcription')
with col3:
    stop_button = st.button('Stop Transcription')

# Add Save Transcription button
save_transcription = st.button('Save Transcription')




# Handle new recording button
if new_recording_button:
    st.session_state["transcriptions"] = ""
    st.session_state["_rerun"] = True
    st.write("Ready for new recording. Press 'Start Transcription' to begin.")

# Handle save transcription button
if save_transcription:
    if st.session_state["transcriptions"]:
        st.download_button(
            label="Download Transcription",
            data=st.session_state["transcriptions"],
            file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.warning("No transcription to save.")

# Start transcription logic

# --- Threaded audio recording and transcription ---
def audio_record_worker(stream, RATE, INTERVAL, audio_queue, stop_event):
    chunk_size = RATE * INTERVAL
    while not stop_event.is_set():
        data = stream.read(chunk_size, exception_on_overflow=False)
        audio_queue.put(data)

def transcription_worker(audio_queue, processor, model, device, language, task, live_text, stop_event):
    while not stop_event.is_set() or not audio_queue.empty():
        try:
            data = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        audio_chunk = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / RATE))
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
        task_start = time.time()
        transcription_text = transcribe_audio(audio_chunk, processor, model, language=language, task=task)
        time_taken = time.time() - task_start
        update_transcription(transcription_text, time_taken)

if start_button:
    input_device_index = devices.index(input_device)
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = int(p.get_device_info_by_index(input_device_index)['defaultSampleRate'])
    WHISPER_RATE = 16000  # Whisper expects 16kHz input
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=input_device_index)
    st.write(f"Listening for audio from **{input_device}**... Speak now. :studio_microphone:")
    st.write(f"Transcription interval is **{TRANSCRIPTION_INTERVAL}s** :hourglass_flowing_sand:")
    live_text = st.empty()  # Placeholder for live transcription text
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    # Clear buffer at start
    transcription_buffer = []
    st.session_state['transcriptions'] = ""
    record_thread = threading.Thread(target=audio_record_worker, args=(stream, RATE, TRANSCRIPTION_INTERVAL, audio_queue, stop_event))
    transcribe_thread = threading.Thread(target=transcription_worker, args=(audio_queue, processor, model, device, language, task, live_text, stop_event))
    record_thread.start()
    transcribe_thread.start()

    while not stop_button:
        # Update UI and session_state with current buffer
        st.session_state['transcriptions'] = ''.join(transcription_buffer)
        live_text.markdown(f"{st.session_state['transcriptions']}")
        time.sleep(0.1)

    stop_event.set()
    record_thread.join()
    transcribe_thread.join()

    # Final update after threads finish
    st.session_state['transcriptions'] = ''.join(transcription_buffer)
    live_text.markdown(f"{st.session_state['transcriptions']}")
    st.write("Stopped listening.")
    stream.stop_stream()
    stream.close()
    p.terminate()



