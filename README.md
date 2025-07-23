# Gen-AI-Mini-Projects

## Setup Notes

**Important:** For audio processing (mp3/m4a conversion and transcription), use **Python 3.12 or lower**. Python 3.13+ is not supported due to removal of the `audioop` module, which is required by `pydub` and other audio libraries.

Recommended: Install Python 3.12 and create your virtual environment with it.
Gen AI Mini Projects


## Note for macOS Users

If you encounter errors when installing `PyAudio` (such as `portaudio.h file not found`), you need to install the PortAudio library using Homebrew:

```sh
brew install portaudio
```

After installing PortAudio, re-run:

```sh
pip install -r requirements.txt
```

This will allow `PyAudio` to build and install successfully on macOS.


## Running the Apps

**Note:** When running the Streamlit app for the first time, it may take a while to start because it needs to download about 200MB of data before loading. Please be patient during the initial startup.

### How to Run the Apps

- To run the live transcription app (using your microphone):

  ```sh
  cd Live_audio2text_app
  streamlit run app.py
  ```

- To run the audio-to-text app for MP3/M4A files:

  ```sh
  cd audio2text_from_mp3
  streamlit run streamlit_app.py
  ```
