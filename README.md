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
