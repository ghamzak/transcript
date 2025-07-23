from pydub import AudioSegment
import sys
import os

def m4a_to_mp3(m4a_path, mp3_path=None):
    """
    Convert an m4a file to mp3 using pydub.
    Args:
        m4a_path (str): Path to the input m4a file.
        mp3_path (str, optional): Path to save the output mp3 file. If None, saves as same name with .mp3 extension.
    Returns:
        str: Path to the converted mp3 file.
    """
    if not mp3_path:
        mp3_path = os.path.splitext(m4a_path)[0] + ".mp3"
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(mp3_path, format="mp3")
    return mp3_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python m4a_to_mp3.py <input.m4a> [output.mp3]")
        sys.exit(1)
    m4a_path = sys.argv[1]
    mp3_path = sys.argv[2] if len(sys.argv) > 2 else None
    out_path = m4a_to_mp3(m4a_path, mp3_path)
    print(f"Converted: {m4a_path} -> {out_path}")
