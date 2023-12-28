"""
Created on Dec 24,2023

@author: sutirtha
"""

import crepe
import pandas as pd
import os
import glob
from tqdm import tqdm
from scipy.io import wavfile
from typing import Optional, Union, Tuple, List


def process_wav_files(folder_path):
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        csv_file = wav_file.replace(".wav", ".csv")

        # Skip if CSV already exists
        if os.path.exists(csv_file):
            print(f"CSV for {wav_file} already exists. Skipping.")
            continue

        try:
            sr, audio = wavfile.read(wav_file)
            # Process with CREPE
            time, frequency, confidence, activation = crepe.predict(
                audio, sr=sr, viterbi=True
            )

            # Create DataFrame and save as CSV
            df = pd.DataFrame({"time": time, "frequency": frequency})
            df.to_csv(csv_file, index=False)

            print(f"CSV file created for: {wav_file}")
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")


# Specify your folder path here
folder_path = "MIR-1K/LyricsWav"
process_wav_files(folder_path)
