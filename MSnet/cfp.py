# -*- coding: utf-8 -*-
"""
Created on Dec 18, 2023

@author: Sutirtha

Document:

load_audio(filepath, sr=None, mono=True, dtype='float32')
    Parameters:
        sr:(number>0) sample rate;
            default = None(use raw audio sample rate)
        mono:(bool) convert signal to mono;
            default = True
        dtype:(numeric type) data type of x;
            default = 'float32'
    Returns:
        x:(np.ndarray) audio time series
        sr:(number>0) sample rate of x
feature_extraction(x, sr, Hop=320, Window=2049, StartFreq=80.0, StopFreq=1000.0, NumPerOct=48)
    Parameters:
        x:(np.ndarray) audio time series
        sr:(number>0) sample rate of x
        Hop: Hop size
        Window: Window size
        StartFreq: smallest frequency on feature map
        StopFreq: largest frequency on feature map
        NumPerOct: Number of bins per octave
    Returns:
        Z: mix cfp feature
        time: feature map to time
        CenFreq: feature map to frequency
        tfrL0: STFT spectrogram
        tfrLF: generalized cepstrum (GC)
        tfrLQ: generalized cepstrum of spectrum (GCOS)

get_CenFreq(StartFreq=80, StopFreq=1000, NumPerOct=48)
get_time(fs, Hop, end)
midi2hz(midi)
hz2midi(hz)

"""
import soundfile as sf
import numpy as np
from typing import Optional, Union, Tuple, List

np.seterr(divide="ignore", invalid="ignore")
import scipy
import scipy.signal
import pandas as pd
import librosa
import numpy as np


def STFT(x: np.ndarray, fr: float, fs: int, Hop: int, h: np.ndarray) -> tuple:
    """
    Computes the Short-Time Fourier Transform (STFT) of an input signal.

    Args:
    x (np.ndarray): Input signal.
    fr (float): Frequency resolution.
    fs (int): Sampling frequency of the input signal.
    Hop (int): Hop size between successive frames.
    h (np.ndarray): Window function applied to each frame.

    Returns:
    tuple: A tuple (tfr, f, t, N) where 'tfr' is the computed STFT, 'f' is the array of frequency bins, 't' is the array of time bins, and 'N' is the number of frequency bins.
    """
    t = np.arange(Hop, np.ceil(len(x) / float(Hop)) * Hop, Hop)
    N = int(fs / float(fr))
    window_size = len(h)
    # f = fs*np.linspace(0, 0.5, np.round(N/2), endpoint=True)
    f = fs * np.linspace(0, 0.5, int(np.round(N / 2)), endpoint=True)

    Lh = int(np.floor(float(window_size - 1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=float)

    for icol in range(0, len(t)):
        ti = int(t[icol])
        tau = np.arange(
            int(-min([round(N / 2.0) - 1, Lh, ti - 1])),
            int(min([round(N / 2.0) - 1, Lh, len(x) - ti])),
        )
        indices = np.mod(N + tau, N) + 1
        tfr[indices - 1, icol] = (
            x[ti + tau - 1] * h[Lh + tau - 1] / np.linalg.norm(h[Lh + tau - 1])
        )

    tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))
    return tfr, f, t, N


def nonlinear_func(X: np.ndarray, g: float, cutoff: int) -> np.ndarray:
    """
    Applies a nonlinear function to the input matrix, with optional frequency cutoff.

    Args:
    X (np.ndarray): Input matrix (e.g., a spectrogram).
    g (float): Exponent for the nonlinearity. If g is 0, a logarithmic function is applied.
    cutoff (int): Frequency bins below and above this index are set to zero.

    Returns:
    np.ndarray: The transformed matrix after applying the nonlinearity and frequency cutoff.
    """
    cutoff = int(cutoff)
    if g != 0:
        X[X < 0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X


def Freq2LogFreqMapping(
    tfr: np.ndarray, f: np.ndarray, fr: float, fc: float, tc: float, NumPerOct: int
) -> tuple:
    """
    Maps a linear frequency spectrogram to a logarithmic frequency scale.

    Args:
    tfr (np.ndarray): The input spectrogram.
    f (np.ndarray): Array of linear frequency bins.
    fr (float): Frequency resolution.
    fc (float): Lowest center frequency.
    tc (float): Time constant determining the highest center frequency.
    NumPerOct (int): Number of bins per octave.

    Returns:
    tuple: A tuple (tfrL, central_freq) where 'tfrL' is the transformed spectrogram and 'central_freq' is an array of central frequencies in the logarithmic scale.
    """
    StartFreq = fc
    StopFreq = 1 / tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=float)
    for i in range(1, Nest - 1):
        l = int(round(central_freq[i - 1] / fr))
        r = int(round(central_freq[i + 1] / fr) + 1)
        # rounding1
        if l >= r - 1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (
                        central_freq[i] - central_freq[i - 1]
                    )
                elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                    freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (
                        central_freq[i + 1] - central_freq[i]
                    )
    tfrL = np.dot(freq_band_transformation, tfr)
    return tfrL, central_freq


def Quef2LogFreqMapping(
    ceps: np.ndarray, q: np.ndarray, fs: int, fc: float, tc: float, NumPerOct: int
) -> tuple:
    """
    Maps a quefrency domain representation (like cepstrum) to a logarithmic frequency scale.

    Args:
    ceps (np.ndarray): The input quefrency domain representation.
    q (np.ndarray): Array of quefrency bins.
    fs (int): Sampling frequency.
    fc (float): Lowest center frequency.
    tc (float): Time constant determining the highest center frequency.
    NumPerOct (int): Number of bins per octave.

    Returns:
    tuple: A tuple (tfrL, central_freq) where 'tfrL' is the transformed quefrency representation and 'central_freq' is an array of central frequencies in the logarithmic scale.
    """
    StartFreq = fc
    StopFreq = 1 / tc
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    f = 1 / q
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=float)
    for i in range(1, Nest - 1):
        for j in range(
            int(round(fs / central_freq[i + 1])),
            int(round(fs / central_freq[i - 1]) + 1),
        ):
            if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (
                    central_freq[i] - central_freq[i - 1]
                )
            elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (
                    central_freq[i + 1] - central_freq[i]
                )

    tfrL = np.dot(freq_band_transformation, ceps)
    return tfrL, central_freq


def CFP_filterbank(
    x: np.ndarray,
    fr: float,
    fs: int,
    Hop: int,
    h: np.ndarray,
    fc: float,
    tc: float,
    g: np.ndarray,
    NumPerOctave: int,
) -> tuple:
    """
    Applies a Constant-Q Transform-like filterbank (CFP) to the input signal.

    Args:
    x (np.ndarray): Input signal.
    fr (float): Frequency resolution.
    fs (int): Sampling frequency.
    Hop (int): Hop size between frames.
    h (np.ndarray): Window function.
    fc (float): Lowest center frequency for the filterbank.
    tc (float): Time constant for the filterbank.
    g (np.ndarray): Array of gamma values for nonlinear transformation.
    NumPerOctave (int): Number of filters per octave.

    Returns:
    tuple: A tuple containing various frequency and quefrency representations of the input signal.
    """
    NumofLayer = np.size(g)

    [tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
    tfr = np.power(abs(tfr), g[0])
    tfr0 = tfr  # original STFT
    ceps = np.zeros(tfr.shape)

    if NumofLayer >= 2:
        for gc in range(1, NumofLayer):
            if np.remainder(gc, 2) == 1:
                tc_idx = round(fs * tc)
                ceps = np.real(np.fft.fft(tfr, axis=0)) / np.sqrt(N)
                ceps = nonlinear_func(ceps, g[gc], tc_idx)
            else:
                fc_idx = round(fc / fr)
                tfr = np.real(np.fft.fft(ceps, axis=0)) / np.sqrt(N)
                tfr = nonlinear_func(tfr, g[gc], fc_idx)

    tfr0 = tfr0[: int(round(N / 2)), :]
    tfr = tfr[: int(round(N / 2)), :]
    ceps = ceps[: int(round(N / 2)), :]

    HighFreqIdx = int(round((1 / tc) / fr) + 1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx, :]
    tfr = tfr[:HighFreqIdx, :]
    HighQuefIdx = int(round(fs / fc) + 1)

    q = np.arange(HighQuefIdx) / float(fs)

    ceps = ceps[:HighQuefIdx, :]

    tfrL0, central_frequencies = Freq2LogFreqMapping(tfr0, f, fr, fc, tc, NumPerOctave)
    tfrLF, central_frequencies = Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
    tfrLQ, central_frequencies = Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)

    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies


def load_audio(
    filepath: str, sr: Optional[int] = None, mono: bool = True, dtype: str = "float32"
) -> tuple:
    """
    Loads an audio file from a given path and optionally converts it to mono and resamples.

    Args:
    filepath (str): Path to the audio file.
    sr (Optional[int]): Target sampling rate for resampling. If None, the original sampling rate is used.
    mono (bool): If True, converts the audio to mono. Default is True.
    dtype (str): Data type for the returned audio array. Default is 'float32'.

    Returns:
    tuple: A tuple (x, fs) where 'x' is the loaded audio signal and 'fs' is its sampling frequency.
    """
    if ".mp3" in filepath:
        from pydub import AudioSegment
        import tempfile
        import os

        mp3 = AudioSegment.from_mp3(filepath)
        _, path = tempfile.mkstemp()
        mp3.export(path, format="wav")
        del mp3
        x, fs = sf.read(path)
        os.remove(path)
    else:
        x, fs = sf.read(filepath)

    if mono and len(x.shape) > 1:
        x = np.mean(x, axis=1)
    if sr:
        x = scipy.signal.resample_poly(x, sr, fs)
        fs = sr
    x = x.astype(dtype)

    return x, fs


def feature_extraction(
    x: np.ndarray,
    fs: int,
    Hop: int = 512,
    Window: int = 2049,
    StartFreq: float = 80.0,
    StopFreq: float = 1000.0,
    NumPerOct: int = 48,
) -> tuple:
    """
    Extracts features from an audio signal using a filterbank approach.

    Args:
    x (np.ndarray): Input audio signal.
    fs (int): Sampling frequency of the signal.
    Hop (int): Hop size for STFT. Default is 512.
    Window (int): Window size for STFT. Default is 2049.
    StartFreq (float): Start frequency for the filterbank. Default is 80.0 Hz.
    StopFreq (float): Stop frequency for the filterbank. Default is 1000.0 Hz.
    NumPerOct (int): Number of filters per octave. Default is 48.

    Returns:
    tuple: A tuple containing various transformed representations of the input signal.
    """
    fr = 2.0  # frequency resolution
    h = scipy.signal.blackmanharris(Window)  # window size
    g = np.array([0.24, 0.6, 1])  # gamma value

    tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(
        x, fr, fs, Hop, h, StartFreq, 1 / StopFreq, g, NumPerOct
    )
    Z = tfrLF * tfrLQ
    time = t / fs
    return Z, time, CenFreq, tfrL0, tfrLF, tfrLQ


def midi2hz(midi: np.ndarray) -> np.ndarray:
    """
    Converts MIDI note numbers to frequencies in Hz.

    Args:
    midi (np.ndarray): Array of MIDI note numbers.

    Returns:
    np.ndarray: Array of corresponding frequencies in Hz.
    """

    return 2 ** ((midi - 69) / 12.0) * 440


def hz2midi(hz: np.ndarray) -> np.ndarray:
    """
    Converts frequencies in Hz to MIDI note numbers.

    Args:
    hz (np.ndarray): Array of frequencies in Hz.

    Returns:
    np.ndarray: Array of corresponding MIDI note numbers.
    """
    return 69 + 12 * np.log2(hz / 440.0)


def get_CenFreq(
    StartFreq: float = 80, StopFreq: float = 1000, NumPerOct: int = 48
) -> list:
    """
    Generates a list of central frequencies for a given frequency range and resolution.

    Args:
    StartFreq (float): Start frequency for the range. Default is 80 Hz.
    StopFreq (float): Stop frequency for the range. Default is 1000 Hz.
    NumPerOct (int): Number of central frequencies per octave. Default is 48.

    Returns:
    list: A list of central frequencies within the specified range.
    """
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []
    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    return central_freq


def get_time(fs: int, Hop: int, end: int) -> np.ndarray:
    """
    Generates a time array for an audio signal based on its sampling frequency and hop size.

    Args:
    fs (int): Sampling frequency of the audio signal.
    Hop (int): Hop size.
    end (int): The length of the audio signal.

    Returns:
    np.ndarray: An array of time points corresponding to the audio signal.
    """
    return np.arange(Hop / fs, end, Hop / fs)


def lognorm(x: np.ndarray) -> np.ndarray:
    """
    Applies logarithmic normalization to the input array.

    Args:
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: Logarithmically normalized array.
    """
    return np.log(1 + x)


def norm(x: np.ndarray) -> np.ndarray:
    """
    Normalizes the input array to the range [0, 1]. If the input array is empty or has zero variance, it returns the array as is.

    Args:
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: Normalized array, or the original array if it's empty or has zero variance.
    """
    if x.size == 0 or np.max(x) == np.min(x):
        return x
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def cfp_process(
    fpath: str,
    ypath: Optional[str] = None,
    csv: bool = False,
    sr: Optional[int] = None,
    hop: int = 256,
    model_type: str = "vocal",
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Processes an audio file to extract features using the Constant Q Transform (CQT) and cepstral analysis.

    Args:
    fpath (str): File path of the input audio file.
    ypath (Optional[str]): File path of the ground truth data. If provided, it's used to load the ground truth for comparison. Default is None.
    csv (bool): Indicates if the ground truth data is in CSV format. Default is False.
    sr (Optional[int]): Sampling rate to be used for audio processing. If None, the original sampling rate of the audio file is used. Default is None.
    hop (int): Hop size for the STFT. Default is 256.
    model_type (str): Type of model to use, can be 'vocal' or 'melody'. Default is 'vocal'.

    Returns:
    Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    - If ground truth path (`ypath`) is provided, returns a tuple (W, gt, CenFreq, time), where 'W' is the feature matrix, 'gt' is the ground truth data, 'CenFreq' is the array of central frequencies, and 'time' is the time array.
    - If ground truth path is not provided, returns a tuple (W, CenFreq, time), without the ground truth data.

    The function prints the progress and data shape information during processing.
    """

    print("CFP process in " + str(fpath) + " ... (It may take some times)")
    y, sr = load_audio(fpath, sr=sr)
    if "vocal" in model_type:
        Z, time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(
            y, sr, Hop=hop, StartFreq=31.0, StopFreq=1250.0, NumPerOct=60
        )
    tfrL0 = norm(lognorm(tfrL0))[np.newaxis, :, :]
    tfrLF = norm(lognorm(tfrLF))[np.newaxis, :, :]
    tfrLQ = norm(lognorm(tfrLQ))[np.newaxis, :, :]
    W = np.concatenate((tfrL0, tfrLF, tfrLQ), axis=0)
    print("Done!")
    print("Data shape: " , W.shape)
    if ypath:
        if csv:
            ycsv = pd.read_csv(ypath, names=["time", "frequency"])
            gt0 = ycsv["time"].values
            gt0 = gt0[1:, np.newaxis]

            gt1 = ycsv["frequency"].values
            gt1 = gt1[1:, np.newaxis]
            gt = np.concatenate((gt0, gt1), axis=1)
        else:
            gt = np.loadtxt(ypath)
        return W, gt, CenFreq, time
    else:
        return W, CenFreq, time
