# python predict_on_audio.py

import torch
import argparse
import numpy as np
from MSnet.MelodyExtraction import MeExt
import os
from MSnet import cfp

from pypianoroll import Multitrack, Track


def write_midi(
    filepath,
    pianorolls,
    program_nums=None,
    is_drums=None,
    track_names=None,
    velocity=100,
    tempo=170.0,
    beat_resolution=16,
):
    # if not os.path.exists(filepath):
    #    os.makedirs(filepath)

    if not np.issubdtype(pianorolls.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")
    if isinstance(program_nums, int):
        program_nums = [program_nums]
    if isinstance(is_drums, int):
        is_drums = [is_drums]
    if pianorolls.shape[2] != len(program_nums):
        raise ValueError("`pianorolls` and `program_nums` must have the same" "length")
    if pianorolls.shape[2] != len(is_drums):
        raise ValueError("`pianorolls` and `is_drums` must have the same" "length")
    if program_nums is None:
        program_nums = [0] * len(pianorolls)
    if is_drums is None:
        is_drums = [False] * len(pianorolls)

    multitrack = Multitrack(beat_resolution=beat_resolution, tempo=tempo)
    for idx in range(pianorolls.shape[2]):
        # plt.subplot(10,1,idx+1)
        # plt.imshow(pianorolls[..., idx].T,cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
        if track_names is None:
            track = Track(pianorolls[..., idx], program_nums[idx], is_drums[idx])
        else:
            track = Track(
                pianorolls[..., idx], program_nums[idx], is_drums[idx], track_names[idx]
            )
        multitrack.append_track(track)
    # plt.savefig(cf.MP3Name)
    multitrack.write(filepath)


def smoothing(roll):
    #     step1  Turn consecutively pitch labels into notes.
    new_map = np.zeros(roll.shape)
    min_note_frames = 3
    last_midinote = 0
    count = 0
    for i in range(len(roll)):
        midinote = np.argmax(roll[i, :])
        if midinote > 0 and midinote == last_midinote:
            count += 1
        else:
            if count >= min_note_frames:
                new_map[i - count - 1 : i, last_midinote] = 1
            last_midinote = midinote
            count = 0
    note_map = new_map
    else_map = roll - note_map
    #     Step2  Connect the breakpoint near the note.
    new_map = np.zeros(roll.shape)
    for i in range(len(else_map)):
        midinote = np.argmax(else_map[i, :])
        if midinote > 0:
            if note_map[i - 1, midinote - 1] > 0:
                new_map[i, midinote - 1] = 1
                else_map[i, midinote] = 0
            elif note_map[i - 1, midinote + 1] > 0:
                new_map[i, midinote + 1] = 1
                else_map[i, midinote] = 0
            elif (i + 1) < len(else_map) and note_map[i + 1, midinote - 1] > 0:
                new_map[i, midinote - 1] = 1
                else_map[i, midinote] = 0
            elif (i + 1) < len(else_map) and note_map[i + 1, midinote + 1] > 0:
                new_map[i, midinote + 1] = 1
                else_map[i, midinote] = 0
    note_map = note_map + new_map
    #     step3  Turn vibrato pitch labels into notes.
    new_map = np.zeros(roll.shape)
    min_note_frames = 3
    last_midinote = 0
    note_list = []
    count = 0
    for i in range(len(else_map)):
        midinote = np.argmax(else_map[i, :])
        if midinote > 0 and np.abs(midinote - last_midinote) <= 1:
            last_midinote = midinote
            note_list.append(midinote)
            count += 1
        else:
            if count >= min_note_frames:
                median_note = note_list[int((len(note_list) / 2))]
                new_map[i - count - 1 : i, median_note] = 1
                else_map[i - count - 1 : i, :] = 0
            last_midinote = midinote
            note_list = []
            count = 0

    note_map = note_map + new_map
    #     step4  Connect nearby notes with the same pitch label.
    last_midinote = 0
    for i in range(len(note_map)):
        midinote = np.argmax(note_map[i, :])
        if last_midinote != 0 and midinote == 0:
            if (i + 1) < len(note_map) and np.argmax(
                note_map[i + 1, :]
            ) == last_midinote:
                note_map[i, last_midinote] = 1
            elif (i + 2) < len(note_map) and np.argmax(
                note_map[i + 2, :]
            ) == last_midinote:
                note_map[i : i + 2, last_midinote] = 1
            elif (i + 3) < len(note_map) and np.argmax(
                note_map[i + 3, :]
            ) == last_midinote:
                note_map[i : i + 3, last_midinote] = 1
            elif (i + 4) < len(note_map) and np.argmax(
                note_map[i + 4, :]
            ) == last_midinote:
                note_map[i : i + 4, last_midinote] = 1
            elif (i + 5) < len(note_map) and np.argmax(
                note_map[i + 5, :]
            ) == last_midinote:
                note_map[i : i + 5, last_midinote] = 1
            last_midinote = midinote
        else:
            last_midinote = midinote
    return note_map


def seq2roll(seq):
    roll = np.zeros((len(seq), 128))

    for i, item in enumerate(seq):
        if item > 0:
            midinote = int(round(cfp.hz2midi(item)))
            roll[i, midinote] = 1

    roll = smoothing(roll)
    return roll


def main(filepath, model_type, output_dir, gpu_index, evaluate, mode):
    songname = "input/ABJones_1_lyrics.wav"
    model_path = "MSnet/pretrain_model/model_vocal"


    est_arr = MeExt(
        songname, model_type=model_type, model_path=model_path, GPU=False, mode=mode
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    txt_songname = songname.replace(".wav", ".txt")
    txt_songname = txt_songname.replace("input/", output_dir)
    print("Save the result in " + txt_songname)
    np.savetxt(txt_songname, est_arr)
    txt_songname = txt_songname.replace(".txt", ".mid")
    rolls = seq2roll(est_arr[:, 1])
    print("Saving MIDI output in Path: " + txt_songname)
    write_midi(
        txt_songname,
        np.expand_dims(rolls.astype(bool), 2),
        program_nums=[0],
        is_drums=[False],
    )


def parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "-fp",
        "--filepath",
        help="Path to input audio (default: %(default)s",
        type=str,
        default="train01.wav",
    )
    p.add_argument(
        "-t",
        "--model_type",
        help="Model type: vocal or melody (default: %(default)s",
        type=str,
        default="vocal",
    )
    p.add_argument(
        "-gpu",
        "--gpu_index",
        help="Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s",
        type=int,
        default=None,
    )
    p.add_argument(
        "-o",
        "--output_dir",
        help="Path to output folder (default: %(default)s",
        type=str,
        default="output/",
    )
    p.add_argument(
        "-e",
        "--evaluate",
        help="Path of ground truth (default: %(default)s",
        type=str,
        default=None,
    )
    p.add_argument(
        "-m",
        "--mode",
        help="The mode of CFP: std and fast (default: %(default)s",
        type=str,
        default="std",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parser()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    main(
        args.filepath,
        args.model_type,
        args.output_dir,
        args.gpu_index,
        args.evaluate,
        args.mode,
    )
