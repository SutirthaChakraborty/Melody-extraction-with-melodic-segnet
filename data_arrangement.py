#  python data_arrangement.py -df MIR-1K/LyricsWav -t vocal -o data

"""
Created on Dec 25,2023
@author: sutirtha
"""

import numpy as np
from MSnet.cfp import cfp_process
from MSnet.utils import (
    getlist_mdb,
    getlist_mdb_vocal,
    select_vocal_track,
    csv2ref,
    split_data_set,
)
import argparse
import h5py
import pickle
from typing import Optional, Union, Tuple, List


def seq2map(seq: np.ndarray, CenFreq: np.ndarray) -> np.ndarray:
    """
    Converts a sequence of frequency values into a binary map based on given center frequencies.

    Args:
    seq (np.ndarray): An array of frequency values.
    CenFreq (np.ndarray): An array of center frequencies to compare against.

    Returns:
    np.ndarray: A binary map where each column corresponds to a frequency in 'seq' and each row corresponds to a center frequency in 'CenFreq'. A value of 1 indicates the frequency in 'seq' is less than or equal to the corresponding center frequency in 'CenFreq'.
    """

    CenFreq[0] = 0
    gtmap = np.zeros((len(CenFreq), len(seq)))
    for i in range(len(seq)):
        for j in range(len(CenFreq)):
            
            if seq[i] < 0.1:
                gtmap[0, i] = 1
                break
            elif CenFreq[j] > seq[i]:
                gtmap[j, i] = 1
                break
    return gtmap


def batchize(
    data: np.ndarray, gt: np.ndarray, xlist: list, ylist: list, size: int = 430
) -> tuple:
    """
    Divides data and ground truth arrays into smaller batches of a specified size.

    Args:
    data (np.ndarray): The input data array.
    gt (np.ndarray): The ground truth data array.
    xlist (list): List to store the batched input data.
    ylist (list): List to store the batched ground truth data.
    size (int, optional): The size of each batch. Default is 430.

    Returns:
    tuple: A tuple containing two lists, 'xlist' with batched input data and 'ylist' with batched ground truth data.
    """

    if data.shape[-1] != gt.shape[-1]:
        new_length = min(data.shape[-1], gt.shape[-1])
        data = data[:, :, :new_length]
        gt = gt[:, :new_length]
    num = int(gt.shape[-1] / size)
    if gt.shape[-1] % size != 0:
        num += 1
    for i in range(num):
        if (i + 1) * size > gt.shape[-1]:
            batch_x = np.zeros((data.shape[0], data.shape[1], size))
            batch_y = np.zeros((gt.shape[0], size))

            tmp_x = data[:, :, i * size :]
            tmp_y = gt[:, i * size :]

            batch_x[:, :, : tmp_x.shape[-1]] += tmp_x
            batch_y[:, : tmp_y.shape[-1]] += tmp_y
            xlist.append(batch_x)
            ylist.append(batch_y)
            break
        else:
            batch_x = data[:, :, i * size : (i + 1) * size]
            batch_y = gt[:, i * size : (i + 1) * size]
            xlist.append(batch_x)
            ylist.append(batch_y)

    return xlist, ylist


def batchize_val(data: np.ndarray, size: int = 430) -> np.ndarray:
    """
    Divides validation data into smaller batches of a specified size.

    Args:
    data (np.ndarray): The input data array for validation.
    size (int, optional): The size of each batch. Default is 430.

    Returns:
    np.ndarray: An array containing the batched validation data.
    """

    xlist = []
    num = int(data.shape[-1] / size)
    if data.shape[-1] % size != 0:
        num += 1
    for i in range(num):
        if (i + 1) * size > data.shape[-1]:
            batch_x = np.zeros((data.shape[0], data.shape[1], size))

            tmp_x = data[:, :, i * size :]

            batch_x[:, :, : tmp_x.shape[-1]] += tmp_x
            xlist.append(batch_x)
            break
        else:
            batch_x = data[:, :, i * size : (i + 1) * size]
            xlist.append(batch_x)

    return np.array(xlist)


def main(data_folder, model_type, output_folder):
    batch_size = 430
    train_songlist, val_songlist, test_songlist = split_data_set()
    xlist = []
    ylist = []
    for songname in train_songlist:
        filepath = data_folder + "/" + songname + ".wav"
        data, CenFreq, time_arr = cfp_process(
            filepath, model_type=model_type, sr=44100, hop=256
        )
        
        ypath = filepath.replace(".wav", ".csv")
        lpath = "data/lpath.csv"
        ref_arr = select_vocal_track(ypath, lpath,time_arr)
        
        gt_map = seq2map(ref_arr[:, 1], CenFreq)
        print("x shape",data.shape, "y shape",ref_arr.shape,"map shape",gt_map.shape)
        
        xlist, ylist = batchize(data, gt_map, xlist, ylist, size=batch_size)

    xlist = np.array(xlist)
    ylist = np.array(ylist)
    print("xlist:"+str(xlist.shape)+"ylist:"+str(ylist.shape))
    hf = h5py.File("./data/train_vocal.h5", "w")
    hf.create_dataset("x", data=xlist)
    hf.create_dataset("y", data=ylist)
    hf.close()

    xlist = []
    ylist = []
    for songname in val_songlist:
        filepath = data_folder + "/" + songname + ".wav"
        data, CenFreq, time_arr = cfp_process(
            filepath, model_type=model_type, sr=44100, hop=256
        )
        
        data = batchize_val(data, size=batch_size)
        ypath = filepath.replace(".wav", ".csv")
        lpath = "data/lpath.csv"
        ref_arr = select_vocal_track(ypath, lpath,time_arr)
        
        xlist.append(data)
        ylist.append(ref_arr)

    with open(output_folder + "/val_x_vocal.pickle", "wb") as fp:
        pickle.dump(xlist, fp)

    with open(output_folder + "/val_y_vocal.pickle", "wb") as fp:
        pickle.dump(ylist, fp)


def parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "-df",
        "--data_folder",
        help="Path to the dataset folder (default: %(default)s",
        type=str,
        default="/MIR-1K/LyricsWav",
    )
    p.add_argument(
        "-t",
        "--model_type",
        help="Model type: vocal or melody (default: %(default)s",
        type=str,
        default="vocal",
    )
    p.add_argument(
        "-o",
        "--output_folder",
        help="Path to output foler (default: %(default)s",
        type=str,
        default="./data/",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parser()
    main(args.data_folder, args.model_type, args.output_folder)
