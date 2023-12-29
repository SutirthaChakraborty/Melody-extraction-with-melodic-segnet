import csv
import mir_eval
import numpy as np
import pickle

import pandas as pd


def getlist_mdb():
    train = [
        "AimeeNorwich_Child",
        "AimeeNorwich_Flying",
        "AlexanderRoss_GoodbyeBolero",
        "AlexanderRoss_VelvetCurtain",
        "AvaLuna_Waterduct",
        "BigTroubles_Phantom",
        "CroqueMadame_Oil",
        "CroqueMadame_Pilot",
        "DreamersOfTheGhetto_HeavyLove",
        "EthanHein_1930sSynthAndUprightBass",
        "EthanHein_GirlOnABridge",
        "FacesOnFilm_WaitingForGa",
        "FamilyBand_Again",
        "Handel_TornamiAVagheggiar",
        "HeladoNegro_MitadDelMundo",
        "HopAlong_SisterCities",
        "JoelHelander_Definition",
        "JoelHelander_ExcessiveResistancetoChange",
        "JoelHelander_IntheAtticBedroom",
        "KarimDouaidy_Hopscotch",
        "KarimDouaidy_Yatora",
        "LizNelson_Coldwar",
        "LizNelson_ImComingHome",
        "LizNelson_Rainfall",
        "Meaxic_TakeAStep",
        "Meaxic_YouListen",
        "Mozart_BesterJungling",
        "MusicDelta_80sRock",
        "MusicDelta_Beatles",
        "MusicDelta_BebopJazz",
        "MusicDelta_Beethoven",
        "MusicDelta_Britpop",
        "MusicDelta_ChineseChaoZhou",
        "MusicDelta_ChineseDrama",
        "MusicDelta_ChineseHenan",
        "MusicDelta_ChineseJiangNan",
        "MusicDelta_ChineseXinJing",
        "MusicDelta_ChineseYaoZu",
        "MusicDelta_CoolJazz",
        "MusicDelta_Country1",
        "MusicDelta_Country2",
        "MusicDelta_Disco",
        "MusicDelta_FreeJazz",
        "MusicDelta_FunkJazz",
        "MusicDelta_GriegTrolltog",
        "MusicDelta_Grunge",
        "MusicDelta_Hendrix",
        "MusicDelta_InTheHalloftheMountainKing",
        "MusicDelta_LatinJazz",
        "MusicDelta_ModalJazz",
        "MusicDelta_Punk",
        "MusicDelta_Reggae",
        "MusicDelta_Rock",
        "MusicDelta_Rockabilly",
        "MusicDelta_Shadows",
        "MusicDelta_SpeedMetal",
        "MusicDelta_Vivaldi",
        "MusicDelta_Zeppelin",
        "PurlingHiss_Lolita",
        "Schumann_Mignon",
        "StevenClark_Bounty",
        "SweetLights_YouLetMeDown",
        "TheDistricts_Vermont",
        "TheScarletBrand_LesFleursDuMal",
        "TheSoSoGlos_Emergency",
        "Wolf_DieBekherte",
    ]
    validation = [
        "AmarLal_Rest",
        "AmarLal_SpringDay1",
        "BrandonWebster_DontHearAThing",
        "BrandonWebster_YesSirICanFly",
        "ClaraBerryAndWooldog_AirTraffic",
        "ClaraBerryAndWooldog_Boys",
        "ClaraBerryAndWooldog_Stella",
        "ClaraBerryAndWooldog_TheBadGuys",
        "ClaraBerryAndWooldog_WaltzForMyVictims",
        "HezekiahJones_BorrowedHeart",
        "InvisibleFamiliars_DisturbingWildlife",
        "MichaelKropf_AllGoodThings",
        "NightPanther_Fire",
        "SecretMountains_HighHorse",
        "Snowmine_Curfews",
    ]
    test = [
        "AClassicEducation_NightOwl",
        "Auctioneer_OurFutureFaces",
        "CelestialShore_DieForUs",
        "ChrisJacoby_BoothShotLincoln",
        "ChrisJacoby_PigsFoot",
        "Creepoid_OldTree",
        "Debussy_LenfantProdigue",
        "MatthewEntwistle_DontYouEver",
        "MatthewEntwistle_FairerHopes",
        "MatthewEntwistle_ImpressionsOfSaturn",
        "MatthewEntwistle_Lontano",
        "MatthewEntwistle_TheArch",
        "MatthewEntwistle_TheFlaxenField",
        "Mozart_DiesBildnis",
        "MusicDelta_FusionJazz",
        "MusicDelta_Gospel",
        "MusicDelta_Pachelbel",
        "MusicDelta_SwingJazz",
        "Phoenix_BrokenPledgeChicagoReel",
        "Phoenix_ColliersDaughter",
        "Phoenix_ElzicsFarewell",
        "Phoenix_LarkOnTheStrandDrummondCastle",
        "Phoenix_ScotchMorris",
        "Phoenix_SeanCaughlinsTheScartaglen",
        "PortStWillow_StayEven",
        "Schubert_Erstarrung",
        "StrandOfOaks_Spacestation",
    ]
    return train, validation, test


def getlist_mdb_vocal():
    train_songlist = [
        "AimeeNorwich_Child",
        "AlexanderRoss_GoodbyeBolero",
        "AlexanderRoss_VelvetCurtain",
        "AvaLuna_Waterduct",
        "BigTroubles_Phantom",
        "DreamersOfTheGhetto_HeavyLove",
        "FacesOnFilm_WaitingForGa",
        "FamilyBand_Again",
        "Handel_TornamiAVagheggiar",
        "HeladoNegro_MitadDelMundo",
        "HopAlong_SisterCities",
        "LizNelson_Coldwar",
        "LizNelson_ImComingHome",
        "LizNelson_Rainfall",
        "Meaxic_TakeAStep",
        "Meaxic_YouListen",
        "MusicDelta_80sRock",
        "MusicDelta_Beatles",
        "MusicDelta_Britpop",
        "MusicDelta_Country1",
        "MusicDelta_Country2",
        "MusicDelta_Disco",
        "MusicDelta_Grunge",
        "MusicDelta_Hendrix",
        "MusicDelta_Punk",
        "MusicDelta_Reggae",
        "MusicDelta_Rock",
        "MusicDelta_Rockabilly",
        "PurlingHiss_Lolita",
        "StevenClark_Bounty",
        "SweetLights_YouLetMeDown",
        "TheDistricts_Vermont",
        "TheScarletBrand_LesFleursDuMal",
        "TheSoSoGlos_Emergency",
        "Wolf_DieBekherte",
    ]
    val_songlist = [
        "BrandonWebster_DontHearAThing",
        "BrandonWebster_YesSirICanFly",
        "ClaraBerryAndWooldog_AirTraffic",
        "ClaraBerryAndWooldog_Boys",
        "ClaraBerryAndWooldog_Stella",
        "ClaraBerryAndWooldog_TheBadGuys",
        "ClaraBerryAndWooldog_WaltzForMyVictims",
        "HezekiahJones_BorrowedHeart",
        "InvisibleFamiliars_DisturbingWildlife",
        "Mozart_DiesBildnis",
        "NightPanther_Fire",
        "SecretMountains_HighHorse",
        "Snowmine_Curfews",
    ]
    test_songlist = [
        "AClassicEducation_NightOwl",
        "Auctioneer_OurFutureFaces",
        "CelestialShore_DieForUs",
        "Creepoid_OldTree",
        "Debussy_LenfantProdigue",
        "MatthewEntwistle_DontYouEver",
        "MatthewEntwistle_Lontano",
        "Mozart_BesterJungling",
        "MusicDelta_Gospel",
        "PortStWillow_StayEven",
        "Schubert_Erstarrung",
        "StrandOfOaks_Spacestation",
    ]
    return train_songlist, val_songlist, test_songlist


import os
import glob
import random
from typing import Optional, Union, Tuple, List


def split_data_set(
    directory: str = "MIR-1K/LyricsWav",
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Splits a dataset of audio files into training, validation, and test sets based on specified ratios.

    Args:
    directory (str, optional): Path to the directory containing audio files. Default is "MIR-1K/LyricsWav".
    train_ratio (float, optional): The proportion of the dataset to include in the training set. Default is 0.6.
    validation_ratio (float, optional): The proportion of the dataset to include in the validation set. Default is 0.2.
    test_ratio (float, optional): The proportion of the dataset to include in the test set. Default is 0.2.
    seed (int, optional): The seed for random number generation to ensure reproducibility. Default is 42.

    Returns:
    tuple: A tuple containing three lists of file names, corresponding to the training, validation, and test sets, respectively.

    Note:
    The sum of train_ratio, validation_ratio, and test_ratio should be equal to 1.0. The function assumes the directory contains .wav files and splits based on file names, not the actual content of the files.
    """

    # Set the seed for reproducibility
    random.seed(seed)

    # Get all .wav files with full paths
    full_paths = glob.glob(os.path.join(directory, "*.wav"))

    # Extract file stems from full paths
    file_stems = [os.path.splitext(os.path.basename(path))[0] for path in full_paths]

    # file_stems = file_stems[:10]
    # Shuffle the list of file stems
    random.shuffle(file_stems)

    # Calculate split indices
    total_files = len(file_stems)
    train_end = int(train_ratio * total_files)
    validation_end = train_end + int(validation_ratio * total_files)

    # Split the file stems
    train_files = file_stems[:train_end]
    validation_files = file_stems[train_end:validation_end]
    test_files = file_stems[validation_end:]

    return train_files, validation_files, test_files


def getlist_ADC2004():
    test_songlist = [
        "daisy1",
        "daisy2",
        "daisy3",
        "daisy4",
        "opera_fem2",
        "opera_fem4",
        "opera_male3",
        "opera_male5",
        "pop1",
        "pop2",
        "pop3",
        "pop4",
    ]
    return test_songlist


def getlist_MIREX05():
    test_songlist = [
        "train01",
        "train02",
        "train03",
        "train04",
        "train05",
        "train06",
        "train07",
        "train08",
        "train09",
    ]
    return test_songlist


def melody_eval(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """
    Evaluates the estimated melody against a reference melody using various metrics.

    Args:
    ref (np.ndarray): A 2D array where the first column represents time and the second column represents the reference melody frequencies.
    est (np.ndarray): A 2D array similar to 'ref', but for the estimated melody.

    Returns:
    np.ndarray: An array containing evaluation metrics - Voicing Recall, Voicing False Alarm, Raw Pitch Accuracy, Raw Chroma Accuracy, and Overall Accuracy, each multiplied by 100 for percentage representation.

    Note:
    This function uses the `mir_eval` library for melody evaluation.
    """

    ref_time = ref[:, 0]
    ref_frequency = ref[:, 1]

    est_time = est[:, 0]
    est_frequency = est[:, 1]

    output_eval = mir_eval.melody.evaluate(
        ref_time, ref_frequency, est_time, est_frequency
    )
    VR = output_eval["Voicing Recall"] * 100.0
    VFA = output_eval["Voicing False Alarm"] * 100.0
    RPA = output_eval["Raw Pitch Accuracy"] * 100.0
    RCA = output_eval["Raw Chroma Accuracy"] * 100.0
    OA = output_eval["Overall Accuracy"] * 100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr


def csv2ref(ypath: str) -> np.ndarray:
    """
    Converts a CSV file containing time and frequency information into a numpy array.

    Args:
    ypath (str): The file path to the CSV file.

    Returns:
    np.ndarray: A 2D numpy array where the first column is time and the second column is frequency.
    """
    ycsv = pd.read_csv(ypath, names=["time", "frequency"])
    gtt = ycsv["time"].values
    gtf = ycsv["frequency"].values
    ref_arr = np.concatenate((gtt[:, None], gtf[:, None]), axis=1)
    return ref_arr


def select_vocal_track(ypath: str, lpath: str, time_arr: np.ndarray) -> np.ndarray:
    ycsv = pd.read_csv(ypath, names=["time", "frequency"])
    gt_time = ycsv["time"].values[1:].astype(float)  # Convert to float and skip header
    gt_freq = ycsv["frequency"].values[1:].astype(float)  # Convert to float and skip header

    # Initialize an array for frequencies that matches the shape of time_arr
    matched_freq = np.zeros_like(time_arr)

    # For each time in time_arr, find the closest time in gt_time and get the corresponding frequency
    for i, t in enumerate(time_arr):
        # Find index of closest time in gt_time
        idx = np.abs(gt_time - t).argmin()
        # Assign corresponding frequency to matched_freq
        matched_freq[i] = gt_freq[idx]

    # Combine time_arr and matched frequencies
    gt = np.column_stack((time_arr, matched_freq))
    return gt

# def select_vocal_track(ypath: str, lpath: str,time_arr) -> np.ndarray:
#     """
#     Selects the vocal track from a given dataset based on time and frequency information.

#     Args:
#     ypath (str): The file path to the CSV file containing time and frequency information.
#     lpath (str): The file path to the labels file.

#     Returns:
#     np.ndarray: A 2D numpy array where the first column is time and the second column is the selected vocal frequency (zero if not vocal).

#     Note:
#     The function currently sets the vocal frequency to zero for all times. The commented-out part suggests intended functionality for selecting vocals based on labels.
#     """
#     ycsv = pd.read_csv(ypath, names=["time", "frequency"])
#     gt0 = ycsv["time"].values
#     gt0 = gt0[:, np.newaxis][1:].astype(float)

#     gt1 = ycsv["frequency"].values
#     gt1 = gt1[:, np.newaxis][1:].astype(float)

#     # z = np.zeros(gt1.shape)

#     # f = open(lpath, "r")
#     # lines = f.readlines()

#     # for line in lines:

#     #     if 'start_time' in line.split(',')[0]:
#     #         continue
#     #     st = float(line.split(',')[0])
#     #     et = float(line.split(',')[1])
#     #     sid = line.split(',')[2]
#     # for i in range(len(gt1)):
#     #     if st < gt0[i,0] < et and 'singer' in sid:
#     #         z[i,0] = gt1[i,0]

#     gt = np.concatenate((gt0, gt1), axis=1)
#     return gt


def save_csv(data: list, savepath: str):
    """
    Saves a list of evaluation arrays to a CSV file.

    Args:
    data (list): A list of arrays, each containing evaluation metrics.
    savepath (str): The file path to save the CSV file.

    Note:
    The CSV file will contain headers for the evaluation metrics.
    """
    with open(savepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["VR", "VFA", "RPA", "RCA", "OA"])
        for est_arr in data:
            writer.writerow(est_arr)


def load_list(savepath: str) -> list:
    """
    Loads a list from a pickle file.

    Args:
    savepath (str): The file path to the pickle file.

    Returns:
    list: The list loaded from the pickle file.
    """
    with open(savepath, "rb") as file:
        xlist = pickle.load(file)
    return xlist
