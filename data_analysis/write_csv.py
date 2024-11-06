import os

from get_data_utils import get_all_file
from process_utils import process_markers, compute_error_mark
import pandas as pd
from biosiglive import MskFunctions, InverseKinematicsMethods
import numpy as np

if __name__ == '__main__':
    measurement_dir = "D:\Documents\Programmation\scapula_cluster\data_collection_mesurement"
    calib_matrix_dir = "D:\Documents\Programmation\scapula_cluster\calibration_matrix"
    participants = [f"P{i}" for i in range(9, 17)]
    files, participants = get_all_file(participants, r"Q:\Projet_hand_bike_markerless\vicon",
                                       to_include=["flex", "abd"],
                                       to_exclude=["processed", "dyn", "gear", "xml", "anato_cluster", "anato", "sprint"],
                                       file_ext=".c3d")
    all_data_dic = {}
    all_mean = [[], []]
    all_std = [[], []]
    systems = ["cluster", "skin", "locator"]
    axis = ["x", "y", "z", "mean"]
    markers_names = ["AA", "AI", "TS"]
    params = sum([[f"{n}_{a}" for a in axis] for n in markers_names], [])
    dof_names = ["retraction/protraction", "lateral/medial rotation", "internal/external rotation", "mean"]
    params += sum([[f"{n}_{a}" for a in axis] for n in dof_names], [])
    data_df = pd.DataFrame()
    model = r"Q:\Projet_hand_bike_markerless\vicon\model_reduce_left.bioMod"

    for file, part in zip(files, participants):
        file_name = file.split(os.sep)[-1].replace(".c3d", "")
        print(f"Processing {file_name} for {part}")
        all_markers, names = process_markers(file, part,
                                      measurements_dir_path=measurement_dir,
                                      calibration_matrix_dir=calib_matrix_dir)
        if np.argwhere(all_markers == 0).shape[0] != 0:
            print(f"at least one marker is 0 in {file_name} for {part}")
            continue
        msk = MskFunctions(model, all_markers[:3, ...].shape[-1], system_rate=120)
        q, _, _ = msk.compute_inverse_kinematics(all_markers[:3, ...],
                                                 method=InverseKinematicsMethods.BiorbdKalman)
        all_markers = all_markers * 1000
        q = q * 180 / np.pi
        mean_markers = np.mean(all_markers, axis=0)
        q_shaped = np.zeros((3, 6, q.shape[1]))
        count_axis = 0
        count_markers = 0
        for i in range(q.shape[0]):
            q_shaped[count_axis, count_markers, :] = q[i, :]
            count_axis += 1
            if count_axis == 3:
                count_axis = 0
                count_markers += 1
        al_q_rot = q_shaped[:, 3:, :]
        mean_q = np.mean(al_q_rot, axis=0)
        mark_by_sys = np.zeros((3, 3, 3, all_markers.shape[2]))  # axis * markers, system, frame
        count = 0
        for i in range(3):
            mark_by_sys[:, :, i, :] = all_markers[:, count:count+3, :]
            count += 3

        mean_mark = np.mean(mark_by_sys, axis=0)
        mark_by_sys = mark_by_sys.mean(axis=-1)
        mean_mark = mean_mark.mean(axis=-1)
        al_q_rot = al_q_rot.mean(axis=-1)
        mean_q = mean_q.mean(axis=-1)
        sub_df = pd.DataFrame()
        for s, system in enumerate(systems):
            subsub_df = pd.DataFrame()
            #subsub_df["frame"] = np.linspace(0, all_markers.shape[2]-1, all_markers.shape[2]).astype(int)
            subsub_df["participant"] = [part] #* all_markers.shape[2]
            subsub_df["essai"] = [file_name[-1]]
            subsub_df["file"] = [file_name] #* all_markers.shape[2]
            subsub_df["system"] = [system] #* all_markers.shape[2]
            for a, ax in enumerate(axis):
                for m, mark in enumerate(markers_names):
                    subsub_df[f"{mark}_{ax}"] = mark_by_sys[m, a, s] if ax != "mean" else mean_mark[m, s]
            for q_idx, q_name in enumerate(dof_names):
                subsub_df[f"dof_{q_name}"] = al_q_rot[q_idx, s] if q_name != "mean" else mean_q[s]

            sub_df = pd.concat((sub_df, subsub_df))
        data_df = pd.concat((data_df, sub_df))
    data_df.to_csv("D:\Documents\Programmation\scapula_cluster\data_analysis\data_df.csv")
