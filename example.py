from scapula_cluster.from_cluster_to_anato import ScapulaCluster
import glob
import json
import os
from pyomeca import Markers
import matplotlib.pyplot as plt

def find_files(path, participant, delete_old_files=False):
    all_files = glob.glob(path + f"\{participant}/Session_1/**.c3d")
    if len(all_files) == 0:
        all_files = glob.glob(path + f"\{participant}/**.c3d")
    if len(all_files) == 0:
        all_files = glob.glob(path + f"\{participant}/session_1/**.c3d")
    if len(all_files) == 0:
        raise ValueError(f"no c3d files found for participant {participant}")
    if delete_old_files:
        for file in all_files:
            if "processed" in file:
                os.remove(file)
    # files = [file for file in all_file if ("abd" in file or "flex" in file or "cluster" in file)
    #          and file[:-4] + "_processed.c3d" not in all_files and "processed" not in file]
    files = [file for file in all_files if ("anato" in file)
             and file[:-4] + "_processed.c3d" not in all_files and "processed" not in file and "cluster" not in file]
    return files


def process(data_path, measurement_path, calibration_matrix_path, participant, delete_old_files=False, source="vicon"):
    data = json.load(open(measurement_path + os.sep + f"measurements_{participant}.json"))[f"with_{source}"]
    measure = data["measure"]
    calib = calibration_matrix_path + os.sep + data["calibration_matrix_name"]
    new_cluster = ScapulaCluster(
        measure[0], measure[1], measure[2], measure[3], measure[4], measure[5], calib
    )
    all_files = find_files(data_path, participant, delete_old_files=delete_old_files)
    new_cluster.process(c3d_files=all_files, cluster_marker_names=["M1", "M2", "M3"], save_file=True)


def plot_results(path, participant):
    all_files = glob.glob(path + f"\{participant}/Session_1/**.c3d")
    if len(all_files) == 0:
        all_files = glob.glob(path + f"\{participant}/**.c3d")
    if len(all_files) == 0:
        all_files = glob.glob(path + f"\{participant}/session_1/**.c3d")
    if len(all_files) == 0:
        raise ValueError(f"no c3d files found for participant {participant}")
    files = [file for file in all_files if ("anato" in file)
             and file[:-4] + "_processed.c3d" not in all_files and "processed" in file and "cluster" not in file]
    for file in files:
        color = ["r", "g", "b", "y", "c", "m"]
        # color = ["forestgreen", "turquoise", "lawngreen", "red", "tomato", "sienna", "ligthblue", "navy", "stateblue"]
        mark_type = ["o", "P", "X", "s", "v", "2"]
        markers = Markers.from_c3d(file)
        fig = plt.figure(file)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])
        for idx, mark in enumerate(markers.channel.values):
            if mark in ["M1"] : #, "M2", "M3"]:
                color = "r"
                marker = "o"
            elif mark in ["M2"]:  # , "M2", "M3"]:
                color = "r"
                marker = "x"
            elif mark in ["M3"]:  # , "M2", "M3"]:
                color = "r"
                marker = "P"
            elif mark in ['scap_aa_from_cluster', 'scap_ia_from_cluster', 'scap_ts_from_cluster']:
                color = "g"
                marker = "o"
            else:
                color = "b"
                marker = "o"

            ax.scatter(markers[0, idx, :1], markers[1, idx, :1], markers[2, idx, :1], c=color, marker=marker)
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        plt.show()

if __name__ == "__main__":
    data_dir = r"Q:\Projet_hand_bike_markerless\vicon"
    participants = ["P9"] #, "P10", "P11", "P12", "P13", "P14"]
    measurement_dir = "data_collection_mesurement"
    calibration_matrix_path = "calibration_matrix"
    for p, part in enumerate(participants):
        print("participant", part)
        process(data_dir, measurement_dir, calibration_matrix_path, part, delete_old_files=False, source="depth")
        plot_results(data_dir, part)
