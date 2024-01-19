from scapula_cluster.from_cluster_to_anato import ScapulaCluster
import glob
import json
import os


def find_files(path, participant):
    all_files = glob.glob(path + f"{participant}/Session_1/**.c3d")
    if len(all_files) == 0:
        all_files = glob.glob(path + f"{participant}/**.c3d")
    if len(all_files) == 0:
        all_files = glob.glob(path + f"{participant}/session_1/**.c3d")
    if len(all_files) == 0:
        raise ValueError(f"no c3d files found for participant {participant}")
    files = [file for file in all_files if "abd" in file or "flex" in file or "cluster" in file]
    return files


def process(data_path, measurement_path, calibration_matrix_path, participant):
    data = json.load(open(measurement_path + os.sep + f"measurements_{participant}.json"))["with_vicon"]
    measure = data["measure"]
    calib = calibration_matrix_path + os.sep + data["calibration_matrix_name"]
    new_cluster = ScapulaCluster(
        measure[0], measure[1], measure[2], measure[3], measure[4], measure[5], calib
    )
    all_files = find_files(data_path, participant)
    new_cluster.process(c3d_files=all_files, cluster_marker_names=["M1", "M2", "M3"], save_file=True)


if __name__ == "__main__":
    data_dir = "/mnt/shared/Projet_hand_bike_markerless/vicon/"
    participants = ["P11" ]#"P5", "P6", "P7", "P8", "P9", "P10", ]
    measurement_dir = "/home/amedeo/Documents/programmation/rgbd_mocap/data_collection_mesurement"
    calibration_matrix_path = "/home/amedeo/Documents/programmation/scapula_cluster/calibration_matrix"
    for p, part in enumerate(participants):
        process(data_dir, measurement_dir, calibration_matrix_path, part)

