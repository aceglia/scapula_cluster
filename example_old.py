from scapula_cluster.from_cluster_to_anato import ScapulaCluster
import glob

if __name__ == "__main__":
    l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia = 85, 37, 48, 140, -9, 46
    calibration_matrix = "/home/amedeo/Documents/programmation/scapula_cluster/calibration_matrix/calibration_mat_left_reflective_markers.json"
    new_cluster = ScapulaCluster(
        l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia, calibration_matrix
    )
    data_files = "/mnt/shared/Projet_hand_bike_markerless/vicon/P11/session_1"
    all_files = glob.glob(data_files + "/**.c3d")
    names = ["M1", "M2", "M3"]
    new_cluster.process(c3d_files=all_files, cluster_marker_names=names, save_file=True)
    [aa_ts, aa_ia, ia_ts] = new_cluster.get_landmarks_distance()
