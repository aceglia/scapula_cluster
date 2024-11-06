from scapula_cluster.from_cluster_to_anato import ScapulaCluster
import glob

if __name__ == "__main__":
    l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia = 103, 32, 36, 148, -9, 70
    calibration_matrix = "calibration_matrix/calibration_mat_left_RGBD_screen.json"
    new_cluster = ScapulaCluster(
        l_collar_TS, l_pointer_TS, l_pointer_IA, l_collar_IA, angle_wand_ia, l_wand_ia, calibration_matrix
    )
    all_files = ["calib.c3d"]
    # all_files = glob.glob(data_files + "/**.c3d")
    names = ["M1", "M2", "M3"]
    new_cluster.process(c3d_files=all_files, cluster_marker_names=names, save_file=True)
    [aa_ts, aa_ia, ia_ts] = new_cluster.get_landmarks_distance()
