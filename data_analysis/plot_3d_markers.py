from process_utils import init_scapula_cluster, correct_sl
from get_data_utils import get_all_file
import matplotlib.pyplot as plt
from pyomeca import Markers
import pandas as pd
from process_utils import process_markers


if __name__ == '__main__':
    part = "P9"
    files, participants = get_all_file([part], r"D:/Documents/Programmation/scapula_cluster/data",
                                       to_include=["abd_90"], to_exclude=["processed", "dyn"], file_ext=".c3d")
    scapula_cluster = init_scapula_cluster(part, "D:/Documents/Programmation/scapula_cluster/data_collection_mesurement",
                                           "D:/Documents/Programmation/scapula_cluster/calibration_matrix", "with_vicon")
    measurement_dir = "D:/Documents/Programmation/scapula_cluster/data_collection_mesurement"
    calib_matrix_dir = "D:/Documents/Programmation/scapula_cluster/calibration_matrix"
    all_markers, names = process_markers(files[0], part,
                                         measurements_dir_path=measurement_dir,
                                         calibration_matrix_dir=calib_matrix_dir)
    markers_data = Markers.from_c3d(files[0])
    markers_pos = markers_data.values
    markers_names = markers_data.channel.values.tolist()
    markers_cluster = markers_pos[:, [markers_names.index(name) for name in ["M1", "M2", "M3"]]]
    anato_pos = scapula_cluster.process(
        marker_cluster_positions=markers_cluster, cluster_marker_names=["M1", "M2", "M3"], save_file=False
    )
    scap_markers = markers_pos[:, [markers_names.index(name) for name in ["scapaa", "scapts", "scapia"]]]
    sl_markers = markers_pos[:, [markers_names.index(name) for name in ["slaa", "slts", "slai"]]]
    sl_corrected = correct_sl(sl_markers[:, 0], sl_markers[:, 1], sl_markers[:, 2], marker_correction=155)
    df = pd.read_csv("data_df.csv")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    df_file = df
    df_part = df_file.loc[df_file["participant"] == "P10"]
    colors = ["r", "g", "b", "y"]
    count = 0
    model = r"Q:\Projet_hand_bike_markerless\vicon\model_reduce_left.bioMod"

    for grp_name, grp_idx in df_part.groupby('system').groups.items():
        ax.scatter(df_part.loc[grp_idx, "AA_x"], df_part.loc[grp_idx, "AA_y"], df_part.loc[grp_idx, "AA_z"], label=grp_name, color=colors[count])
        ax.scatter(df_part.loc[grp_idx, "TS_x"], df_part.loc[grp_idx, "TS_y"], df_part.loc[grp_idx, "TS_z"], label=grp_name, color=colors[count])
        ax.scatter(df_part.loc[grp_idx, "AI_x"], df_part.loc[grp_idx, "AI_y"], df_part.loc[grp_idx, "AI_z"], label=grp_name, color=colors[count])
        count += 1
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.legend()
    # plt.show()
    # fig = plt.figure()
    all_markers = all_markers * 1000
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_box_aspect([1, 1, 1])
    # ax.scatter(anato_pos[0, :], anato_pos[1, :], anato_pos[2, :], c="r", marker="o")
    # ax.scatter(scap_markers[0, :], scap_markers[1, :], scap_markers[2, :], c="b", marker="o")
    # # ax.scatter(sl_markers[0, :], sl_markers[1, :], sl_markers[2, :], c="g", marker="o")
    # ax.scatter(sl_corrected[0, :], sl_corrected[1, :], sl_corrected[2, :], c="y", marker="o")
    ax.scatter(all_markers[0, :3, :], all_markers[1, :3, :], all_markers[2, :3, :], c="r", marker="o")
    ax.scatter(all_markers[0, 3:6, :], all_markers[1, 3:6, :], all_markers[2, 3:6, :], c="b", marker="x")
    ax.scatter(all_markers[0, 6:9, :], all_markers[1, 6:9, :], all_markers[2, 6:9, :], c="g", marker="^")
    plt.show()
