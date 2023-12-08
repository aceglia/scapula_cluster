import numpy as np
import matplotlib.pyplot as plt
from pyomeca import Markers
import glob


def create_axis_coordinates(M1, M2, M3):
    first_axis_vector = M3 - M1
    second_axis_vector = M2 - M1
    third_axis_vector = -np.cross(first_axis_vector, second_axis_vector, axis=0)
    second_axis_vector = np.cross(third_axis_vector, first_axis_vector, axis=0)
    n_frames = first_axis_vector.shape[1]
    rt = np.zeros((4, 4, n_frames))
    rt[:3, 0, :] = first_axis_vector / np.linalg.norm(first_axis_vector, axis=0)
    rt[:3, 1, :] = second_axis_vector / np.linalg.norm(second_axis_vector, axis=0)
    rt[:3, 2, :] = third_axis_vector / np.linalg.norm(third_axis_vector, axis=0)
    rt[:3, 3, :] = M1
    rt[3, 3, :] = 1
    return rt


def ajust_sl(pt1, pt2, pt3, marker_correction):
    n_frames = pt1.shape[1]
    mat_hom_sl_g = create_axis_coordinates(pt1[:3, :], pt2[:3, :], pt3[:3, :])
    pt_sl = np.ones((4, 3, n_frames))
    for i in range(n_frames):
        mat = mat_hom_sl_g[:, :, i]
        matinv = np.linalg.inv(mat_hom_sl_g[:, :, i])
        pt_sl[:, 0, i] = matinv.dot(pt1[:, i])
        pt_sl[:, 1, i] = matinv.dot(pt2[:, i])
        pt_sl[:, 2, i] = matinv.dot(pt3[:, i])
        pt_sl[2, 0, i] += marker_correction
        pt_sl[2, 1, i] += marker_correction
        pt_sl[2, 2, i] += marker_correction
        pt_sl[:, 0, i] = mat.dot(pt_sl[:, 0, i])
        pt_sl[:, 1, i] = mat.dot(pt_sl[:, 1, i])
        pt_sl[:, 2, i] = mat.dot(pt_sl[:, 2, i])
    mat_sl = np.zeros((4, 3, n_frames))
    mat_sl[:, 0, :], mat_sl[:, 1, :], mat_sl[:, 2, :] = pt_sl[:, 2, :], pt_sl[:, 0, :], pt_sl[:, 1, :]
    return mat_sl


def eval_rmse(pt, pt_process):
    error = []
    for i in range(pt.shape[1]):
        if np.mean(pt[:3, i, :]) == 0 or np.mean(pt_process[:3, i, :]) == 0:
            error.append(0)
        else:
            pt_wt_nan_idx = np.argwhere(np.isfinite(pt[:3, i, :]))[:, 1]
            pt_process_wt_nan_idx = np.argwhere(np.isfinite(pt_process[:3, i, :]))[:, 1]
            all_idx = np.intersect1d(pt_wt_nan_idx, pt_process_wt_nan_idx)
            error.append(np.sqrt(np.mean((pt[:3, i, all_idx] - pt_process[:3, i, all_idx]) ** 2, axis=0)).mean())
    return np.array(error)


def plot(all_markers_dir, noms_de_fichier):
    color = ["r", "g", "b", "y", "c", "m"]
    # color = ["forestgreen", "turquoise", "lawngreen", "red", "tomato", "sienna", "ligthblue", "navy", "stateblue"]
    mark_type = ["o", "P", "X", "s", "v", "2"]
    ax = None
    for d, dic in enumerate(all_markers_dir):
        fig = plt.figure(noms_de_fichier[d])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])
        for k, key in enumerate(dic):
            idx = np.argwhere(np.mean(dic[key][0, :, :], axis=1) != 0)[:, 0]
            # for i in idx:
            ax.scatter(dic[key][0, idx, :], dic[key][1, idx, :], dic[key][2, idx, :], c=color[k], label=key)
        plt.legend()
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()


def load_markers(all_files, marker_names, frame_of_interest=None, scapula_locator_correction=None):
    list_all_markers = []
    if frame_of_interest:
        if len(frame_of_interest) != len(all_files):
            raise ValueError("frame_of_interest and all_files must have the same length if provided.")
        for f, frame in enumerate(frame_of_interest):
            if isinstance(frame, int):
                frame_of_interest[f] = [frame, None]
            elif isinstance(frame, list):
                if len(frame) != 2:
                    frame_of_interest[f] = [frame[0], None]
    else:
        frame_of_interest = [[0, None] for _ in range(len(all_files))]
    for i in range(len(all_files)):
        s, e = frame_of_interest[i][0], frame_of_interest[i][1]
        list_all_markers_data = Markers.from_c3d(all_files[i])
        if e:
            all_markers_data = list_all_markers_data.values[:, :, s:e]
        else:
            all_markers_data = list_all_markers_data.values[:, :, s:]
        mat_sl = np.zeros((4, 3, list_all_markers_data.values.shape[2]))
        mat_scap = np.zeros((4, 3, list_all_markers_data.values.shape[2]))
        mat_cluster = np.zeros((4, 3, list_all_markers_data.values.shape[2]))

        for n, name in enumerate(list_all_markers_data.channel.values):
            if name in marker_names["sl"]:
                mat_sl[:, marker_names["sl"].index(name), :] = all_markers_data[:, n, :]
            elif name in marker_names["scap"]:
                mat_scap[:, marker_names["scap"].index(name), :] = all_markers_data[:, n, :]
            elif name in marker_names["cluster"]:
                mat_cluster[:, marker_names["cluster"].index(name), :] = all_markers_data[:, n, :]
        mat_sl = (
            ajust_sl(mat_sl[:, 1, :], mat_sl[:, 2, :], mat_sl[:, 0, :], scapula_locator_correction)
            if np.sum(mat_sl) != 0
            else mat_sl
        )
        list_all_markers.append(dict(mat_sl=mat_sl, mat_scap=mat_scap, mat_cluster=mat_cluster))
    return list_all_markers


def compute_error(
    all_files, marker_names, frame_of_interest=None, scapula_locator_correction=None, show_plot=True, print_error=True
):
    all_markers_dict = load_markers(all_files, marker_names, frame_of_interest, scapula_locator_correction)
    rmse_sl_scap = np.zeros((len(all_files), 3))
    rmse_scap_cluster = np.zeros((len(all_files), 3))
    rmse_sl_cluster = np.zeros((len(all_files), 3))
    for i in range(len(all_markers_dict)):
        rmse_sl_scap[i, :] = eval_rmse(all_markers_dict[i]["mat_sl"], all_markers_dict[i]["mat_scap"])
        rmse_scap_cluster[i, :] = eval_rmse(all_markers_dict[i]["mat_scap"], all_markers_dict[i]["mat_cluster"])
        rmse_sl_cluster[i, :] = eval_rmse(all_markers_dict[i]["mat_sl"], all_markers_dict[i]["mat_cluster"])
        if print_error:
            print("Error file {} :".format(all_files[i]))
            print("RMSE for AA, AI, TS between SL-SCAP : {}".format(rmse_sl_scap[i, :]))
            print("RMSE for AA, AI, TS between SCAP-CLUSTER : {}".format(rmse_scap_cluster[i, :]))
            print("RMSE for AA, AI, TS between SL-CLUSTER : {}".format(rmse_sl_cluster[i, :]))
    if print_error:
        print("Mean RMSE for AA, AI, TS between SL-SCAP : {}".format(np.mean(rmse_sl_scap, axis=0)))
        print("Mean RMSE for AA, AI, TS between SCAP-CLUSTER : {}".format(np.mean(rmse_scap_cluster, axis=0)))
        print("Mean RMSE for AA, AI, TS between SL-CLUSTER : {}".format(np.mean(rmse_sl_cluster, axis=0)))
    if show_plot:
        plot(all_markers_dict, all_files)
    return rmse_sl_cluster, rmse_sl_scap, rmse_scap_cluster


if __name__ == "__main__":
    data_files = "data_scap/P8/"
    all_files = glob.glob(data_files + "/**_processed.c3d")
    if len(all_files) == 0:
        raise RuntimeError("No files found. Please make sure to process your C3D files first using the example.")
    marker_names = {
        "sl": ["slaa", "slai", "slts"],
        "scap": ["scapaa", "scapia", "scapts"],
        "cluster": ["scap_aa_from_cluster", "scap_ia_from_cluster", "scap_ts_from_cluster"],
    }
    error = compute_error(all_files, marker_names, scapula_locator_correction=-155, show_plot=True, print_error=True)
