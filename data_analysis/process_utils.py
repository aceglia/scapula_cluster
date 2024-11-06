from scapula_cluster.from_cluster_to_anato import ScapulaCluster
import json
import os
import numpy as np
from pyomeca import Markers


def process_markers(file_path, part, to_get=("cluster", "skin", "sl"),
                    measurements_dir_path="../data_collection_mesurement",
                    calibration_matrix_dir="calibration_matrix", config="with_vicon"):
    scapula_cluster = init_scapula_cluster(part, measurements_dir_path, calibration_matrix_dir, config)
    markers_data = Markers.from_c3d(file_path)
    markers_pos = markers_data.values
    markers_names = markers_data.channel.values.tolist()
    all_names = []
    final_data = None
    for key in to_get:
        if "cluster" in key:
            if "M1" not in markers_names or "M2" not in markers_names or "M3" not in markers_names:
                markers = np.zeros((4, 3, markers_pos.shape[2]))
            else:
                markers_cluster = markers_pos[:, [markers_names.index(name) for name in ["M1", "M2", "M3"]]]
                markers = scapula_cluster.process(
                    marker_cluster_positions=markers_cluster, cluster_marker_names=["M1", "M2", "M3"], save_file=False
                )
                all_names.extend(["cluster_aa", "cluster_ai", "cluster_ts"])
        elif "skin" in key:
            skin_names = ["scapaa", "scapia", "scapts"]
            names_in_list = [name for name in skin_names if name in markers_names]
            idx_in_list = [skin_names.index(name) for name in names_in_list]
            idx_global = [markers_names.index(name) for name in names_in_list]
            markers = np.zeros((4, 3, markers_pos.shape[2]))
            markers_tmp = markers_pos[:, idx_global, :]
            markers[:, idx_in_list, :] = markers_tmp
            all_names.extend(skin_names)
        elif "sl" in key:
            sl_names = ["slaa", "slai", "slts"]
            names_in_list = [name for name in sl_names if name in markers_names]
            idx_in_list = [sl_names.index(name) for name in names_in_list]
            idx_global = [markers_names.index(name) for name in names_in_list]
            markers = np.zeros((4, 3, markers_pos.shape[2]))
            markers_tmp = markers_pos[:, idx_global, :]
            markers[:, idx_in_list, :] = markers_tmp
            markers = correct_sl(markers[:, 0, :], markers[:, 1, :], markers[:, 2, :])

            all_names.extend(sl_names)
        else:
            raise ValueError(f"Unknown key {key}")

        if markers.shape[1] == 0:
            markers = np.zeros((4, 3, markers_pos.shape[2]))

        final_data = markers if final_data is None else np.concatenate((final_data, markers), axis=1)
    all_marks = final_data[:3, ...] * 0.001
    return all_marks, all_names


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


def correct_sl(pt1, pt2, pt3, marker_correction=-155):
    n_frames = pt1.shape[1]
    mat_hom_sl_g = create_axis_coordinates(pt1[:3, :], pt2[:3, :], pt3[:3, :])
    pt_sl = np.ones((4, 3, n_frames))
    for i in range(n_frames):
        # mat = create_axis_coordinates(pt1[:3, i:i+1], pt2[:3,  i:i+1], pt3[:3,  i:i+1])[:, :, 0]
        mat = mat_hom_sl_g[:, :, i]
        matinv = np.linalg.inv(mat)
        pt_sl[:, 0, i] = matinv.dot(pt1[:, i])
        pt_sl[:, 1, i] = matinv.dot(pt2[:, i])
        pt_sl[:, 2, i] = matinv.dot(pt3[:, i])
        pt_sl[2, 0, i] += marker_correction
        pt_sl[2, 1, i] += marker_correction
        pt_sl[2, 2, i] += marker_correction
        pt_sl[:, 0, i] = mat.dot(pt_sl[:, 0, i])
        pt_sl[:, 1, i] = mat.dot(pt_sl[:, 1, i])
        pt_sl[:, 2, i] = mat.dot(pt_sl[:, 2, i])
    return pt_sl


def compute_error_mark(ref_mark, mark):
    err_markers_list = []
    std_list = []
    for i in range(ref_mark.shape[1]):
        nan_index = np.argwhere(np.isnan(ref_mark[:, i, :]))
        new_markers_depth_tmp = np.delete(mark[:, i, :], nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(ref_mark[:, i, :], nan_index, axis=1)
        nan_index = np.argwhere(np.isnan(new_markers_depth_tmp))
        new_markers_depth_tmp = np.delete(new_markers_depth_tmp, nan_index, axis=1)
        new_markers_vicon_int_tmp = np.delete(new_markers_vicon_int_tmp, nan_index, axis=1)
        if new_markers_vicon_int_tmp.shape[1] != 0 and new_markers_depth_tmp.shape[1] != 0:
            err_markers_list.append(np.median(
                np.sqrt(np.mean(((new_markers_depth_tmp * 1000 - new_markers_vicon_int_tmp * 1000) ** 2), axis=0))
                    ))
            std_list.append(np.std(new_markers_depth_tmp * 1000 - new_markers_vicon_int_tmp * 1000, axis=0))
    return err_markers_list, std_list


def init_scapula_cluster(
        participant, measurements_dir_path, calibration_matrix_dir, config="with_vicon"
):
    measurement_data = json.load(open(measurements_dir_path + os.sep + f"measurements_{participant}.json"))
    measurements = measurement_data[config]["measure"]
    calibration_matrix = calibration_matrix_dir + os.sep + measurement_data[config]["calibration_matrix_name"]
    scapula_cluster = ScapulaCluster(
        measurements[0],
        measurements[1],
        measurements[2],
        measurements[3],
        measurements[4],
        measurements[5],
        calibration_matrix,
    )
    return scapula_cluster
