from get_data_utils import get_all_file
from process_utils import process_markers
from biosiglive import MskFunctions, InverseKinematicsMethods
import bioviz


if __name__ == '__main__':
    measurement_dir = "D:\Documents\Programmation\scapula_cluster\data_collection_mesurement"
    calib_matrix_dir = "D:\Documents\Programmation\scapula_cluster\calibration_matrix"
    part = "P11"
    files, participants = get_all_file([part], r"Q:\Projet_hand_bike_markerless\vicon",
                                       to_include=["abd_90"], to_exclude=["processed", "gear"])

    all_markers, _ = process_markers(files[0], participants[0],
                                  measurements_dir_path=measurement_dir,
                                  calibration_matrix_dir=calib_matrix_dir)
    model = r"Q:\Projet_hand_bike_markerless\vicon\model_reduce_left.bioMod"
    msk = MskFunctions(model, all_markers[:3, ...].shape[-1], system_rate=120)
    q, _, _ = msk.compute_inverse_kinematics(all_markers[:3, ...], method=InverseKinematicsMethods.BiorbdKalman)
    b = bioviz.Viz(model)
    b.load_movement(q)
    b.load_experimental_markers(all_markers[:3, ...])
    b.exec()



