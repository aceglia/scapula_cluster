import biorbd
import bioviz
import numpy as np
from biosiglive import load, MskFunctions, InverseKinematicsMethods, save

data_mat = load("data_scap/P8/markers_file_recons.bio", merge=False)
all_bioviz = []

for i, data in enumerate(data_mat):
    if "abduction_90_2" in data["file: "]:
        print("curent_file is ", data["file: "])
        mat_sl = data["data"]["mat_sl"]
        mat_cluster = data["data"]["mat_cluster"]
        mat_scap = data["data"]["mat_scap"]
        idx_nan = np.argwhere(np.isnan(mat_sl[:, 0, :]))
        mat_sl[idx_nan[:, 0], 0, idx_nan[:, 1]] = mat_cluster[idx_nan[:, 0], 0, idx_nan[:, 1]]
        # mat_scap[idx_nan[:, 0], 0, idx_nan[:, 1]] = mat_cluster[idx_nan[:, 0], 0, idx_nan[:, 1]]

        mat_sl[:, 0, :] = mat_cluster[:, 0, :]
        model = biorbd.Model("models/wu_reduc_left.bioMod")
        all_mark = np.concatenate((mat_sl, mat_cluster, mat_scap), axis=1)
        ik_function = MskFunctions(model=model, data_buffer_size=mat_sl[:, :, :].shape[2])
        q, _ = ik_function.compute_inverse_kinematics(
            markers=all_mark[:3, :, :] * 0.001, method=InverseKinematicsMethods.BiorbdLeastSquare
        )
        print("q sl:", np.mean(q[:3, :], axis=1) * 180 / np.pi)
        print("q cluster:", np.mean(q[6:9, :], axis=1) * 180 / np.pi)
        print("q scap:", np.mean(q[12:15, :], axis=1) * 180 / np.pi)
        # q_cluster, _ =  ik_function.compute_inverse_kinematics(markers=mat_cluster[:3, :, :]*0.001,
        #                                        method=InverseKinematicsMethods.BiorbdKalman)
        b = bioviz.Viz(loaded_model=model, show_markers=True)
        b.load_movement(q)
        b.load_experimental_markers(all_mark[:3, :, :] * 0.001)
        b.exec()

should_continue = True
while should_continue:
    for i, b in enumerate(all_bioviz):
        if b.vtk_window.is_active:
            b.update()
        else:
            should_continue = False
# b.exec()
# bioviz.Kinogram()
