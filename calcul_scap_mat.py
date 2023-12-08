import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import pyomeca
from pyomeca import Markers
import ezc3d
import csv
import glob
import json
import os


# valeur x premier marqueur frame 0 =  markers[0,0,0]
# 1:coord x,y ou z, 2: indic du marker , 3: nombre frame


def extract_mat_hom(l, alpha_deux, chemin_fichier, nom_matrice):
    if "droite" in chemin_fichier:
        droite = True
    else:
        droite = False
    try:
        with open(chemin_fichier, 'r') as fichier:
            data = json.load(fichier)
            if nom_matrice in data:
                matrice = data[nom_matrice]
            else:
                raise RuntimeError(f"La matrice '{nom_matrice}' n'a pas été trouvée dans le fichier.")
    except Exception as e:
        raise RuntimeError(f"Une erreur s'est produite : {str(e)}")

    matrice = np.array(matrice).astype(dtype=np.float64)
    if nom_matrice == "matrice_homogene_3T2":
        if droite:
            matrice[0, 0] = np.cos(alpha_deux)
            matrice[0, 2] = np.sin(alpha_deux)
            matrice[2, 0] = - np.sin(alpha_deux)
            matrice[2, 2] = np.cos(alpha_deux)
        else:
            matrice[0, 0] = np.cos(alpha_deux)
            matrice[0, 2] = -np.sin(alpha_deux)
            matrice[2, 0] = np.sin(alpha_deux)
            matrice[2, 2] = np.cos(alpha_deux)
    if nom_matrice == "matrice_homogene_2T1":
        matrice[0, 3] = -l
    return matrice


def create_c3d_file(marker_pos, marker_names, save_path: str, fps):
    # mettre booléen pour inserer m1m2m3 ou pas selon le choix
    """
    Write data to a c3d file
    """
    c3d = ezc3d.c3d()
    # Fill it with random data
    c3d["parameters"]["POINT"]["RATE"]["value"] = [fps]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_names
    c3d["data"]["points"] = marker_pos
    # Write the data
    c3d.write(save_path)


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


def transformation_homogene(AI, TS, alpha_deux, l, file_name):
    _1Ta = extract_mat_hom(l, alpha_deux, file_name, "matrice_homogene_1Ta")
    _2T1 = extract_mat_hom(l, alpha_deux, file_name, "matrice_homogene_2T1")
    _3T2 = extract_mat_hom(l, alpha_deux, file_name, "matrice_homogene_3T2")
    # definition des matrices homogènes
    return [_1Ta.dot(TS), _1Ta.dot(_2T1.dot(_3T2.dot(AI)))]

def transformation_homogene_inv(AI, TS, alpha_deux, l, file_name):
    _1Ta = np.linalg.inv(extract_mat_hom(l, alpha_deux, file_name, "matrice_homogene_1Ta"))
    _2T1 = np.linalg.inv(extract_mat_hom(l, alpha_deux, file_name, "matrice_homogene_2T1"))
    _3T2 = np.linalg.inv(extract_mat_hom(l, alpha_deux, file_name, "matrice_homogene_3T2"))
    # definition des matrices homogènes
    return [_1Ta.dot(TS), _1Ta.dot(_2T1.dot(_3T2.dot(AI)))]


def transfo_glob(AI_TS, alpha_deux, l, marker, Rot_am, mat_file_path, list, rot_m_a_d):
    AI, TS = AI_TS[0], AI_TS[1]
    pos_AI_Ra = transformation_homogene(AI, TS, alpha_deux, l, mat_file_path)[1]
    pos_TS_Ra = transformation_homogene(AI, TS, alpha_deux, l, mat_file_path)[0]
    pos_AI_Rm, pos_TS_Rm, pos_AA_Rm = Rot_am.dot(pos_AI_Ra), Rot_am.dot(pos_TS_Ra), Rot_am.dot(np.array(
        [0, 0, 0, 1]))
    # effectue les transformation de AA, AI et TS pour les avoir dans la base Rm
    Rot_m_g_frame = create_axis_coordinates(marker[:3, 2, :], marker[:3, 1, :], marker[:3, 0, :])

    Rot_g_m_frame = np.zeros_like(Rot_m_g_frame)
    markers_glob_rm = np.zeros((4, 3, marker.shape[2]))

    marker_sld = np.array(([54.48, 0, 0, 1], [30.24, -47.18, 0, 1], [0, 0, 0, 1])).T[:, :, np.newaxis]
    new_marker_sld = np.array(([52.77, 0, 0, 1], [29.327, -43.55, 0, 1], [0, 0, 0, 1])).T[:, :]
    new_marker_sld_ra = np.dot(rot_m_a_d, new_marker_sld)
    marker_sld_ra = np.dot(rot_m_a_d, marker_sld[:, :, 0])
    markers_rm_sd = np.array(([-142.83, 3.03, -145.86, 1],
                              [-65.02, 58.94, -58.75, 1],
                            [10.10, -54.49, -27.87, 1])).T[:, :, np.newaxis]
    markers_rm = np.array(([[-141.67800377,    3.14689024, -144.50582377,    1.],
                            [-64.75954871, 58.45721983, -56.71, 1.],
                            [10.10824837, -54.48936515, -27.87, 1.]])).T[:, :, np.newaxis]
    mark_scap = list[0]["mat_sl"]
    rot_scap_cluster = create_axis_coordinates(mark_scap[:3, 0, 137:], mark_scap[:3, 1, 137:], mark_scap[:3, 2, 137:])
    mark_repere_scap = np.zeros_like(mark_scap)
    mark_repere_scap_theorique = np.zeros_like(mark_scap)
    # crée la matrice homogène de Rm vers Rg pour chaque frame, prend donc les coordonnées de M1, M2 et M3
    n_frames = Rot_m_g_frame.shape[2]
    coord_gen = np.zeros((4, 3, n_frames))
    marker_m = np.zeros_like(marker)
    marker_scap_g_sd = np.zeros_like(marker)
    test_marker_scap = np.zeros_like(marker)
    marker_g = np.zeros_like(marker)
    for i in range(n_frames):
        mat = create_axis_coordinates(marker[:3, 2, i:i+1], marker[:3, 1, i:i+1], marker[:3, 0, i:i+1])[:, :, 0]
        # Rot_g_m_frame[:, :, i] = np.linalg.inv(Rot_m_g_frame[:, :, i])
        # marker_m[:, :, i] = np.dot(np.linalg.inv(mat), marker[:, :, i])
        # marker_scap_g_sd[:, :, i] = np.dot(np.linalg.inv(mat), mark_scap[:, :, i])
        #
        # marker_g[:, :, i] = np.dot(mat, marker_sld[:, :, 0])
        # mark_repere_scap[:, :, i] = np.dot(rot_scap_cluster[:, :, 0], marker[:, :, 0])
        #
        # # marker_g[:, :, i] = np.dot(mat, marker_sld[:, :, 0])
        # markers_glob_rm[:, :, i] = np.dot(np.linalg.inv(Rot_m_g_frame[:, :, i]), marker[:, :, i])
        # # marker_scap_g_sd[:, :, i] = np.dot(mat, markers_rm_sd[:, :, 0])
        # test_marker_scap[:, :, i] = np.dot(mat, markers_rm[:, :, 0])
        # # marker_m_ra[:, :, i] = np.dot(np.linalg.inv(Rot_m_a_frame[:, :, i]), marker_sld[:, :, 0])
        # # mat = Rot_m_g_frame[:, :, i]
        coord_gen[:, 1, i] = np.dot(mat, pos_AI_Rm)
        coord_gen[:, 2, i] = np.dot(mat, pos_TS_Rm)
        coord_gen[:, 0, i] = np.dot(mat, pos_AA_Rm)
    # dist_coord_gen_to_marker = marker - coord_gen.mean(axis=1)[:,  np.newaxis, :]
    # new_mark = mark_scap[:, :1, :].mean(axis=1)[:,  np.newaxis, :] + dist_coord_gen_to_marker
    # m_mean = mark_scap.mean(axis=1)[:,  np.newaxis, :]
    # m_mean_bis = coord_gen.mean(axis=1)[:,  np.newaxis, :]
    #
    # rot_scap_cluster_theorique = create_axis_coordinates(coord_gen[:3, 0, :],
    #                                                      coord_gen[:3, 1, :],
    #                                                      coord_gen[:3, 2, :])
    #
    # for i in range(n_frames):
    #     mark_repere_scap_theorique[:, :, i] = np.dot(rot_scap_cluster_theorique[:, :, 0],
    #                                                  mark_repere_scap[:, :, i])

    # x_y_z = Rot_m_g_frame[:3, 3, :1]
    # vecx = Rot_m_g_frame[:3, 0,:1]
    # vecy = Rot_m_g_frame[:3, 1,:1]
    # vecz = Rot_m_g_frame[:3, 2,:1]
    # x_y_z = Rot_g_m_frame[:3, 3, :1]
    # vecx = Rot_g_m_frame[:3, 0 ,:1]
    # vecy = Rot_g_m_frame[:3, 1 ,:1]
    # # vecz = Rot_g_m_frame[:3, 2 ,:1]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1, 1, 1])
    # ax.axes.set_xlim3d(left=-1500, right=1500)
    # ax.axes.set_ylim3d(bottom=-1500, top=1500)
    # ax.axes.set_zlim3d(bottom=-1500, top=1500)
    # ax.scatter(marker[0, :, 137], marker[1, :, 137], marker[2, :, 137], color="r")
    # ax.scatter(m_mean[0, :, 137], m_mean[1, :, 137], m_mean[2, :, 137], color="y")
    # ax.scatter(m_mean_bis[0, :, 137], m_mean_bis[1, :, 137], m_mean_bis[2, :, 137], color="g")
    # ax.scatter(new_marker_sld[0, :, ], new_marker_sld[1, :], new_marker_sld[2, :])
    # ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecx[0], vecx[1], vecx[2], length=60, normalize=True, color="r")
    # ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecy[0], vecy[1], vecy[2], length=60, normalize=True, color="g")
    # ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecz[0], vecz[1], vecz[2], length=60, normalize=True, color="b")
    # ax.scatter(mark_scap[0, :, 137], mark_scap[1, :, 137], mark_scap[2, :, 137], color="g")
    # ax.scatter(new_mark[0, :, 137], new_mark[1, :, 137], new_mark[2, :, 137], color="b")
    # ax.scatter(coord_gen[0, :, 137], coord_gen[1, :, 137], coord_gen[2, :, 137], color="y")


    # origin = np.dot(Rot_m_g_frame[:, :, 0], np.zeros((4, 1)))
    # ax.scatter(origin[0, :], origin[1, :], origin[2, :], c="b")
    # # ax.scatter()
    # ax.scatter(marker_scap_g_sd[0, :, 137], marker_scap_g_sd[1, :, 137], marker_scap_g_sd[2, :, 137], c="r")
    # ax.scatter(pos_AA_Rm[0], pos_AA_Rm[1], pos_AA_Rm[2], c="b")
    # ax.scatter(pos_AI_Rm[0], pos_AI_Rm[1], pos_AI_Rm[2], c="b")
    # ax.scatter(pos_TS_Rm[0], pos_TS_Rm[1], pos_TS_Rm[2], c="b")
    # ax.scatter(mark_repere_scap[0, :,], mark_repere_scap[1, :,], mark_repere_scap[2, :,], c="b")

    # ax.scatter(mark_repere_scap_theorique[0, :,], mark_repere_scap_theorique[1, :,], mark_repere_scap_theorique[2, :,], c="r")


    # ax.scatter(markers_glob_rm[0, :, :1], markers_glob_rm[1, :, :1], markers_glob_rm[2, :, :1], c="r")
    # ax.scatter(marker_sld[0, :, :1], marker_sld[1, :, :1], marker_sld[2, :, :1], c="b")

    # ax.scatter(marker[0, 0, :], marker[1, 0, :], marker[2, 0, :], c="r")
    # ax.scatter(marker[0, 1, :], marker[1, 1, :], marker[2, 1, :], c="r")
    # ax.scatter(marker[0, 2, :], marker[1, 2, :], marker[2, 2, :], c="r")
    # ax.scatter(marker_scap_g_sd[0, :, :1], marker_scap_g_sd[1, :, :1], marker_scap_g_sd[2, :, :1], c="b")
    # ax.scatter(test_marker_scap[0, :, :], test_marker_scap[1, :, :], test_marker_scap[2, :, :], color="purple")
    # ax.scatter(marker_g[0, :, :1], marker_g[1, :, :1], marker_g[2, :, :1], c="b")
    # ax.scatter(coord_gen[0, :, :], coord_gen[1, :, :], coord_gen[2, :, :], c="y")
    #
    # plt.show()
    return coord_gen


def ajust_sl(pt1, pt2, pt3):
    # pt1: AI, pt2: TS, pt3: AA
    n_frames = pt1.shape[1]
    mat_hom_sl_g = create_axis_coordinates(pt1[:3, :], pt2[:3, :], pt3[:3,
                                                                   :])  # matrice de passage du repère créé associé au sl vers le repere global
    pt_sl = np.ones((4, 3, n_frames))
    for i in range(n_frames):
        mat = mat_hom_sl_g[:, :, i]
        matinv = np.linalg.inv(mat_hom_sl_g[:, :, i])
        pt_sl[:, 0, i] = matinv.dot(pt1[:, i])
        pt_sl[:, 1, i] = matinv.dot(pt2[:, i])
        pt_sl[:, 2, i] = matinv.dot(pt3[:, i])
        pt_sl[2, 0, i] += -155
        pt_sl[2, 1, i] += -155
        pt_sl[2, 2, i] += -155
        pt_sl[:, 0, i] = mat.dot(pt_sl[:, 0, i])
        pt_sl[:, 1, i] = mat.dot(pt_sl[:, 1, i])
        pt_sl[:, 2, i] = mat.dot(pt_sl[:, 2, i])
    # ordre marqueurs AI, TS, AA
    return pt_sl


def eval_err_dist(pt, pt_process):
    # pt et pt_process sont mat avec coord des points de la forme: xyz1, type marqueur, frame, il faut penser que les numérotations de marqueurs
    err = np.zeros(pt.shape[1])
    for i in range(pt.shape[1]):
        err[i] = np.linalg.norm(pt[:3, i, :] - pt_process[:3, i, :], axis=0).mean()
    return err


# for i in range (len(list_all_markers)):
# marker_sl.append(ajust_sl(list_all_markers[i][:,4,:],list_all_markers[i][:,5,:],list_all_markers[i][:,3,:]))


# tot=np.zeros((len(list_all_markers),3))
# for i in range (len(list_all_markers)):
#     tot[i,:]=eval_err_dist(marker_sl[i][:,:,:],markers_process[i][:,:,:])
# print(tot.mean(axis=0))
# #tot donne la distance séparant les points mesurés et calucl

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
    from biosiglive import save
    color = ["r", "g", "b", "y", "c", "m"]
    mark_type = ["o", "P", "X", "s", "v", "2"]
    ax = None
    for d, dic in enumerate(all_markers_dir):
        fig = plt.figure(noms_de_fichier[d])
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
        for k, key in enumerate(dic):
            idx = np.argwhere(np.mean(dic[key][0, :, :], axis=1) != 0)
            ax.scatter(dic[key][0, idx, :], dic[key][1, idx, :], dic[key][2, idx, :], marker=mark_type[k], c=color[k], label=key)
        plt.legend()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def load_markers(all_files, dic_init_mat, marker_names, mat_file_path, frame_of_interest=None):
    pos_AI_TS_d,pos_AI_TS_g, angle_equerre, equerre, Rot_a_m_d,Rot_a_m_g = dic_init_mat["pos_AI_TS_d"], \
                                                        dic_init_mat["pos_AI_TS_g"], \
                                                        dic_init_mat["angle_equerre"], \
                                                        dic_init_mat["equerre"], \
                                                        dic_init_mat["Rot_a_m_d"],\
                                                        dic_init_mat["Rot_a_m_g"]
    Rot_m_a_d = np.linalg.inv(Rot_a_m_d)
    Rot_a_m=Rot_a_m_d if "droite" in mat_file_path else Rot_a_m_g
    pos_AI_TS=pos_AI_TS_d if "droite" in mat_file_path else pos_AI_TS_g

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
        list_all_markers_data = (Markers.from_c3d(all_files[i]))
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
        # if "anat" in all_files[i] or "calib" in all_files[i] or "dyn" in all_files[i] or "pedal" in all_files[i]:
        #     print("------------- WARNING -------------"
        #           "\nThe adjustment is not apply on the anato file due to a quick fix. To remove with other data. ")
        #     mat_sl = mat_sl
        # else:
        mat_sl_temp = ajust_sl(mat_sl[:, 1, :], mat_sl[:, 2, :], mat_sl[:, 0, :]) if np.sum(mat_sl) != 0 else mat_sl
        mat_sl[:,0,:],mat_sl[:,1,:],mat_sl[:,2,:] = mat_sl_temp[:,2,:],mat_sl_temp[:,0,:],mat_sl_temp[:,1,:]

        list_all_markers.append(dict(mat_sl=mat_sl, mat_scap=mat_scap))
        mat_cluster = transfo_glob(pos_AI_TS, (angle_equerre) * (np.pi / 180), (equerre + 7.5),
                         mat_cluster, Rot_a_m, mat_file_path, list_all_markers, Rot_m_a_d)
        list_all_markers[-1]["mat_cluster"] = mat_cluster

    return list_all_markers


def _load_initial_matrix(lb1, lp1, lb2, lp2, angle_equerre, equerre):
    alpha_un = 0 * (np.pi / 180)
    beta = 40 * (np.pi / 180)
    # définition des angles conforme au plan de la pièce

    # M_droite_Ra = np.array(
    #     [[-52.24, 27.87, -18.50, 1], [2.24, 27.87, -18.50, 1], [-25.00, 27.87, -65.68, 1]]
    # )[:, :, np.newaxis]
    M_droite_Ra = np.array(
        [[-50.94933718,
          25.51144007,
          -19.83687349, 1], [1.32050063,
                             26.15961499,
                             -19.32593232, 1], [-18.72896108,
                                                27.22885813,
                                                -63.91126181, 1]]
    )[:, :, np.newaxis]
    M_gauche_Ra = np.array(
        [[2.24, 27.87, 18.5, 1], [-52.24, 27.87, 18.5, 1], [-25, 27.87, 65.68, 1]]
    )[:, :, np.newaxis]
    # M_gauche_Ra = np.array(
    #     [[2.24, 27.87, 18.5, 1], [-50.24, 27.87, 21.97, 1], [-20.96, 27.87, 60.89, 1]]
    # )[:, :, np.newaxis]
    # coord des pts M1,M2,M3 dans base Ra
    pos_AI_TS_d = np.array([5.69, (lp2 + 7.3 - 80), (lb2 + 14.25), 1]), np.array(
        [-(lb1 + 11.25), lp1 + 7.3 - 80, 5.69, 1])
    pos_AI_TS_g = np.array([-5.69, (lp2 + 7.3 - 80), - (lb2 + 14.25), 1]), np.array(
        [-(lb1 + 11.25), lp1 + 7.3 - 80, -3.5, 1])
    # pos_AI_TS_d = np.array([4.69, (lb2 - 25.3), (lp2 + 14.25), 1]), np.array(
    #     [-(lb1 + 11.25), -(lp1 + 12.7), 4.69, 1])

    Rot_m_a_d = create_axis_coordinates(M_droite_Ra[0, :3], M_droite_Ra[1, :3], M_droite_Ra[2, :3])[:, :, 0]
    Rot_m_a_g = create_axis_coordinates(M_gauche_Ra[0, :3], M_gauche_Ra[1, :3], M_gauche_Ra[2, :3])[:, :, 0]
    # creation du repère lié aux marqueurs et de matrice homogene de Rm vers Ra
    Rot_a_m_d = np.linalg.inv(Rot_m_a_d)
    Rot_a_m_g = np.linalg.inv(Rot_m_a_g)
    return dict(Rot_a_m_d=Rot_a_m_d, Rot_a_m_g=Rot_a_m_g, pos_AI_TS_d=pos_AI_TS_d, pos_AI_TS_g=pos_AI_TS_g, alpha_un=alpha_un,
                beta=beta, angle_equerre=angle_equerre, equerre=equerre)


def _load_initial_matrix(lb1, lp1, lb2, lp2, angle_equerre, equerre):
    # cette fonction prend en argument l'ensemble des mesures effectuées par l'utilisateur pour retourner
    # (sous forme de dictionnaire) l'ensemble des arguments utiles aux différentes transformations
    # (matrices homogènes de Ra vers Rm positions de AI et TS etc...)
    alpha_un = 0 * (np.pi / 180)
    beta = 40 * (np.pi / 180)
    # définition des angles conforme au plan de la pièce


    # M_droite_Ra = np.array([[-52.24, 27.87, -18.50, 1], [2.24, 27.87, -18.50, 1],
    #                         [-25.00, 27.87, -65.68, 1]])[:, :, np.newaxis]
    # ANCIENS MARQUEURS NON CORRIGES

    # M_gauche_Ra = np.array(
    #     [[2.24, 27.87, 18.5, 1], [-52.24, 27.87, 18.5, 1], [-25, 27.87, 65.68, 1]]
    # )[:, :, np.newaxis]
    # ANCIENS MARQUEURS NON CORRIGES

    # M_droite_Ra = np.array([[-50.94933718, 25.51144007,-19.83687349, 1], [1.32050063, 26.15961499,-19.32593232, 1],
    #                         [-18.72896108,27.22885813,-63.91126181, 1]])[:, :, np.newaxis]
    #ANCIENS MARQUEURS corrigées

    M_gauche_Ra = np.array(
        [[15.64,37.4,9.32, 1], [-65.64,37.4,9.32, 1], [-25,24.89,64.1, 1]]
    )[:, :, np.newaxis]
    # NOUVEAU MARQUEURS

    M_droite_Ra = np.array(
        [[-65.64,37.4,-9.32, 1], [15.64,37.4,-9.32, 1], [-25,24.89,-64.1, 1]]
    )[:, :, np.newaxis]
    # NOUVEAU MARQUEURS

    # M_gauche_Ra = np.array(
    #     [[20.7,42.1,6.4, 1], [-70.7,42.1,6.4, 1], [-25,29.58,69.96, 1]]
    # )[:, :, np.newaxis]
    # # NOUVEAU MARQUEURS RGBD
    #
    # M_droite_Ra = np.array(
    #     [[-70.7,42.1,-6.4, 1], [20.7,42.1,-6.4, 1], [-25,29.58,-69.96, 1]]
    # )[:, :, np.newaxis]
    # # NOUVEAU MARQUEURS RGBD
    #
    # M_droite_Ra = np.array(
    #     [[-72.55,40.66,-7.29, 1], [19.35,44.05,-7.18, 1], [-27.5,29.58,-69.96, 1]]
    # )[:, :, np.newaxis]
    # # NOUVEAU MARQUEURS RGBD_config
    #
    # M_gauche_Ra = np.array(
    #     [[22.55,40.66,7.29, 1], [-69.35,44.05,7.18, 1], [-22.5,29.58,69.96, 1]]
    # )[:, :, np.newaxis]
    # # NOUVEAU MARQUEURS RGBD_config

    pos_AI_TS_d = np.array([5.69, (lp2 +7.3-80), (lb2 + 14.25), 1]), np.array(
        [-(lb1 + 11.25), lp1 +7.3-80, 5.69, 1])
    pos_AI_TS_g = np.array([-5.69, (lp2 + 7.3 - 80), -(lb2 + 16.25), 1]), np.array(
        [-(lb1 + 11.25), lp1 + 7.3 - 80, -5.69, 1])

    #enregistrement des positions de AI et TS en fonctrion des données rentrées par l'utilisateur

    Rot_m_a_d = create_axis_coordinates(M_droite_Ra[0, :3], M_droite_Ra[1, :3], M_droite_Ra[2, :3])[:, :, 0]
    Rot_m_a_g = create_axis_coordinates(M_gauche_Ra[0, :3], M_gauche_Ra[1, :3], M_gauche_Ra[2, :3])[:, :, 0]
    # creation du repère lié aux marqueurs et de matrice homogene de Rm vers Ra
    Rot_a_m_d = np.linalg.inv(Rot_m_a_d)
    Rot_a_m_g = np.linalg.inv(Rot_m_a_g)
    #matrice homogène de Ra vers Rm

    return dict(Rot_a_m_d=Rot_a_m_d, Rot_a_m_g=Rot_a_m_g, pos_AI_TS_d=pos_AI_TS_d, pos_AI_TS_g=pos_AI_TS_g, alpha_un=alpha_un,
                beta=beta, angle_equerre=angle_equerre, equerre=equerre)


def compute_error(all_files, dic_init_mat, marker_names, mat_file_path, show=True,
                  frame_of_interest=None, print_error=True):
    all_markers_dict = load_markers(all_files,
    dic_init_mat,
    marker_names,
    mat_file_path,
    frame_of_interest
                                    )
    from biosiglive import save
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
        data_dic = {"file: ": noms_de_fichier[i], "data": all_markers_dict[i],
                    "rmse_sl_scap": rmse_sl_scap[i, :], "rmse_scap_cluster": rmse_scap_cluster[i, :],
                    "rmse_sl_cluster": rmse_sl_cluster[i, :], "marker_names": marker_names,}
        # save(add_data=True, data_dict=data_dic, data_path="data_scap/P7/markers_file_recons.bio")
    if print_error:
        print("Mean RMSE for AA, AI, TS between SL-SCAP : {}".format(np.mean(rmse_sl_scap, axis=0)))
        print("Mean RMSE for AA, AI, TS between SCAP-CLUSTER : {}".format(np.mean(rmse_scap_cluster, axis=0)))
        print("Mean RMSE for AA, AI, TS between SL-CLUSTER : {}".format(np.mean(rmse_sl_cluster, axis=0)))
    if show:
        plot(all_markers_dict, all_files)
    return rmse_sl_cluster, rmse_sl_scap, rmse_scap_cluster


if __name__ == '__main__':
    # lb1, lp1, lp2, lb2, angle_equerre, equerre = 90, 32, 65, 158, 9, 63
    # lb1, lp1, lp2, lb2, angle_equerre, equerre = 90, 32, 55, 158, -9, 63
    # lb1, lp1, lp2, lb2, angle_equerre, equerre = 109, 19, 55, 158, -9, 77
    # lb1, lp1, lp2, lb2, angle_equerre, equerre = 83, 28, 19, 132, -9, 39
    lb1, lp1, lp2, lb2, angle_equerre, equerre = 74, 35, 38, 135, -9, 46

    # data_files = "G:/Memoire S5/Prise de mesures/tiges/"
    # mat_file_path = "G:/Memoire S5/Projet/tracking_conf_mat_droite_02.json"
    data_files = "/home/amedeo/Documents/programmation/scapula_cluster/data_scap/P8/"
    mat_file_path = "/home/amedeo/Documents/programmation/scapula_cluster/calibration_matrix/tracking_conf_mat_gauche_03.json"

    # main_dir = "/media/amedeo/E6AC-7D2A/lucas/test_scap_robin"
    all_files = glob.glob(data_files + "/**.c3d")
    # all_files = [file for file in all_files if "test_droite" not in file]
    # frame_interet = [226, 226, 226]
    frame_of_interest = None
    # all_files.append("G:/Memoire S5/Prise de mesures/test_scap_robin/martin_anat.c3d")
    # liste avec les datapath de chaque essai
    noms_de_fichier = [os.path.basename(fichier) for fichier in all_files]
    marker_names = {"sl": ['slaa', 'slai', 'slts'],
                    "scap": ['scapaa', 'scapia', 'scapts'],
                    "cluster": ['M3', 'M2', 'M1']}

    dic_init_mat = _load_initial_matrix(lb1=lb1, lp1=lp1, lb2=lb2,
                                        lp2=lp2, angle_equerre=angle_equerre, equerre=equerre)
    # dic_init_mat_bis = _load_initial_matrix_bis(lb1, lp1, lb2, lp2, angle_equerre, equerre)

    error = compute_error(all_files, dic_init_mat, marker_names, mat_file_path, True, frame_of_interest)
    # error_bis = compute_error(all_files, dic_init_mat_bis, marker_names, mat_file_path, True, frame_of_interest)


    # matrice homogene de Ra vers Rm

# 1e colonne: AI, puis TS et AA