import numpy as np
from pyomeca import Markers
from scapula_cluster.from_cluster_to_anato import ScapulaCluster
import matplotlib.pyplot as plt


def create_axis_coordinates(M1, M2, M3):
    first_axis_vector = M2 - M1
    second_axis_vector = M3 - M1
    second_axis_vector = -np.cross(first_axis_vector, second_axis_vector, axis=0)
    # third_axis_vector = np.cross(first_axis_vector, second_axis_vector, axis=0)

    third_axis_vector = np.cross(first_axis_vector, second_axis_vector, axis=0)
    n_frames = first_axis_vector.shape[1]
    rt = np.zeros((4, 4, n_frames))
    rt[:3, 0, :] = first_axis_vector / np.linalg.norm(first_axis_vector, axis=0)
    rt[:3, 1, :] = second_axis_vector / np.linalg.norm(second_axis_vector, axis=0)
    rt[:3, 2, :] = third_axis_vector / np.linalg.norm(third_axis_vector, axis=0)
    rt[:3, 3, :] = M2
    rt[3, 3, :] = 1
    return rt

if __name__ == '__main__':
    data = Markers.from_c3d("calib.c3d", usecols=['M1', 'M2', 'M3', 'scapaa',
                                                  'scapts', 'scapia', 'slts', 'slai'])
    # fig = plt.figure()
    # markers = data.values
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1, 1, 1])
    # for idx, mark in enumerate(data.channel.values):
    #     if mark in ["M1"]:  # , "M2", "M3"]:
    #         color = "r"
    #         marker = "o"
    #     elif mark in ["M2"]:  # , "M2", "M3"]:
    #         color = "r"
    #         marker = "x"
    #     elif mark in ["M3"]:  # , "M2", "M3"]:
    #         color = "r"
    #         marker = "P"
    #     elif mark in ['scap_aa_from_cluster', 'scap_ia_from_cluster', 'scap_ts_from_cluster']:
    #         color = "g"
    #         marker = "o"
    #     else:
    #         color = "b"
    #         marker = "o"
    #     ax.scatter(markers[0, idx, :1], markers[1, idx, :1], markers[2, idx, :1], c=color, marker=marker)
    # ax.set_xlabel("X Label")
    # ax.set_ylabel("Y Label")
    # ax.set_zlabel("Z Label")
    # plt.show()
    data = data.values
    data = data[:, :, 1055:1056]
    # plot the markers in 3D
    rm2rg = create_axis_coordinates(data[:3, 3, :1], data[:3, 4, :1], data[:3, 5, :1])
    rg2rm = np.linalg.inv(rm2rg[:, :, 0])[:, :, np.newaxis]
    rot = rg2rm
    points_in_ra = np.copy(data[:, :3, :1])
    points_in_ra[:, 0, 0] = (rot[:, :, 0].dot(data[:, 0, 0]))
    points_in_ra[:, 1, 0] = (rot[:, :, 0].dot(data[:, 1, 0]))
    points_in_ra[:, 2, 0] = (rot[:, :, 0].dot(data[:, 2, 0]))
    print("M1 in ra : {}".format([points_in_ra[0, 0, 0], points_in_ra[1, 0, 0], points_in_ra[2, 0, 0], 1]))
    print("M2 in ra : {}".format([points_in_ra[0, 1, 0], points_in_ra[1, 1, 0], points_in_ra[2, 1, 0], 1]))
    print("M3 in ra : {}".format([points_in_ra[0, 2, 0], points_in_ra[1, 2, 0], points_in_ra[2, 2, 0], 1]))
    points_init_in_ra = np.copy(data[:, :3, :1])
    points_init_in_ra[:, 0, 0] = (rot[:, :, 0].dot(data[:, 3, 0]))
    points_init_in_ra[:, 1, 0] = (rot[:, :, 0].dot(data[:, 4, 0]))
    points_init_in_ra[:, 2, 0] = (rot[:, :, 0].dot(data[:, 5, 0]))
    # distance between markers
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect([1, 1, 1])
    ax.scatter(points_in_ra[0, :3, 0], points_in_ra[1, :3, 0], points_in_ra[2, :3, 0], c='r', marker='x')
    ax.scatter(points_init_in_ra[0, :3, 0], points_init_in_ra[1, :3, 0], points_init_in_ra[2, :3, 0], c='b', marker='x')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    fig = plt.figure("bis")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    x_y_z = rm2rg[:3, 3, :]
    vecx = rm2rg[:3, 0, :]
    vecy = rm2rg[:3, 1, :]
    vecz = rm2rg[:3, 2, :]
    ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecx[0], vecx[1], vecx[2], length=60, normalize=False, color="r")
    ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecy[0], vecy[1], vecy[2], length=60, normalize=False, color="g")
    ax.quiver(x_y_z[0], x_y_z[1], x_y_z[2], vecz[0], vecz[1], vecz[2], length=60, normalize=False, color="b")

    ax.scatter(data[0, 0+3, 0], data[1, 0+3, 0], data[2, 0+3, 0], c='r', marker='o')
    ax.scatter(data[0, 1+3, 0], data[1, 1+3, 0], data[2, 1+3, 0], c='g', marker='o')
    ax.scatter(data[0, 2+3, 0], data[1, 2+3, 0], data[2, 2+3, 0], c='b', marker='o')
    ax.scatter(data[0, 0, 0], data[1, 0, 0], data[2, 0, 0], c='r', marker='x')
    ax.scatter(data[0, 1, 0], data[1, 1, 0], data[2, 1, 0], c='g', marker='x')
    ax.scatter(data[0, 2, 0], data[1, 2, 0], data[2, 2, 0], c='b', marker='x')
    # ax.scatter(data[0, :3, 0], data[1, :3, 0], data[2, :3, 0], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
