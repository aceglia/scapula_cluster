import numpy as np
from pyomeca import Markers
from scapula_cluster.from_cluster_to_anato import ScapulaCluster
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = Markers.from_c3d("nouvel_ecran.c3d", usecols=['M1', 'M2', 'M3', 'B1', 'B2', 'B3'])
    data = data.values
    # plot the markers in 3D
    create_axis_coordinate = ScapulaCluster._create_axis_coordinates
    rm2rg = create_axis_coordinate(data[:3, 4, :1], data[:3, 5, :1], data[:3, 3, :1])
    rg2rm = np.linalg.inv(rm2rg[:, :, 0])[:, :, np.newaxis]
    rot = rg2rm
    points_in_ra = np.copy(data[:, :3, :1])
    points_in_ra[:, 0, 0] = (rot[:, :, 0].dot(data[:, 0, 0]))
    points_in_ra[:, 1, 0] = (rot[:, :, 0].dot(data[:, 1, 0]))
    points_in_ra[:, 2, 0] = (rot[:, :, 0].dot(data[:, 2, 0]))
    print("M1 in ra : {}".format([-points_in_ra[0, 0, 0], points_in_ra[2, 0, 0], points_in_ra[1, 0, 0], 1]))
    print("M2 in ra : {}".format([-points_in_ra[0, 1, 0], points_in_ra[2, 1, 0], points_in_ra[1, 1, 0], 1]))
    print("M3 in ra : {}".format([-points_in_ra[0, 2, 0], points_in_ra[2, 2, 0], points_in_ra[1, 2, 0], 1]))
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
    ax.scatter(data[0, 0+3, 0], data[1, 0+3, 0], data[2, 0+3, 0], c='r', marker='o')
    ax.scatter(data[0, 1+3, 0], data[1, 1+3, 0], data[2, 1+3, 0], c='g', marker='o')
    ax.scatter(data[0, 2+3, 0], data[1, 2+3, 0], data[2, 2+3, 0], c='b', marker='o')
    ax.scatter(data[0, :3, 0], data[1, :3, 0], data[2, :3, 0], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
