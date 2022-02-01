import cv2
import pandas as pd
import numpy as np

def imgto3D(imgL,imgR,minv,f,b,u,v):
    max_disp = 192+160
    win_size = 6
    uniqueness_ratio = 15
    speckle_window_size = 200
    speckle_range = 2
    block_size = 11
    P1 = 8 * 3 * win_size ** 2
    P2 = 32 * 3 * win_size ** 2

    max_disparity = 192+160
    min_disparity = 0
    num_disparities = 6*16
    window_size = 6
    stereo = cv2.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities, blockSize = 11, uniquenessRatio = 15, speckleWindowSize = 200, speckleRange = 2, disp12MaxDiff = 2, P1 = P1, P2 = P2)

    disparity = stereo.compute(imgL, imgR).astype(np.float32)/16
    points = np.zeros((376, 1241,3))
    disparity[disparity <= 0.0] = 0.1

    z = np.float32((f*b)/(disparity))
    x = minv[0,0]*z*u+minv[0,1]*z*v+minv[0,2]*z
    y = minv[1,0]*z*u+minv[1,1]*z*v+minv[1,2]*z
    points[:,:,0] = x
    points[:,:,1] = y
    points[:,:,2] = z

    return z,points


def lidar2img(pointcloud, imheight, imwidth, Tr, P0):
    pointcloud = pointcloud[pointcloud[:, 0] > 0]
    pointcloud = np.hstack([pointcloud[:, :3], np.ones(pointcloud.shape[0]).reshape((-1, 1))])
    cam_xyz = Tr.dot(pointcloud.T)
    cam_xyz = cam_xyz[:, cam_xyz[2] > 0]
    depth = cam_xyz[2].copy()
    cam_xyz /= cam_xyz[2]
    cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])

    projection = P0.dot(cam_xyz)
    pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')
    indices = np.where((pixel_coordinates[:, 0] < imwidth)
                       & (pixel_coordinates[:, 0] >= 0)
                       & (pixel_coordinates[:, 1] < imheight)
                       & (pixel_coordinates[:, 1] >= 0)
                       )
    pixel_coordinates = pixel_coordinates[indices]
    depth = depth[indices]

    render = np.zeros((imheight, imwidth))
    for j, (u, v) in enumerate(pixel_coordinates):
        if u >= imwidth or u < 0:
            continue
        if v >= imheight or v < 0:
            continue
        render[v, u] = depth[j]

    render[render == 0.0] = 3861.45
    return render

