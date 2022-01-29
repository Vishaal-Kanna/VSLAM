import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import match
import reconstruct
import V_odom

sequence = '00'
seq_dir = 'Datasets/KITTI/data_odometry_gray/{}/'.format(sequence)
dir2 = 'Datasets/KITTI/data_odometry_velodyne/{}/'.format(sequence)
left_image_files = os.listdir(seq_dir + 'image_0')
right_image_files = os.listdir(seq_dir + 'image_1')
velodyne_files = os.listdir(dir2 + 'velodyne/')
num_frames = 10 #len(left_image_files)
#self.lidar_path = self.seq_dir + 'velodyne/'

calib = pd.read_csv(seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
P0 = np.array(calib.loc['P0:']).reshape((3,4))
P1 = np.array(calib.loc['P1:']).reshape((3,4))
Tr = np.array(calib.loc['Tr:']).reshape((3,4))
k_left, r_left, t_left, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
t_left = (t_left / t_left[3])[:3]
t_right = (t_right / t_right[3])[:3]
f = k_left[0,0]
b = t_right[0] - t_left[0]
In= np.matrix([[718.856 ,   0.    , 607.1928,0],\
        [  0.    , 718.856 , 185.2157,0],\
        [  0.    ,   0.    ,   1.,0],\
        [  0.    ,   0.    ,   0,1.]])
Ex=np.matrix([[1., 0., 0.,0],\
        [0., 1., 0.,0],\
        [0., 0., 1.,0],\
        [0., 0., 0,1.]]),

Pmatrix = np.dot(In,Ex)
minv = np.linalg.inv(Pmatrix)

u = np.zeros((376, 1241))
v = np.zeros((376, 1241))
for i in range(0,376):
    for j in range(0,1241):
        u[i][j] = 1241-j
        v[i][j] = 376-i
        
xf = np.zeros(num_frames)
yf = np.zeros(num_frames)
zf = np.zeros(num_frames)
trajectory1 = np.zeros((num_frames, 3, 4))
T_tot = np.eye(4)
trajectory1[0] = T_tot[:3, :]

for i in range(0,num_frames-1):
    if i%100==0:
       print(i)
    imgL1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[i],0)
    imgR1 = cv2.imread(seq_dir + 'image_1/' + right_image_files[i],0)
    imgL2 = cv2.imread(seq_dir + 'image_0/' + left_image_files[i+1],0)
    imgR2 = cv2.imread(seq_dir + 'image_1/' + right_image_files[i+1],0)
    lidar = np.fromfile(dir2 + 'velodyne/' + velodyne_files[i], dtype=np.float32, count=-1).reshape((-1, 4))
    
    depth1,points1 = imgto3D(imgL1,imgR1,minv,f,b,u,v)
    depth2,points2 = imgto3D(imgL2,imgR2,minv,f,b,u,v)  

    mask = np.zeros(points1[:,:,0].shape, dtype=np.uint8)
    ymax = 376
    xmax = 1241
    cv2.rectangle(mask, (96,0), (xmax,ymax), (255), thickness = -1)

    det = cv2.SIFT_create()
    kp1, des1 = det.detectAndCompute(imgL1,mask)
    kp2, des2 = det.detectAndCompute(imgL2,mask)

    matches = FLANN_matcher(des1,des2,2)

    filter_matches = filter_matches_distance(matches, 0.2)
    
    #visualize_matches(imgL1, kp1, imgL2, kp2, filter_matches)
    render = lidar2img(lidar, 376, 1241, Tr, P0)
    indices = np.where(render < 3000)
    depth1[indices] = render[indices]

    rmat, tvec, image1_points, image2_points = estimate_motion(filter_matches, kp1, kp2, k_left , minv, depth1)
    #print(rmat,tvec)
    Tmat = np.eye(4)
    Tmat[:3, :3] = rmat
    Tmat[:3, 3] = tvec.T
    #print(Tmat)
    T_tot = T_tot.dot(np.linalg.inv(Tmat))
    #print(Tmat)
    trajectory1[i+1, :, :] = T_tot[:3, :]

plt.plot(trajectory[:, :, 3][:, 0],trajectory[:, :, 3][:, 2])