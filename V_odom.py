def estimate_motion(match, kp1, kp2, In, minv, depth1, max_depth=3000):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    
    image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in match])
    
    object_points = np.zeros((0, 3))
    delete = []
    
    for i, (u, v) in enumerate(image1_points):
            z = depth1[int(v), int(u)]
            if z > max_depth:
                delete.append(i)
                continue
            x = minv[0,0]*z*u+minv[0,1]*z*v+minv[0,2]*z
            y = minv[1,0]*z*u+minv[1,1]*z*v+minv[1,2]*z
            object_points = np.vstack([object_points, np.array([x, y, z])])
            
    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)
        
    # Use PnP algorithm with RANSAC for robustness to outliers
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, In, None)
    #print('Number of inliers: {}/{} matched features'.format(len(inliers), len(match)))
        
    # Above function returns axis angle rotation representation rvec, use Rodrigues formula
    # to convert this to our desired format of a 3x3 rotation matrix
    rmat = cv2.Rodrigues(rvec)[0]
       
    return rmat, tvec, image1_points, image2_points