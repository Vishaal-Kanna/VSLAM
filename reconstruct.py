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

    #stereo2 = cv2.ximgproc.createRightMatcher(stereo)

    #lamb = 8000
    #sig = 1.5
    #visual_multiplier = 1.0
    #wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    #wls_filter.setLambda(lamb)
    #wls_filter.setSigmaColor(sig)

    #disparity = stereo.compute(imgL, imgR)

    #disparity2 = stereo2.compute(imgR, imgL)
    #disparity2 = np.int16(disparity2)
    #print('1')
    #filteredImg = wls_filter.filter(disparity, imgL, None, disparity2)
    #_, filteredImg = cv2.threshold(filteredImg, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity = stereo.compute(imgL, imgR).astype(np.float32)/16
    #print('2')
    points = np.zeros((376, 1241,3))
            
    disparity[disparity <= 0.0] = 0.1
    #print('3')
    z = np.float32((f*b)/(disparity))
    #print('4')
    x = minv[0,0]*z*u+minv[0,1]*z*v+minv[0,2]*z
    y = minv[1,0]*z*u+minv[1,1]*z*v+minv[1,2]*z
    points[:,:,0] = x
    points[:,:,1] = y
    points[:,:,2] = z

    return z,points
