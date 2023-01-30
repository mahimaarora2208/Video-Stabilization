"""
Copyright (c) 2021
@author Naveen Mangla (nmangla@umd.edu)
@author Mahima Arora (marora1@umd.edu)
@author Charu Sharma  
"""


import cv2
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import medfilt


def get_mesh_flow(old_frame,old_pts,new_pts,grid,radius=300) :
    cols, rows = old_frame.shape[1]//grid, old_frame.shape[0]//grid

    H, _ = cv2.findHomography(old_pts, new_pts, cv2.RANSAC)
    sudd_x_motion = {}; sudd_y_motion = {}

    for i in range(rows):
        for j in range(cols):
            pt = np.array([grid*j, grid*i,1])                                  
            ptrans = H@pt
            ptrans = ptrans/ptrans[-1]

            sudd_x_motion[i, j] = pt[0]-ptrans[0]
            sudd_y_motion[i, j] = pt[1]-ptrans[1]
    
    med_x_motion = {}; med_y_motion = {}


    ## Translating features motion onto mesh vertex
    for i in range(rows):
        for j in range(cols):
            vertex = [grid*j, grid*i]
            for pt, st in zip(old_pts, new_pts):
                pt = np.array([pt[0],pt[1],1])

                # velocity = point - feature point in current frame

                dst = np.sqrt((vertex[0]-pt[0])**2+(vertex[1]-pt[1])**2)

                if dst < radius:
                    ptrans = H@pt
                    ptrans = ptrans/ptrans[-1]

                    try:
                        med_x_motion[i, j].append(st[0]-ptrans[0])
                    except:
                        med_x_motion[i, j] = [st[0]-ptrans[0]]
                    try:
                        med_y_motion[i, j].append(st[1]-ptrans[1])
                    except:
                        med_y_motion[i, j] = [st[1]-ptrans[1]]
    
    # apply median filter (f-1) on obtained motion for each vertex
    x_motion_mesh = np.zeros((rows, cols), dtype=float)
    y_motion_mesh = np.zeros((rows, cols), dtype=float)
    
    for key in sudd_x_motion.keys():
        try:
            med_x_motion[key].sort()
            x_motion_mesh[key] = sudd_x_motion[key]+ med_x_motion[key][len(med_x_motion[key])//2]
        except KeyError:
            x_motion_mesh[key] = sudd_x_motion[key]
        try:
            med_y_motion[key].sort()
            y_motion_mesh[key] = sudd_y_motion[key]+med_y_motion[key][len(med_y_motion[key])//2]
        except KeyError:
            y_motion_mesh[key] = sudd_y_motion[key]
    
    # apply second median filter (f-2) over the motion mesh for outliers
    x_motion_mesh = medfilt(x_motion_mesh, kernel_size=[3, 3])  # using scipy medfilt 
    y_motion_mesh = medfilt(y_motion_mesh, kernel_size=[3, 3])
    
    return x_motion_mesh, y_motion_mesh


# To get spacial guassian weights over the window size
def gauss(t, r, window_size):
    return np.exp((-9*(r-t)**2)/window_size**2)


# To get optimized mesh vertex profiles in x-direction & y-direction
def get_optimized_path(path, lamda = 100,beta = 1,buffer_size = 100, iterations = 10, window_size = 32):
    p = np.empty_like(path)
    
    W = np.zeros((buffer_size, buffer_size))

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i,j] = gauss(i, j, window_size) 

    bar = tqdm(total=path.shape[0]*path.shape[1])
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            y = []
            d = None
            
            for channels in range(1, path.shape[2]+1):
                if channels <= buffer_size:
                    P = np.asarray(path[i, j, :channels])
                    if d is not None:
                        for _ in range(iterations):
                            alpha = path[i, j, :channels] + lamda*W[:channels, :channels]@P
                            alpha[:-1] = alpha[:-1] + beta*d
                            gamma = 1 + lamda*W[:channels, :channels]@np.ones((channels,))
                            gamma[:-1] = gamma[:-1] + beta
                            P = np.divide(alpha, gamma)
                else:
                    P = np.asarray(path[i, j, channels-buffer_size:channels])
                    for _ in range(iterations):
                        alpha = path[i, j, channels-buffer_size:channels] + lamda*W@P
                        alpha[:-1] = alpha[:-1] + beta*d[1:]
                        gamma = 1 + lamda*W@np.ones((buffer_size, ))
                        gamma[:-1] = gamma[:-1] + beta
                        P = np.divide(alpha, gamma)

                d = np.asarray(P)
                y.append(P[-1])
            p[i, j, :] = np.asarray(y)
            bar.update(1)
    bar.close()
    return p


## Comparison between original and smoothed path 
# To plot the vertex profiles
def plot_vertex_profiles(xpaths, sxpaths):
    print("Plotting the vertex profiles")

    for i in tqdm(range(0, xpaths.shape[0])):
        for j in range(0, xpaths.shape[1], 10):
            plt.plot(xpaths[i, j, :])
            plt.plot(sxpaths[i, j, :])
            plt.savefig(path+'results/paths/'+str(i)+'_'+str(j)+'.png')
            plt.clf()
    
# To get a update motion mesh for each frame with which it needs to be warped
def get_frame_warp(x_meshes, y_meshes, xpaths, ypaths, optx_paths, opty_paths):
    # U = P-C
    x_meshes = np.dstack((x_meshes, x_meshes[:,:,-1]))
    y_meshes = np.dstack((y_meshes, y_meshes[:,:,-1]))
    new_x_meshes = optx_paths - xpaths
    new_y_meshes = opty_paths - ypaths

    return x_meshes, y_meshes, new_x_meshes, new_y_meshes

# To get a mesh warped frame according to given motion meshes
def warp_frame(frame, x_mesh, y_mesh):
    # define handles on mesh in x-direction
    map_x = np.zeros((frame.shape[0], frame.shape[1]), np.float32)
    
    # define handles on mesh in y-direction
    map_y = np.zeros((frame.shape[0], frame.shape[1]), np.float32)
    
    for i in range(x_mesh.shape[0]-1):
        for j in range(x_mesh.shape[1]-1):

            src = [[j*grid_size, i*grid_size],
                   [j*grid_size, (i+1)*grid_size],
                   [(j+1)*grid_size, i*grid_size],
                   [(j+1)*grid_size, (i+1)*grid_size]]
            src = np.asarray(src)
            
            dst = [[j*grid_size+x_mesh[i, j], i*grid_size+y_mesh[i, j]],
                   [j*grid_size+x_mesh[i+1, j], (i+1)*grid_size+y_mesh[i+1, j]],
                   [(j+1)*grid_size+x_mesh[i, j+1], i*grid_size+y_mesh[i, j+1]],
                   [(j+1)*grid_size+x_mesh[i+1, j+1], (i+1)*grid_size+y_mesh[i+1, j+1]]]
            dst = np.asarray(dst)

            H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
            
            
            for k in range(grid_size*i, grid_size*(i+1)):
                for l in range(grid_size*j, grid_size*(j+1)):
                    pt = np.array([l,k,1]).reshape(3,1)
                    pt_ = H@pt
                    pt_ = pt_/pt_[-1]
                    
                    map_x[k, l] = pt_[0]
                    map_y[k, l] = pt_[1]
    
    # repeat motion vectors for remaining frame in y-direction
    for i in range(grid_size*x_mesh.shape[0], map_x.shape[0]):
            map_x[i, :] = map_x[grid_size*x_mesh.shape[0]-1, :]
            map_y[i, :] = map_y[grid_size*x_mesh.shape[0]-1, :]
    
    # repeat motion vectors for remaining frame in x-direction
    for j in range(grid_size*x_mesh.shape[1], map_x.shape[1]):
            map_x[:, j] = map_x[:, grid_size*x_mesh.shape[0]-1]
            map_y[:, j] = map_y[:, grid_size*x_mesh.shape[0]-1]
            
    # deforms mesh
    new_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return new_frame


########### Change this path according to your directory ############
######################################################################

path = ""

#######################################################################
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1000,
                    qualityLevel = 0.1,
                    minDistance = 7,
                    blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

#####################################################################


######### Video Reading #################

start_time = time.time()

cap = cv2.VideoCapture(path + "input_video.mp4")
frames = []

# Getting video properties
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

print("Reading Video")
while (cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    
    cv2.imshow('Frame',frame)
    frames.append(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else: 
    break
cap.release()
cv2.destroyAllWindows()

#########################################


############ Computation ################

#### Calculating optical flow Lucas-Kanade 
old_frame = frames[0]

# Block of size in mesh
grid_size = 16

x_motion_meshes = np.zeros((old_frame.shape[0]//grid_size, old_frame.shape[1]//grid_size, 1))
y_motion_meshes = np.zeros((old_frame.shape[0]//grid_size, old_frame.shape[1]//grid_size, 1))
x_paths = np.zeros((old_frame.shape[0]//grid_size, old_frame.shape[1]//grid_size, 1))
y_paths = np.zeros((old_frame.shape[0]//grid_size, old_frame.shape[1]//grid_size, 1))



for i in tqdm(range(len(frames)-1)):
    c_frame = frames[i+1]
    old_frame = frames[i]

    c_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # finding corners
    corners = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    
    # Calculating the Optical flow
    flow, flag, _ = cv2.calcOpticalFlowPyrLK(old_gray, c_gray, corners, None, **lk_params)

    #Selecting good feature points 
    if flow is not None:

        good_new = flow[flag==1]
        good_old = corners[flag==1]
    
    # Estimaing motion mesh for old_frame
    motion_mesh_x, motion_mesh_y = get_mesh_flow(old_frame,good_old,good_new,grid_size) 

    try:
        x_motion_meshes = np.dstack((x_motion_meshes,motion_mesh_x))
        y_motion_meshes = np.dstack((y_motion_meshes,motion_mesh_y))

    except:
        x_motion_meshes = np.expand_dims(motion_mesh_x, axis=2)
        y_motion_meshes = np.expand_dims(motion_mesh_y, axis=2)

    new_x_path = x_paths[:, :, -1] + motion_mesh_x
    new_y_path = y_paths[:, :, -1] + motion_mesh_y
    x_paths,y_paths= np.dstack((x_paths,new_x_path)),np.dstack((y_paths,new_y_path))

print('Optimization')
optimized_path_x = get_optimized_path(x_paths)
optimized_path_y = get_optimized_path(y_paths)

# visualize optimized paths
plot_vertex_profiles(x_paths, optimized_path_x)

# get updated mesh warps
x_motion_meshes, y_motion_meshes, new_motion_meshes_x, new_motion_meshes_y = get_frame_warp(x_motion_meshes, y_motion_meshes, x_paths, y_paths, optimized_path_x, optimized_path_y)



frame_width = old_frame.shape[1]
frame_height = old_frame.shape[0]


# Generate stabilized video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('results/stable.mp4', fourcc, frame_rate, (frame_width, frame_height))


print('Generating the Motion Vectors')
for frame_num in tqdm(range(len(frames) - 1)):
    
    frame = frames[frame_num]
    
    # reconstruct from frames
    x_motion_mesh = x_motion_meshes[:, :, frame_num]
    y_motion_mesh = y_motion_meshes[:, :, frame_num]
      
    new_x_motion_mesh = new_motion_meshes_x[:, :, frame_num]
    new_y_motion_mesh = new_motion_meshes_y[:, :, frame_num]
    
    # mesh warping
    new_frame = warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh)
    

    output = cv2.hconcat(frame, new_frame)
    cv2.imshow("window",output)
    cv2.waitKey(1)
    
    out.write(output)
    print('writing')
    
    # draw old motion vectors
    r = 5
    for i in range(x_motion_mesh.shape[0]):
        for j in range(x_motion_mesh.shape[1]):
            theta = np.arctan2(y_motion_mesh[i, j], x_motion_mesh[i, j])
            cv2.line(frame, (j*grid_size, i*grid_size), (int(j*grid_size+r*np.cos(theta)), int(i*grid_size+r*np.sin(theta))), 1)
    cv2.imwrite(path+'results/old_motion_vectors/'+str(frame_num)+'.jpg', frame)


    # draw new motion vectors
    for i in range(new_x_motion_mesh.shape[0]):
        for j in range(new_x_motion_mesh.shape[1]):
            theta = np.arctan2(new_y_motion_mesh[i, j], new_x_motion_mesh[i, j])
            cv2.line(new_frame, (j*grid_size, i*grid_size), (int(j*grid_size+r*np.cos(theta)), int(i*grid_size+r*np.sin(theta))), 1)
    cv2.imwrite(path+'results/new_motion_vectors/'+str(frame_num)+'.jpg', new_frame)

out.release()
cv2.destroyAllWindows()

