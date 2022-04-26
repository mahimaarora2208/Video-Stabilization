# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:48:56 2022

@author: sharm
"""

import numpy as np
import cv2


vid = cv2.VideoCapture('New video.mp4') 
# shape: (1080, 1920, 3)

# Get frame count
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) 
 
# Read first frame
_, previous = vid.read() 

previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) 

# Pre-define transformation-store array
transforms = np.zeros((frame_count - 1, 3), np.float32) 


for i in range(frame_count - 2):
    # Detect feature points in previous frame
     previous_pts = cv2.goodFeaturesToTrack(previous_gray,
                                            maxCorners = 200,
                                            qualityLevel = 0.01,
                                            minDistance = 30,
                                            blockSize = 3)
     
   
     # Read next frame
     success, current = vid.read() 
     if not success: 
         break 

     current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) 

     # Get feature points
     # Calculate optical flow
     current_pts, status, error = cv2.calcOpticalFlowPyrLK(previous_gray, 
                                                           current_gray, 
                                                           previous_pts, 
                                                           None) 

     # Sanity check
     assert previous_pts.shape == current_pts.shape 

     # Filter only valid points
     idx = np.where(status == 1)[0]
     previous_pts = previous_pts[idx]
     current_pts = current_pts[idx]

     # Find transformation matrix
     matrix = cv2.estimateRigidTransform(previous_pts, 
                                         current_pts, 
                                         fullAffine = False) 
     
     # Extract traslation and rotation   
     dx = matrix[0,2]
     dy = matrix[1,2]
     da = np.arctan2(matrix[1,0], matrix[0,0])
   
     # Store transformation
     transforms[i] = [dx, dy, da]
   
     # Reset the frame
     previous_gray = current_gray

     print("Frame: " + str(i) +  "/" + str(frame_count) + 
           " -  Tracked points : " + str(len(previous_pts)))


def get_smooth_curve(curve):
    radius = 50  # Smoothing radius
    win_size = 2 * radius + 1
    
    filter_ = np.ones(win_size)/win_size
    
    # Padding the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    
    # Convolution
    curve_smoothed = np.convolve(curve_pad, filter_, mode = 'same')
    
    # Removing the paddings
    curve_smoothed = curve_smoothed[radius:-radius]
    
    return curve_smoothed

def smooth_trajectory(traj):
    smooth_traj = np.copy(traj)
    
    # Filter the x, y and curve angles
    for ii in range(3):
        smooth_traj[: , ii] = get_smooth_curve(traj[: , ii])
        
    return smooth_traj

# Scaling the image to fix the borders
def fix_borders(fr):
    # Scaling 5%
    scaling = cv2.getRotationMatrix2D((fr.shape[1]/2, fr.shape[0]/2), 0, 1.05)  
    fr = cv2.warpAffine(fr, scaling, (fr.shape[1], fr.shape[0]))
    
    return fr


# Get smoothed trajectory
trajectory = np.cumsum(transforms, axis = 0) 
smoothed_trajectory = smooth_trajectory(trajectory) 

# Obtain smooth transforms
difference = smoothed_trajectory - trajectory
smooth_transforms = transforms + difference

# Reset stream the stream
vid.set(cv2.CAP_PROP_POS_FRAMES, 0) 

# Write n_frames-1 transformed frames
for i in range(frame_count - 2):

    # Read next frame
    success, frame = vid.read() 
    if not success:
        break

    # Extract and reconstruct transformations
    dx = smooth_transforms[i,0]
    dy = smooth_transforms[i,1]
    da = smooth_transforms[i,2]

    matrix = np.zeros((2,3), np.float32)
    matrix[0,0] = np.cos(da)
    matrix[0,1] = -np.sin(da)
    matrix[1,0] = np.sin(da)
    matrix[1,1] = np.cos(da)
    matrix[0,2] = dx
    matrix[1,2] = dy

    # Stabalize the frame
    frame_stabilized = cv2.warpAffine(frame, matrix, (frame.shape[1],frame.shape[0]))
    frame_stabilized = fix_borders(frame_stabilized) 

    # Resize the frames
    frame_stabilized = cv2.resize(frame_stabilized, 
                                (int(frame_stabilized.shape[1]/2), 
                                 int(frame_stabilized.shape[0]/2)))
  
    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
  
    cv2.imshow("Before", frame)
    cv2.imshow("After", frame_stabilized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
     
vid.release()        
cv2.destroyAllWindows()