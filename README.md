# Video-Stabilization

This project involves Implementation of [Conference Paper](https://github.com/nvnmangla/Video-Stabilization/blob/main/paper.pdf).

## Introduction:

Video stabilization using Sparse Optical Flow using a Mesh grid. This includes outlier rejection using median filtering, generating vertex profiles, and smoothening. The project was accomplished by a team of 3 members, which includes Naveen Mangla , [Mahima Arora](https://www.linkedin.com/in/ACoAACBA3TsBoA_q_kcTQQXxA5fowvfELuNf-Nw), and [Charu Sharma](https://www.linkedin.com/in/ACoAAB3r70oBDt4ONFnZHCDUyYFGVw_YINTQZcM).

## Steps to Run
```
git clone https://github.com/mahimaarora2208/Video-Stabilization.gitt
cd Video-Stabilization
python3 main.py --video="<your video path>"
```

## Description

There are many challenges faced by researchers when it comes to Video Stabilization but two challenges are prominent. First, camera motion estimation is often difficult and second, successful camera motion filtering often requires future frames. The more one reduces this buffer, the more will the future frame significantly deteriorate the results. 

Moreover, doing optical flow is computationally expensive, therefore we aim to do feature matching with which we get cheaper results but we also use similar techniques as optical flow i.e. they both encode strong spatial smoothness. For another, optical flow is dense and the Mesh Flow is sparse since it uses less profiles and only takes into consideration the mesh vertices instead of all pixel profiles hence, Mesh Flow can be regarded as a down-sampled dense flow.

<p align="center"> 
    <img src="https://github.com/nvnmangla/Video-Stabilization/blob/2f1e92c8286fba8e4792690944c1c1c99a3793f3/Results/steadyVsMesh.png" alt>

</p>
<p align="center"> 
    <em>Taking Pixel Profiles Vs taking Mesh Vertices for computing flow</em>
</p>

## Steps Followed 
- We record a video and place a regular 2D mesh on the video frame
- We then track image corners or "features" between consecutive frames, which yields a motion vector at each feature location.
<p align="center"> 
<img src="https://github.com/nvnmangla/Video-Stabilization/blob/b8a2ebd574213bf376dbc6270b9cd2d19a5ad729/Results/old_motion_vectors/2.jpg" alt>
</p>

<p align="center"> 
    <em>Motion Vectors</em>
</p>

- Next, these motion vectors are transferred to their corresponding nearby mesh vertexes,such that each vertex accumulates several motions from its surrounding feature
- With regards to the camera motion smoothing, we design a filter to smooth the temporal changes of the motion vector at each mesh vertex. This filter is applied to each mesh vertex
- Use PAPS (Predicted Adaptive Path Smoothing is when old frame flow is used to predict future flow of frame. We reduce excessive cropping and wobble distortion using this) for strong stabilization by using previous frame
<p align="center"> 
<img src="https://github.com/nvnmangla/Video-Stabilization/blob/3a4a1ee195857697c229e15350d6f4b0b82d5add/Results/paths/0_20.png" alt>
</p>

<p align="center"> 
    <em>Original vs Optimized Path of Vertex</em>
</p>

- Generate a stabilized video 





## Results 
<p align="center"> 
<img src="https://github.com/nvnmangla/Video-Stabilization/blob/377ca47fcdf89a687b7f9644454d8258a2c0536f/Results/result%20.gif" alt>
</p>

<p align="center"> 
    <em>Input Video  | Stable Video </em>
</p>
