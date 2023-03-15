# Video-Stabilization

This project involves Implementation of [Conference Paper](https://github.com/nvnmangla/Video-Stabilization/blob/main/paper.pdf), 

## Introduction:

Video stabilization using Sparse Optical Flow using a Mesh grid. This includes outlier rejection using median filtering, generating vertex profiles, and smoothening. The project was accomplished by a team of 3 members, which includes Naveen Mangla , [Mahima Arora](https://www.linkedin.com/in/ACoAACBA3TsBoA_q_kcTQQXxA5fowvfELuNf-Nw), and [Charu Sharma](https://www.linkedin.com/in/ACoAAB3r70oBDt4ONFnZHCDUyYFGVw_YINTQZcM).

## Description

There are many challenges faced by researchers when it comes to Video Stabilization but two challenges are prominent. First, camera motion estimation is often difficult and second, successful camera motion filtering often requires future frames. The more one reduces this buffer, the more will the future frame significantly deteriorate the results. 

Moreover, doing optical flow is computationally expensive, therefore we aim to do feature matching with which we get cheaper results but we also use similar techniques as optical flow i.e. they both encode strong spatial smoothness. For another, optical flow is dense and the Mesh Flow is sparse since it uses less profiles and only takes into consideration the mesh vertices instead of all pixel profiles hence, Mesh Flow can be regarded as a down-sampled dense flow.

![Mesh Flow](https://github.com/nvnmangla/Video-Stabilization/blob/377ca47fcdf89a687b7f9644454d8258a2c0536f/Results/result%20.gif)

Results: 
![Video Output](https://github.com/nvnmangla/Video-Stabilization/blob/377ca47fcdf89a687b7f9644454d8258a2c0536f/Results/result%20.gif)
