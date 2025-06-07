
# Particle Filter SLAM
This project (tries to) implements FastSLAM in python with some CUDA assistance for mapping. It also contains a simple lidar robot simulation to test the algorithm. Refer to the following GIF for an example.

![Alt Text](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZXRtdmpmdXN5MGc4bXo1Zzc4aDRldnpob28zNDgzc2k5MzdjODF4MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/mm3rpJBwXkYpODXvNh/giphy.gif)

## Features

- Robot simulation environment
- FastSLAM implementation for mapping and localization
- Real-time visualization of robot movement and mapping
- Particle filter-based position estimation
- some other sweet things :)
---
## Prerequisites

- Python 3.x
- CUDA-capable GPU
- patience
- Install all necessary libraries (a good chunk) as detailed in the requirements.txt

## Project Structure

The project consists of several key components:

- `custom_sim.py`: Contains the robot simulation environment and wall-following logic
- `fastslam.py`: Implements the SLAM algorithm
- `mapping.py`: Implements the occupancy grid and related tools
- `icp.py`: The ICP implementation for matching.
- `tools.py`: Utility functions including angle normalization

## Usage

To run the simulation, run the main.py file.

---
## How It Works
1. Given motion and scan in meters, first the particles are updated with the motion model. For future reference, it
might be good to implement specialized motion models for certain robot chassis.
2. After the particle poses are predicted, ICP finds (or tries, at least) the transformation from the scan to the world.
This transformation is added to the particle's pose, and the mismatch between the scan and world is used as weight
3. With the scan and final poses, all the maps, poses and the scan get sent to the GPU to register.
4. Weights get normalized, and if necessary, particles are resampled.
5. Repeat!
---
#### Disclosure: This readme was made by AI. The project utilizes AI guidance and snippets, but it was built by a human. That makes crappy code my responsibility :(   
--- GG :)