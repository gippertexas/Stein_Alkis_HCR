# Stein_Alkis_HCR
This project is for the Human Centered Robotics graduate course at UT Austin. Taking the navigation controller from [PRELUDE](https://ut-austin-rpl.github.io/PRELUDE) and implementing it onto the Atlas humanoid robot in the environment we have created from our homework.

## Note
Angle per turn can be changed in line 241 of dcm_trajectory_manager. Note: it cannot be larger than pi/2

## Setup
We created this project in Ubuntu 20.04. To setup this repository first install anaconda and activate conda with:
```
source {path to anaconda}/bin/activate
```
Create a virtual environment and install dependencies with:
```
conda env create -f hw4.yml
```
Activate the environment with:
```
conda activate ASE389
```
To run the code:
```
python simulator/pybullet/atlas_main.py
```
