# Stein_Alkis_HCR
This project is for the Human Centered Robotics graduate course at UT Austin. Taking the navigation controller from [PRELUDE](https://ut-austin-rpl.github.io/PRELUDE) and implementing it onto the Atlas humanoid robot in the environment we have created from our homework.

## Introduction
We implemented the navigation controller from the PRELUDE project and a point cloud with clearance to avoid still objects in a hallway environment. 


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
