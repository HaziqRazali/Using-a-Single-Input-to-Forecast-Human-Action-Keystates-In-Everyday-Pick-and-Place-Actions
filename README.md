# Using-a-Single-Input-to-Forecast-Human-Action-Keystates-In-Everyday-Pick-and-Place-Actions

We present a method that uses an input from a single timestep to directly forecast the human pose the instant a pick or place action is performed, as well as the time taken to complete the action. Published at the International Conference on Acoustics, Speech, & Signal Processing (ICASSP) 2022.

Also check out the method that does not predict the pose for every object in the scene but instead, uses eye-gaze to first predict the object-of-interest before predicting the human pose. Published at the International Conference on Robotics and Automation 2022. [[Code]](https://github.com/HaziqRazali/Using-Eye-Gaze-to-Forecast-Human-Pose-in-Everyday-Pick-and-Place-Actions)

# Contents
------------
  * [Requirements](#requirements)
  * [Brief Project Structure](#brief-project-structure)
  * [Results](#results)
  * [Visualization](#visualization)
  * [Training](#training)
  * [Testing](#testing)
  * [Future Work](#usage)
  * [References](#references)

# Requirements
------------

What we used to develop the system

  * ubuntu 18.04
  * anaconda
  
For visualization

  * pyqtgraph
  * pyopengl
  
For training and testing

  * pytorch
  * tensorflow
  * tensorboardx
  * opencv

# Brief Project Structure
------------

    ├── data          : directory containing the data and the scripts to visualize the output
    ├── dataloader    : directory containing the dataloader script
    ├── misc          : directory containing the argparser, loss computations, and checkpointing scripts
    ├── model         : directory containing the model architecture
    ├── results       : directory containing some sample results
    ├── shell scripts : directory containing the shell scripts that run the train and test scripts
    ├── weights       : directory containing the model weights
    ├── test.py       : test script
    ├── train.py      : train script
    
# Results
------------

  * Given the input pose in white and the coordinates of the key object in blue, our method predicts the red key pose the instant the person picks or places the object as well as the time taken to complete the action.

<img src="/misc/intro.png" alt="1" width="500"/>

# Visualization
------------

  * To visualize a sample output shown above, download the [processed data](https://imperialcollegelondon.box.com/s/vklj025ldbhueb3t1penk3oovha86336) then unzip it to `./data` as shown in [Brief Project Structure](#brief-project-structure) and run the following commands:
 
```
conda create -n visualizer python=3.8 anaconda
conda install -c anaconda pyqtgraph
conda install -c anaconda pyopengl

cd data/scripts/draw-functions
python draw-pred.py --frame 26777 --draw_pred_pose 1
```

  * Note that the `./results` folder already contain some sample results.
  
# Training
------------

  * To train a model, download the [processed data](https://imperialcollegelondon.box.com/s/vklj025ldbhueb3t1penk3oovha86336) then unzip it to `./data` as shown in [Brief Project Structure](#brief-project-structure) and run the following commands:

```
conda create -n forecasting python=3.8 anaconda
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge tensorflow
conda install -c conda-forge tensorboardx
conda install -c conda-forge opencv

cd shell_scripts/ICASSP2022
./vanilla-train.sh
```

  * The weights will then be stored in `./weights`

# Testing
------------

  * To test the model, run the following command:

```
./vanilla-test.sh
```

  * The outputs will then be stored in `./results` that can be visualized by following the commands listed in [Visualization](#visualization). Note that the `./weights` folder already contain a set of pretrained weights.

# References
------------

```  
@InProceedings{haziq2022keystate,  
author = {Razali, Haziq and Demiris, Yiannis},  
title = {Using a Single Input to Forecast Human Action Keystates In Everyday Pick and Place Actions},  
booktitle = {International Conference on Acoustics, Speech, & Signal Processing},  
year = {2022}  
}  
```
