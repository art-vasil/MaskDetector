# MaskDetector

## Overview

This project is to detect if human is wearing a mask or not and to play the corresponding sound using OpenCV, Tensorflow 
and Deep learning model.

## Structure

- src

    The source code for mask detection and sound play
    
- utils

    * The deep learning model for mask detection
    * The source code for utilization
    
- app

    The main execution file

- requirements

    All the dependencies for this project

- settings

    Several settings including Camera setting and audio file path
    
## Installation

- Environment

    Windows 10, Ubuntu 18.04, Python 3.6
    
- Dependency Installation

    Please navigate to this project directory and run the following command in the terminal.
    ```
      pip3 install -r requirements.txt
    ```

## Execution

- If you intend to use Web Camera, please set WEB_CAM in settings file with True.

- If you intend to use IP Camera, please set WEB_CAM in settings file with False and set IP_CAM_ADDRESS with the address 
of IP Camera.

- Please set MASK_AUDIO_FILE_PATH & NON_MASK_AUDIO_FILE_PATH in settings with the full paths of audio file you want.

- Please run the following command in the terminal.

    ```
        python3 app.py
    ```

- If you stop project running, please simply click 'q' Key.
