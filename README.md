## Implementation of Events to Polarizartion (E2P) PDAVIS Live Demo

- [Table of Contents](#implementation-of-pdavis-Live-demo)
  * [1. Introduction](#1-introduction)
  * [2. Requirements](#2-requirements)
  * [3. Run](#3-run)
  * [4. Results](#4-results)
    + [4.1. Saving Directory](#41-saving-directory)
  * [5. Citation](#5-citation)
  * [6. Contact](#6-contact)

### 1. Introduction

The polarization event camera PDAVIS is a novel bio-inspired neuromorphic vision sensor 
that outputs both conventional polarization frames and asynchronous, 
continuously per-pixel polarization brightness changes (polarization events) 
with **_fast temporal resolution_** and **_large dynamic range_**.

This project enables live demonstration of the E2P PDAVIS as illustrated below and training new and improved E2P networks; see [train](train) for training

![pdavis_demo_screen_230327.png](media%2Fpdavis_demo_screen_230327.png)


### 2. Requirements
#### 2.1. Ubuntu
From terminal
* create virtual environement. 
  ```
  mkvirtualenv pdavis_demo
  ```
* Python 3.8.10, CUDA 11.3, PyTorch == 1.11.0+cu113, TorchVision == 0.12.0+cu113
  ```
  workon pdavis_demo
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
  ```
* install libcear
  ```
  sudo apt-get update
  sudo apt-get install build-essential pkg-config libusb-1.0-0-dev
  git clone https://gitlab.com/inivation/dv/libcaer.git
  cd libcaer
  git checkout e68c3b4c115f59d5fd030fd44db12c702dddc3a5
  sudo apt install cmake
  cmake -DCMAKE_INSTALL_PREFIX=/usr .
  make -j
  sudo make install
  ```
* build pyaer (needed because pyaer master is broken at this time)
 
See 
* install other dependencies
  ```
  pip install -r requirements.txt
  ```
  
#### 2.2. Windows

We successfully run the PDAVIS demo on Windows 11 inside a WSL2 virtual Ubuntu 22 using [usbipd](https://github.com/dorssel/usbipd-win) to map the PDAVIS to WSL2.
* We use the handy Windows utility [wsl-usb-gui](https://gitlab.com/alelec/wsl-usb-gui) to control usbipd 
* We use the great Windows X server VcXsrv to develop with pycharm and display the demo output to the Windows 11 desktop.
* VcXsrv needs to be set to disable access control

### 3. Run
 1. The pretrained polarization reconstruction model [e2p-0317_215454-e2p-paper_plus_tobi_office-from-scratch.pth](models%2Fe2p-0317_215454-e2p-paper_plus_tobi_office-from-scratch.pth) is in the _models_ folde.
 2. Connect hardware: PDAVIS to USB.

#### Using a single command to launch producer and consumer processes using python multiprocessing
In a terminal``` run
```bash
python -m pdavis_demo
```

#### Using two terminals to run the producer (DAVIS) and consumer (E2P) processes

 1. In first terminal run producer 
    ```bash
    python -m producer 
    ```
 2. In a second terminal run consumer
    ```bash
    python -m consumer
    ```

### 4. Results

<p align="center">
    <img src="demo.png"/> <br />
    <em> 
    Figure 1: PDAVIS live demo: (left) raw polarization events and (right) polarization reconstruction results.
    </em>
</p>

#### 4.1. Saving Directory
The output files are automatically saved at the following location:

	./output
	├── xxx
	│   ├── xxx
    │   |   ├── xxx.png
    │   |   └── ...
	│   ├── xxx
    │   |   ├── xxx.png
    │   |   └── ...
    │   └── xxx
	├── xxx
	│   ├── xxx
    │   |   ├── xxx.png
    │   |   └── ...
	│   ├── xxx
    │   └── xxx
	└── xxx

## 5. Citation

If you find this project useful, please consider citing:

    @inproceedings{gruev2022division,
      title={Division of focal plane asynchronous polarization imager},
      author={Gruev, Viktor and Haessig, Germain and Joubert, Damien and Haque, Justin and Chen, Yingkai and Milde, Moritz and Delbruck, Tobi},
      booktitle={Polarization: Measurement, Analysis, and Remote Sensing XV},
      year={2022},
      organization={SPIE}
    }

### 6. Contact

[Tobi Delbrück](https://www.ini.uzh.ch/~tobi/) (tobi@ini.uzh.ch)

[Haiyang Mei](https://mhaiyang.github.io/) (haiyang.mei@outlook.com)



**[⬆ back to top](#1-introduction)**