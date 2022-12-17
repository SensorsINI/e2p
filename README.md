## Implementation of PDAVIS Live Demo

- [Table of Contents](#implementation-of-pdavis-Live-demo)
  * [1. Introduction](#1-introduction)
  * [2. Requirements](#2-requirements)
  * [3. How to Run It?](#3-how-to-run-it?)
  * [4. Results](#4-results)
    + [4.1. Saving Directory](#41-saving-directory)
  * [5. Citation](#5-citation)
  * [6. Contact](#6-contact)

### 1. Introduction

The polarization event camera PDAVIS is a novel bio-inspired neuromorphic vision sensor that reports both conventional polarization frames and asynchronous, continuously per-pixel polarization brightness changes (polarization events) with \textbf{fast temporal resolution} and \textbf{large dynamic range}.

### 2. Requirements
* Python 3.8.10
* PyTorch == 1.10.0
* TorchVision == 0.11.0
* CUDA 11.4
* tqdm
* timm

Lower version should be fine but not fully tested :-)

### 3. How to Run It?
 1. Download pre-trained polarization reconstruction model `e2p.pth` at [here](https://github.com/SensorsINI/pdavis_demo).
 2. Connect hardware: PDAVIS to USB.
 3. Find out which serial port device the Arduino appears on. You can use dmesg on linux. You can put the serial port into _globals_and_utils.py_ to avoid adding as argument.
 4. In first terminal run producer
  ```shell script
  python -m producer
  ```
 5. In a second terminal run consumer
  ```shell script
  python -m consumer  arduinoPort
  example: python -m consumer.py 
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

[Haiyang Mei](https://mhaiyang.github.io/) (haiyang.mei@outlook.com)

[Tobi Delbrück](https://www.ini.uzh.ch/~tobi/) (tobi@ini.uzh.ch)

**[⬆ back to top](#1-introduction)**