# Deep Polarization Reconstruction with PDAVIS Events
#### Haiyang Mei    
#### _INI_ 
#### _UZH / ETH Zurich_

### Network
1. RPPP
2. Interaction
3. Characteristics

### Experimental Environment
- Ubuntu 20.04.3 LTS
- NVIDIA GeForce RTX 3080 (10G Memory)
- NVIDIA-SMI = 470.94
- cuda = 11.3
- Python = 3.8.10
- torch-1.11.0+cu113 torchaudio-0.11.0+cu113 torchvision-0.12.0+cu113
- cv2 = 4.5.5
- numpy = 1.22.2
- pandas = 1.4.1
- h5py = 3.6.0

### Data Location
1. train txt
- ./data/movingcam/train_mix2.txt

2. synthetic/real data file location
- ./data/raw_demosaicing_polarization_5s_iad/014007_iad.h5
- ./data/real/real-00.h5

3. synthetic/real output location
- /home/mhy/firenet-pdavis/output/v16_mix2
- /home/mhy/firenet-pdavis/output_real
- /home/mhy/firenet-pdavis/txt_eval/v16_mix2.txt

### Generate real PDAVIS hdf5 data
- `python my_extract_aps.py`
- Manually extract from jaer instead of runing `python my_extract_events.py`
- `python my_video2frame.py`
- `python my_merge_h5.py`
- `python my_create_real_list.py`

### 3090/V100 Server
- Python 3.8.13
- PyTorch 1.12.0a0+8a1a93a

- `sudo docker pull nvcr.io/nvidia/pytorch:22.05-py3`
- `pip install -r requirements.txt`
- `apt-get update`
- `apt-get install libgl1`
- `python utils/create_link.py`

- `git clone https://gitee.com/meihaiyang/cutlass.git`
- `cd cutlass/examples/19_large_depthwise_conv2d_torch_extension`
- `./setup.py install`
- A quick check: `python depthwise_conv2d_implicit_gemm.py`
- `add2virtualenv cutlass/examples/19_large_depthwise_conv2d_torch_extension`

## Work with real aedat2 file
for FireNet, refer to project aedat2pvideo

for my own model, run the following command: 
- `python my_test_on_real_data.py`
- `python visualize_pvideo_for_real_data.py` (can visualize both firenet and my own model)


### Train
`sh my_train.sh`

### Test
- `python my_test.py`
  - `python raw2p.py`
  - `python direction2p`
- `python visualize_pvideo.py`
  - `python h5gt2iad.py`
  - `python visualize_pvideo.py`
- `python align_prediction_and_gt.py`

### Evaluation
`python evaluation.py`

### For Real
- `python my_test_firenet.py`
- `python utils/h5gt2iad.py`

### Moving Camera Dataset Visualization
- `workon e2v`

see gt polarization from original raw data as done in glass project
- `cd utils`
  - `python process_raw_demosaicing.py` or `python process_crop_demosaicing.py`
  - `python visualize_pvideo.py`

see gt polarization from v2e output h5 file
- `cd utils`
  - `python h5gt2p.py` or `python h5gt2iad.py`
  - `python visualize_pvideo.py`

### Data Generation

- `workon v2e`
  - `python extract_from_PHSPD_to_pvideo.py`
  - `sh my_run.sh`
  - copy output h5 files to data/train_or_test folder
- `workon e2v`
  - `python h5gt2ph5.py`
  - `python save_frame.py`
- `conda activate flownet2`
  - `sh my_inference.sh`
  - copy vis and npy files to data folder and rename them
- `workon e2v`
  - `python add_flow.py`

### Workflow

- v2e 
  - `sh my_run.sh`
- split into four channels. txt --> 0 45 90 135 txt 
  - `python split.py`
- reconstruction. txt --> frame 
  - `sh my_run_reconstruction.sh`
- compute polarization. frame --> polarization
  - `python f42p.py`
- concat intensity, aolp, and dolp. polarization --> concat
  - `python concat_iad.py`
    - not aligned strictly
- generate gt polarization
  - `python gt2p.py`
- evaluation
  - `python evaluation.py`

### Records
- 17.10.2022
  - Can train on real PDAVIS data
  - Fix the align problem
- 16.07.2022
  - Finally work!
- 01.04.2022
  - generate_h5_dataset
- 02.04.2022
  - check e.h5 and modify it to match the training code or revise the v2e code.
  - try to figure out how `davis_output` option in e2v work.
  


## Install

Dependencies:

- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.0
- [NumPy](https://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [OpenCV](https://opencv.org/)

### Install with Anaconda

The installation requires [Anaconda3](https://www.anaconda.com/distribution/).
You can create a new Anaconda environment with the required dependencies as follows (make sure to adapt the CUDA toolkit version according to your setup):

```bash
conda create -n E2VID
conda activate E2VID
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install pandas
conda install -c conda-forge opencv
```

## Run

- Download the pretrained [FireNet model](https://drive.google.com/file/d/1nBCeIF_Us-rGhCjdU5q1Ch-yrFckjZPa/view?usp=sharing)

or

- Download the pretrained E2VID model:

```bash
wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" -O pretrained/E2VID_lightweight.pth.tar
```

- Download an example file with event data:

```bash
wget "http://rpg.ifi.uzh.ch/data/E2VID/datasets/ECD_IJRR17/dynamic_6dof.zip" -O data/dynamic_6dof.zip
```

Before running the reconstruction, make sure the conda environment is sourced:

```bash
conda activate E2VID
```

- Run reconstruction:

```bash
python run_reconstruction.py \
  -c firenet_1000.pth.tar \
  -i data/dynamic_6dof.zip \
  --auto_hdr \
  --display \
  --show_events
```

## Parameters

Below is a description of the most important parameters:

#### Main parameters

- ``--window_size`` / ``-N`` (default: None) Number of events per window. This is the parameter that has the most influence of the image reconstruction quality. If set to None, this number will be automatically computed based on the sensor size, as N = width * height * num_events_per_pixel (see description of that parameter below). Ignored if `--fixed_duration` is set.
- ``--fixed_duration`` (default: False) If True, will use windows of events with a fixed duration (i.e. a fixed output frame rate).
- ``--window_duration`` / ``-T`` (default: 33 ms) Duration of each event window, in milliseconds. The value of this parameter has strong influence on the image reconstruction quality. Its value may need to be adapted to the dynamics of the scene. Ignored if `--fixed_duration` is not set.
- ``--Imin`` (default: 0.0), `--Imax` (default: 1.0): linear tone mapping is performed by normalizing the output image as follows: `I = (I - Imin) / (Imax - Imin)`. If `--auto_hdr` is set to True, `--Imin` and `--Imax` will be automatically computed as the min (resp. max) intensity values.
- ``--auto_hdr`` (default: False) Automatically compute `--Imin` and `--Imax`. Disabled when `--color` is set.
- ``--color`` (default: False): if True, will perform color reconstruction as described in the paper. Only use this with a [color event camera](http://rpg.ifi.uzh.ch/CED.html) such as the Color DAVIS346.

#### Output parameters

- ``--output_folder``: path of the output folder. If not set, the image reconstructions will not be saved to disk.
- ``--dataset_name``: name of the output folder directory (default: 'reconstruction').

#### Display parameters

- ``--display`` (default: False): display the video reconstruction in real-time in an OpenCV window.
- ``--show_events`` (default: False): show the input events side-by-side with the reconstruction. If ``--output_folder`` is set, the previews will also be saved to disk in ``/path/to/output/folder/events``.

#### Additional parameters

- ``--num_events_per_pixel`` (default: 0.35): Parameter used to automatically estimate the window size based on the sensor size. The value of 0.35 was chosen to correspond to ~ 15,000 events on a 240x180 sensor such as the DAVIS240C.
- ``--no-normalize`` (default: False): Disable event tensor normalization: this will improve speed a bit, but might degrade the image quality a bit.
- ``--no-recurrent`` (default: False): Disable the recurrent connection (i.e. do not maintain a state). For experimenting only, the results will be flickering a lot.
- ``--hot_pixels_file`` (default: None): Path to a file specifying the locations of hot pixels (such a file can be obtained with [this tool](https://github.com/cedric-scheerlinck/dvs_tools/tree/master/dvs_hot_pixel_filter) for example). These pixels will be ignored (i.e. zeroed out in the event tensors).

## Example datasets

We provide a list of example (publicly available) event datasets to get started with E2VID.

- [High Speed (gun shooting!) and HDR Dataset](http://rpg.ifi.uzh.ch/E2VID.html)
- [Event Camera Dataset](http://rpg.ifi.uzh.ch/data/E2VID/datasets/ECD_IJRR17/)
- [Bardow et al., CVPR'16](http://rpg.ifi.uzh.ch/data/E2VID/datasets/SOFIE_CVPR16/)
- [Scherlinck et al., ACCV'18](http://rpg.ifi.uzh.ch/data/E2VID/datasets/HF_ACCV18/)
- [Color event sequences from the CED dataset Scheerlinck et al., CVPR'18](http://rpg.ifi.uzh.ch/data/E2VID/datasets/CED_CVPRW19/)

## Working with ROS

Because PyTorch recommends Python 3 and ROS is only compatible with Python2, it is not straightforward to have the PyTorch reconstruction code and ROS code running in the same environment.
To make things easy, the reconstruction code we provide has no dependency on ROS, and simply read events from a text file or ZIP file.
We provide convenience functions to convert ROS bags (a popular format for event datasets) into event text files.
In addition, we also provide scripts to convert a folder containing image reconstructions back to a rosbag (or to append image reconstructions to an existing rosbag).

**Note**: it is **not** necessary to have a sourced conda environment to run the following scripts. However, [ROS](https://www.ros.org/) needs to be installed and sourced.

### rosbag -> events.txt

To extract the events from a rosbag to a zip file containing the event data:

```bash
python scripts/extract_events_from_rosbag.py /path/to/rosbag.bag \
  --output_folder=/path/to/output/folder \
  --event_topic=/dvs/events
```

### image reconstruction folder -> rosbag

```bash
python scripts/image_folder_to_rosbag.py \
  --datasets dynamic_6dof \
  --image_folder /path/to/image/folder \
  --output_folder /path/to/output_folder \
  --image_topic /dvs/image_reconstructed
```

### Append image_reconstruction_folder to an existing rosbag

```bash
cd scripts
python embed_reconstructed_images_in_rosbag.py \
  --rosbag_folder /path/to/rosbag/folder \
  --datasets dynamic_6dof \
  --image_folder /path/to/image/folder \
  --output_folder /path/to/output_folder \
  --image_topic /dvs/image_reconstructed
```

### Generating a video reconstruction (with a fixed framerate)

It can be convenient to convert an image folder to a video with a fixed framerate (for example for use in a video editing tool).
You can proceed as follows:

```bash
export FRAMERATE=30
python resample_reconstructions.py -i /path/to/input_folder -o /tmp/resampled -r $FRAMERATE
ffmpeg -framerate $FRAMERATE -i /tmp/resampled/frame_%010d.png video_"$FRAMERATE"Hz.mp4
```

## Acknowledgements

This code borrows from the following open source projects, whom we would like to thank:

- [pytorch-template](https://github.com/victoresque/pytorch-template)
