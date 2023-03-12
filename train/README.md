# Deep Polarization Reconstruction with PDAVIS Events
#### _INI_ 
#### _UZH / ETH Zurich_

### Local Experimental Environment
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

### 3090/V100 Server Environment Configuration
- Python 3.8.13
- PyTorch 1.12.0a0+8a1a93a
- `sudo docker pull nvcr.io/nvidia/pytorch:22.05-py3`
- `pip install -r requirements.txt`
- `apt-get update`
- `apt-get install libgl1`

### Using E2PD
Download E2PD from [ [Onedrive]() ] or [ [Baidu Disk](), fetch code: e2pd ] and unzip it to ./data/ folder

- txt file location
  - ./data/E2PD/train.txt
  - ./data/E2PD/test.txt

- hdf5 file location
  - ./data/E2PD/synthetic/xxxxxx_iad.h5
  - ./data/E2PD/real/real-xx.h5

### Recording Data using PDAVIS
- Using PDAVIS to record aedat2 files and save them at ./data/new/real/
- extract PDAVIS frames
  - `python my_extract_aps.py`
- manually extract PDAVIS events from jaer
  - (** Need to open the output xxx-events.txt and delete the first _n_ rows as they are duplicated **)
  - (** To find how many rows are duplicated, we can check the timestamp of the first frame in xxx-timecode.txt and then use this timestamp to find where the timestamp discontinuity appears in xxx-events.txt **)
- convert avi to frames
  - `python my_video2frame.py`
- merge events, frame, frame_idx, frame_ts, intensity, aolp, dolp into one hdf5 file for the use of DNN
  - `python my_merge_h5.py`
- create train/test list txt
  - `python my_create_real_list.py`

### Train
- `sh my_train.sh`

### Test
- test e2p model
  - `python my_test.py`
- test firenet model
  - `python my_test_firenet.py`
- test e2vid model
  - `python my_test_e2vid.py`
- align e2p and firenet for visual comparison
  - `python align3_images.py`
- split the visual comparison results
  - `python split_visual3.py`
- output location
  - ./output_e2p
  - ./output_real

### Evaluation
- obtain the ground truth from h5 file
  - `python utils/h5gt2iad.py`
- evaluation
  - `python eval.py`
- output location
  - ./txt_eval/e2p.txt

### Dataset Visualization
- `python utils/h5gt2iad.py`
- `python visualize_pvideo.py`
