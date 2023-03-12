# Deep Polarization Reconstruction with PDAVIS Events
#### _INI_ 
#### _UZH / ETH Zurich_

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
1. txt file location
- ./data/E2PD/train_mix2.txt
- ./data/E2PD/test_mix2.txt

2. synthetic/real hdf5 file location
- ./data/E2PD/synthetic/xxxxxx_iad.h5
- ./data/E2PD/real/real-xx.h5

3. synthetic/real output location
- ./output/e2p
- ./output_real
- ./txt_eval/e2p.txt

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
- test firenet on real data
  - `python my_test_firenet.py`
- obtain the pdavis frame from h5 file
  - `python utils/h5gt2iad.py`

### Dataset Visualization
- `python utils/h5gt2iad.py`
- `python visualize_pvideo.py`
