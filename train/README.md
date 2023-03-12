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
