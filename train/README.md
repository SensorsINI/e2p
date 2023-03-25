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

### Downloads
- E2PD
  - [ [Onedrive](https://1drv.ms/u/s!AjYBkUJACkBLm1tWpU-N0lmKv36x?e=oIEajs) ] or [ [Baidu Disk](https://pan.baidu.com/s/1JSZqcbFk_52Xd_Ex_bicmQ?pwd=e2pd), fetch code: e2pd ]
- Models
  - FireNet [ [Onedrive](https://1drv.ms/u/s!AjYBkUJACkBLm1-Cnb7Fh_-xZsNR?e=fEdTHJ) ] or [ [Baidu Disk](https://pan.baidu.com/s/1DVEwX8Ax9OSO-_ZpDCEjSw?pwd=ckpt), fetch code: ckpt ]
  - E2P [ [Onedrive](https://1drv.ms/u/s!AjYBkUJACkBLm15aBBmch2qjRL9U?e=Dc9boq) ] or [ [Baidu Disk](https://pan.baidu.com/s/1zfAI2HViEA7ek_8JqRVSQQ?pwd=ckpt), fetch code: ckpt ]
- Results
  - Synthetic [ [Onedrive](https://1drv.ms/u/s!AjYBkUJACkBLm2CEZ8vgOVBtFWbP?e=Scss5J) ] or [ [Baidu Disk](https://pan.baidu.com/s/1HAWaFTgMs2t-6hmZWJwBSA?pwd=outs), fetch code: outs ]
  - Real [ [Onedrive](https://1drv.ms/u/s!AjYBkUJACkBLm10gPsk9ao3Fqh0s?e=93jX0S) ] or [ [Baidu Disk](https://pan.baidu.com/s/1IIdLL7ZNR-5Ne1FW8M0LwQ?pwd=outr), fetch code: outr ]

### One Command to Play Back One HDF5 file
- `python infer.py xxx.h5`

xxx.h5 is the test file in `infer` folder where the corresponding outputs will be saved.

### Using E2PD
Unzip E2PD.zip to ./data/ folder

- txt file location
  - ./data/E2PD/train.txt
  - ./data/E2PD/test.txt

- hdf5 file location
  - ./data/E2PD/synthetic/xxxxxx_iad.h5
  - ./data/E2PD/real/real-xx.h5

### Recording Data using PDAVIS
- Using PDAVIS with jAER to record aedat2 files and save them at ./data/new/real/
- extract PDAVIS frames with [my_extract_events.py](my_extract_events.py)
  - `python my_extract_aps.py`
- manually extract PDAVIS events from jaer using EventFilter DavisTextOutputWriter; see https://docs.google.com/document/d/1fb7VA8tdoxuYqZfrPfT46_wiT1isQZwTHgX8O22dJ0Q/#bookmark=id.9xld1vw3ttt0
  -  Set the options "rewindBeforeRecording" and "dvsEvents" and "closeOnRewind"
  - Click the "StartRecordingAndSaveAs" button and choose output file with .txt extension; you can select the ./data/new/real/...timecode.txt file and delete the timecode part.
    - (** Because of bug in DavisTextOutputWriter, it is essential to open the output xxx-events.txt and delete the first _n_ rows as they are duplicated. __n__ will be large number of perhaps 10k lines! **. If you don't do this you will get bogus timestamps that mess up the synchronization between events and frames and the training will have meaningless GT target.)
    - (** To find how many rows are duplicated, we can check the timestamp of the first frame in xxx-timecode.txt and then use this timestamp to find where the timestamp discontinuity appears in xxx-events.txt **)
    - ![write-events-txt-file.png](train/media/write-events-txt-file.png)
- convert avi to frames with [my_video2frame.py](my_video2frame.py)
  - `python my_video2frame.py`
- merge events, frame, frame_idx, frame_ts, intensity, aolp, dolp into one hdf5 file for the use of DNN using [my_merge_h5.py](my_merge_h5.py)
  - `python my_merge_h5.py`
- create train/test list txt with [my_create_real_list.py](my_create_real_list.py)
  - `python my_create_real_list.py`

Now you should have the new list of training files to add to existing list

### Training input hdf5 files
The HDF5 (.h5) files should have the following contents

![h5_file_contents.png](h5_file_contents.png)

### Train
- `sh my_train.sh`

Steps are illustrated below. 

1. The main config is [e2p.json](e2p.json)
2.  It points to  [train.txt](data%2FE2PD%2Ftrain.txt) which lists the training input files.
3. These files must be at the locations listed in train.txt

![train_steps.png](train/media/train_steps.png)

#### Using tensorboard to visualize training

![tensorboard_output.png](media%2Ftensorboard_output.png)

* Last two rows are predictions and ground truth.
* Left is intensity, middle is aolp and right is dolp,
* For dolp yellow means high degree. black/dark red is low degree.
* For aolp, 0 means -pi/2, 1 means pi/2, 0 angle is horizontally polarized light.

![aolp mapping.png](..%2Fmedia%2Faolp%20mapping.png)

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
  - ./output_synthetic/e2p
  - ./output_real/e2p

### Evaluation
- obtain the ground truth from h5 file
  - `python utils/h5gt2iad.py`
- evaluation
  - `python eval.py`
- output location
  - ./txt_eval/e2p.txt

### Dataset Visualization
Process a list of files set at start of [h5gt2iad.py](utils%2Fh5gt2iad.py) in line: 
`test_list_txt = 'train/data/movingcam/test_real2.txt'`
- `python utils/h5gt2iad.py` [h5gt2iad.py](utils%2Fh5gt2iad.py)

Turn these into movies with [visualize_pvideo.py](visualize_pvideo.py). Edit the line near start to point to the correct folder: 
- `python visualize_pvideo.py` 
