# Original Model
#python inference.py --checkpoint_path /home/mhy/firenet-pdavis/pretrained/firenet_1000.pth.tar --device 0 --events_file_path /home/mhy/firenet-pdavis/data/test/subject12_group3_time4.h5 --output_folder /home/mhy/firenet-pdavis/output/original_firenet

# Retrained Model
#python inference.py --checkpoint_path /home/mhy/firenet-pdavis/ckpt/models/mhy2/0410_161126/model_best.pth --height 256 --width 306 --device 0 --events_file_path /home/mhy/firenet-pdavis/data/test/subject09_group2_time1.h5 --output_folder /home/mhy/firenet-pdavis/output/mhy2 --update
#python inference.py --checkpoint_path /home/mhy/firenet-pdavis/ckpt/models/mhy3/0408_173504/model_best.pth --height 256 --width 306 --device 0 --events_file_path /home/mhy/firenet-pdavis/data/test/subject09_group2_time1.h5 --output_folder /home/mhy/firenet-pdavis/output/mhy3 --update

# Retrained P Model
python inference.py --checkpoint_path /home/mhy/firenet-pdavis/ckpt/models/movingcam_firenet_p/0522_185416/model_best.pth --height 480 --width 640 --device 0 --events_file_path /home/mhy/firenet-pdavis/data/test_p/subject09_group2_time1_pf.h5 --output_folder /home/mhy/firenet-pdavis/output/mhy5 --update