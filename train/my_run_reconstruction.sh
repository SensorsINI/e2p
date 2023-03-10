#python run_reconstruction.py -c pretrained/firenet_1000.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 10 -i /home/mhy/firenet-pdavis/data/gun_bullet_gnome.zip -o output --dataset_name gun_bullet_gnome

dir="/home/mhy/firenet-pdavis/data/v2e/txt/"
name="subject01_group1_time1"
i="${name}_i"
i0="${name}_i0"
i45="${name}_i45"
i90="${name}_i90"
i135="${name}_i135"
i_path="${dir}${i}.txt"
i0_path="${dir}${i0}.txt"
i45_path="${dir}${i45}.txt"
i90_path="${dir}${i90}.txt"
i135_path="${dir}${i135}.txt"

# FireNet
python run_reconstruction.py -c pretrained/firenet_1000.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 33.33 -i $i_path -o output/FireNet --dataset_name $i
#python run_reconstruction.py -c pretrained/firenet_1000.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 33.33 -i $i0_path -o output/FireNet --dataset_name $i0
#python run_reconstruction.py -c pretrained/firenet_1000.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 33.33 -i $i45_path -o output/FireNet --dataset_name $i45
#python run_reconstruction.py -c pretrained/firenet_1000.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 33.33 -i $i90_path -o output/FireNet --dataset_name $i90
#python run_reconstruction.py -c pretrained/firenet_1000.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 33.33 -i $i135_path -o output/FireNet --dataset_name $i135

# E2VID
#python run_reconstruction.py -c pretrained/E2VID_lightweight.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 100 -i /media/iccd/disk/event_dataset/pdvs/txt/Davis346B-2022-02-07T11-04-27+0100-00000000-0-demo.txt -o output/E2VID --dataset_name demo
#python run_reconstruction.py -c pretrained/E2VID_lightweight.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 100 -i /media/iccd/disk/event_dataset/pdvs/txt/Davis346B-2022-02-07T11-04-27+0100-00000000-0-demo_i0.txt -o output/E2VID --dataset_name demo_i0
#python run_reconstruction.py -c pretrained/E2VID_lightweight.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 100 -i /media/iccd/disk/event_dataset/pdvs/txt/Davis346B-2022-02-07T11-04-27+0100-00000000-0-demo_i45.txt -o output/E2VID --dataset_name demo_i45
#python run_reconstruction.py -c pretrained/E2VID_lightweight.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 100 -i /media/iccd/disk/event_dataset/pdvs/txt/Davis346B-2022-02-07T11-04-27+0100-00000000-0-demo_i90.txt -o output/E2VID --dataset_name demo_i90
#python run_reconstruction.py -c pretrained/E2VID_lightweight.pth.tar --display --show_events --fixed_duration --no-normalize --window_duration 100 -i /media/iccd/disk/event_dataset/pdvs/txt/Davis346B-2022-02-07T11-04-27+0100-00000000-0-demo_i135.txt -o output/E2VID --dataset_name demo_i135