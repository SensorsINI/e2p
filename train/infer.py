"""
 @Time    : 25.03.23 16:16
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com

 @Project : e2p
 @File    : infer.py
 @Function:

"""
import os
import argparse

method = 'e2p'

ckpt_path = '../{}.pth'.format(method)


def main(args):

    input_path = os.path.join(args.input_dir, args.input_file)
    name = args.input_file.split('.')[0]
    output_dir = os.path.join(args.output_dir, name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.if_real:
        h = 260
        w = 346
    else:
        h = 480
        w = 640

    call_with_args = 'python my_inference.py --checkpoint_path {} --height {} --width {} --device 0 --events_file_path {} --output_folder {}/{}'.format(
        h, w, ckpt_path, input_path, output_dir, name)

    print(call_with_args)

    ret=os.system(call_with_args)

    if ret==0:
        print('Succeeded')
    else:
        print('failed, check arguments')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--input_file', required=True, type=str,
                        help='input file name')
    parser.add_argument('--input_dir', type=str, default='./infer_input/',
                        help='path to input hdf5 file (default: None)')
    parser.add_argument('--output_dir', type=str, default='./infer_output/',
                        help='path to inference outputs')
    parser.add_argument('--with_gt', action='store_true',
                        help='If true, save the GT frames from the input hdf5 file')
    parser.add_argument('--if_real', action='store_true',
                        help='True indicates the input file comes from real PDAVIS, False means simulated or Sony input')

    args = parser.parse_args()

    main(args)
