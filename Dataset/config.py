import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Preprocess parameters
parser.add_argument('--upper', type=int, default=200, help='')
parser.add_argument('--lower', type=int, default=-200, help='')
parser.add_argument('--expand_slice', type=int, default=20, help='')
parser.add_argument('--min_slices', type=int, default=48, help='')
parser.add_argument('--size', type=int, default=48, help='')
parser.add_argument('--fillters', type=int, default=20, help='number of tumor pixels')
parser.add_argument('--valid_rate', type=float, default=0.2, help='')

# data in/out and dataset
parser.add_argument('--dataset_fixed_path',default = './Dataset/fixed_dataset/',help='fixed trainset root path')
parser.add_argument('--dataset_raw_path',default = './Dataset/raw_dataset/train/',help='raw dataset path')
parser.add_argument('--save',default='TP_Unet',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=16,help='batch size of trainset')

args = parser.parse_args()
