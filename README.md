# Liver and liver tumor segmengtation using deep learning

## Dataset
Tập dữ liệu LiTS2017 là một tập dữ liệu mở được sử dụng cho cuộc thi "Liver Tumor Segmentation Challenge", có thể truy cập thông qua LiTS Challenge website: https://competitions.codalab.org/competitions/17094. Tập dữ liệu này bao gồm 201 3D volumes CT ổ bụng từ 100 bệnh nhân, trong đó có 194 volumes có chứa khối u. Các volumes CT có kích thước ảnh 512 x 512, và có số lượng lát cắt khác nhau. Trong số 200 3D volumes CT, có 131 volumes có nhãn, dùng để huấn luyện và 70 volumes không có nhãn, dùng để kiểm thử. Chỉ có những người tổ chức cuộc thi mới có thể truy cập được vào nhãn của các volumes kiểm thử.
## Thông tin tổng quan dataset
<div align=center><img src="https://github.com/HuydoanRH/TPUnet_LiTS_LTS/blob/main/Image/LiTS_Dataset.png"alt="LiTS Dataset Overview"></div> 

## Một số kết quả sau khi xử lý dataset
<div align=center><img src="https://github.com/HuydoanRH/TPUnet_LiTS_LTS/blob/main/Image/livertumorVisual.png"alt="LiTS Dataset result"></div>  
  
## Usage
### 1) LITS2017 dataset preprocessing:
1. Tải dataset tại link: [Liver Tumor Segmentation Challenge.](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)
2. Giải nén => merge các part vô cùng 1 folder. Lưu volume data và segmentation label data vào các folder khác nhau. Ví dụ như dưới:
```
raw_dataset:
    ├── test  # 20 samples(27~46) 
    │   ├── ct
    │   │   ├── volume-27.nii
    │   │   ├── volume-28.nii
    |   |   |—— ...
    │   └── label
    │       ├── segmentation-27.nii
    │       ├── segmentation-28.nii
    |       |—— ...
    │       
    ├── train # 111 samples(0\~26 and 47\~131)
    │   ├── ct
    │   │   ├── volume-0.nii
    │   │   ├── volume-1.nii
    |   |   |—— ...
    │   └── label
    │       ├── segmentation-0.nii
    │       ├── segmentation-1.nii
    |       |—— ...
```
3. Change root path của volume data và segmentation label ở file: `./Dataset/config.py`
```
    parser.add_argument('--dataset_fixed_path',default = './Dataset/fixed_dataset/',help='fixed trainset root path')
    parser.add_argument('--dataset_raw_path',default = './Dataset/raw_dataset/train/',help='raw dataset path')
```
4. Run `python ./Dataset/dataset_preparation.py`. Nếu không raise lỗi gì thì kết quả được thể hiện ở `./Dataset/fixed_dataset/`
```
│—— train_path_list.txt
│—— val_path_list.txt
│
|—— ct
│       volume-0_slice_01.npy
│       volume-0_slice_02.npy
│       ...
└─ label
        segmentation-0_slice_01.npy
        segmentation-0_slice_02.npy
        ...
```  
### 2) Training TPUNet:

### 3) Evaluate TPUNet