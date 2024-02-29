# /bin/bash
source activate midifusion
export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py --use_unet --config ./configs/CuRIOUS_FLAIR_T1.yml --gpu_id 0  --seed 1234 --comment "" --verbose info --image_folder images  --exp CuRIOUST1 --doc doc --train_path_a /CuRIOUS/imagesTr_slices/train/T1 --train_path_b /CuRIOUS/imagesTr_slices/train/T1

python main.py --use_unet --config ./configs/CuRIOUS_T1_FLAIR.yml --gpu_id 1 --seed 1234 --comment "" --verbose info --image_folder images  --exp CuRIOUST2 --doc doc --train_path_a /CuRIOUS/imagesTr_slices/train/T2 --train_path_b /CuRIOUS/imagesTr_slices/train/T2 &

python main.py --use_unet --config ./configs/Mri_Pelvis_CT_MR.yml --gpu_id 2  --seed 1234 --comment "" --verbose info --image_folder images  --exp GoldAtlasMalePelvisMR --doc doc --train_path_a /GoldAtlasMalePelvis/MRslice/train --train_path_b /GoldAtlasMalePelvis/MRslice/train &

python main.py --use_unet --config ./configs/Mri_Pelvis_MR_CT.yml --gpu_id 3  --seed 1234 --comment "" --verbose info --image_folder images  --exp GoldAtlasMalePelvisCT --doc doc --train_path_a /GoldAtlasMalePelvis/CTslice/train --train_path_b /GoldAtlasMalePelvis/CTslice/train

wait

python main.py --use_unet --config ./configs/IXI.yml --gpu_id 0 --seed 1234 --comment "" --verbose info --image_folder images  --exp IXIPD --doc doc --train_path_a /IXI/IXI-PD_slices/train --train_path_b /IXI/IXI-PD_slices/train

python main.py --use_unet --config ./configs/IXI.yml --gpu_id 1 --seed 1234 --comment "" --verbose info --image_folder images  --exp IXIT1 --doc doc --train_path_a /IXI/IXI-T1_slices/train --train_path_b /IXI/IXI-T1_slices/train

