# /bin/bash
source activate midiffusion
export CUDA_VISIBLE_DEVICES=2

#Curios->IXI
python Sampling.py --use_unet --config ./configs/IXI.yml --seed 1234 --comment "" --exp CuRIOUST1_IXIPD/ --train_path_a /CuRIOUS/imagesTr_slices/test/T1 --train_path_b /CuRIOUS/imagesTr_slices/test/T2 --verbose info --image_folder images_level500  --doc doc --sample --use_pretrained --fid --sample_step 3 --t 1000