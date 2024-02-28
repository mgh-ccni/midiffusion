source activate tensorflow
export CUDA_VISIBLE_DEVICES=1,

python main.py --use_sdedit --config ./configs/Mri_SDEDIT.yml --mean 0.4822 --var 0.1470 --seed 1234 --comment "" --verbose info --image_folder images  --exp SDEDITCuRIOUST1 --doc doc --train_path_a /CuRIOUS/imagesTr_slices/train/T1 --train_path_b /CuRIOUS/imagesTr_slices/train/T1
