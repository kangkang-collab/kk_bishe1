## stage1
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage1.py \
   2 \
   --deterministic

## stage2
#export MMCV_DISABLE_YAPF=1
# export CUDA_VISIBLE_DEVICES=1

# bash ./tools/dist_train.sh \
#    projects/configs/sparsedrive_small_stage2.py \
#    2 \
#    --deterministic