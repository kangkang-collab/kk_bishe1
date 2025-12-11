bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    /home/itslab/kk/SparseDrive_first/ckpt/sparsedrive_stage2.pth \
    2 \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl