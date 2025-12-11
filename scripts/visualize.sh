export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/sparsedrive_small_stage2.py \
	--result-path /home/itslab/kk/SparseDrive/work_dirs/sparsedrive_small_stage2/results_mini.pkl