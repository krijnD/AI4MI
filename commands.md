# Some usefull commands:

# activate environment:
source ai4mi/bin/activate

# ## Data Preprocessing

### 1. Slice Segthor Train/affine/elastic/noise Data
make data/slice_segthor_train
make data/slice_segthor_affine

### Adding multiple datasets to ENET main for training
$ python ENet/main.py --datasets SEGTHOR SEGTHOR_affine --mode full --epoch 2 --dest results/toy2/combined_debug --gpu --debug


