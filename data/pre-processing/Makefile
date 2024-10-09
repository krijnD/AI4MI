# Root path (going up two levels to reach the base AI4MI directory)
ROOT_DIR = ../../

# Slicing segthor_train (slicing only)
data/slice_segthor_train: $(ROOT_DIR)data/segthor_train
	$(info $(green)Slicing $(ROOT_DIR)data/segthor_train using slice_segthor.py$(reset))
	rm -rf $(ROOT_DIR)data/SEGTHOR_tmp $(ROOT_DIR)data/SEGTHOR
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir $(ROOT_DIR)data/SEGTHOR_tmp \
		--shape 256 256 --retain 10 --test_with_labels True
	mv $(ROOT_DIR)data/SEGTHOR_tmp $(ROOT_DIR)data/SEGTHOR

# Slicing segthor_affine (slicing only)
data/slice_segthor_affine: $(ROOT_DIR)data/segthor_affine
	$(info $(green)Slicing $(ROOT_DIR)data/segthor_affine using slice_segthor.py$(reset))
	rm -rf $(ROOT_DIR)data/SEGTHOR_affine_tmp $(ROOT_DIR)data/SEGTHOR_affine
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir $(ROOT_DIR)data/SEGTHOR_affine_tmp \
		--shape 256 256 --retain 10 --test_with_labels True
	mv $(ROOT_DIR)data/SEGTHOR_affine_tmp $(ROOT_DIR)data/SEGTHOR_affine

# Slicing segthor_elastic (slicing only)
data/slice_segthor_elastic: $(ROOT_DIR)data/segthor_elastic
	$(info $(green)Slicing $(ROOT_DIR)data/segthor_elastic using slice_segthor.py$(reset))
	rm -rf $(ROOT_DIR)data/SEGTHOR_elastic_tmp $(ROOT_DIR)data/SEGTHOR_elastic
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir $(ROOT_DIR)data/SEGTHOR_elastic_tmp \
		--shape 256 256 --retain 10 --test_with_labels True
	mv $(ROOT_DIR)data/SEGTHOR_elastic_tmp $(ROOT_DIR)data/SEGTHOR_elastic

# Slicing segthor_noise (slicing only)
data/slice_segthor_noise: $(ROOT_DIR)data/segthor_noise
	$(info $(green)Slicing $(ROOT_DIR)data/segthor_noise using slice_segthor.py$(reset))
	rm -rf $(ROOT_DIR)data/SEGTHOR_noise_tmp $(ROOT_DIR)data/SEGTHOR_noise
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir $(ROOT_DIR)data/SEGTHOR_noise_tmp \
		--shape 256 256 --retain 10 --test_with_labels True
	mv $(ROOT_DIR)data/SEGTHOR_noise_tmp $(ROOT_DIR)data/SEGTHOR_noise
