red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@

data/TOY2_noise:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256 --add_noise
	mv $@_tmp $@



# # Extraction and slicing for Segthor
# data/segthor_train: data/segthor_train.zip
# 	$(info $(yellow)unzip $<$(reset))
# 	sha256sum -c data/segthor_train.sha256
# 	unzip -q $<
# 	rm -f $@/.DS_STORE

# data/SEGTHOR: data/segthor_train
# 	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
# 	rm -rf $@_tmp $@
# 	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir $@_tmp \
# 		--shape 256 256 --retain 10
# 	mv $@_tmp $@

# Extraction of segthor_train (unzipping only)
unzip_segthor_train: data/segthor_train.zip
	$(info $(yellow)Unzipping data/segthor_train.zip$(reset))
	sha256sum -c data/segthor_train.sha256
	unzip -q $< -d data/
	rm -f data/segthor_train/.DS_STORE

# Slicing segthor_train (slicing only)
data/slice_segthor_train: data/segthor_train
	$(info $(green)Slicing data/segthor_train using slice_segthor.py$(reset))
	rm -rf data/SEGTHOR_tmp data/SEGTHOR_train
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir data/SEGTHOR_tmp \
		--shape 256 256 --retain 10 --test_with_labels True
	mv data/SEGTHOR_tmp data/SEGTHOR_train

# Slicing segthor_affine (slicing only)
data/slice_segthor_affine: data/segthor_affine
	$(info $(green)Slicing data/segthor_affine using slice_segthor.py$(reset))
	rm -rf data/SEGTHOR_affine_tmp data/SEGTHOR_affine
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir data/SEGTHOR_affine_tmp \
		--shape 256 256 --retain 10 --test_with_labels True
	mv data/SEGTHOR_affine_tmp data/SEGTHOR_affine

# Slicing segthor_elastic (slicing only)
data/slice_segthor_elastic: data/segthor_elastic
	$(info $(green)Slicing data/segthor_elastic using slice_segthor.py$(reset))
	rm -rf data/SEGTHOR_elastic_tmp data/SEGTHOR_elastic
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir data/SEGTHOR_elastic_tmp \
		--shape 256 256 --retain 10 --test_with_labels True
	mv data/SEGTHOR_elastic_tmp data/SEGTHOR_elastic

# Slicing segthor_noise (slicing only)
data/slice_segthor_noise: data/segthor_noise
	$(info $(green)Slicing data/segthor_noise using slice_segthor.py$(reset))
	rm -rf data/SEGTHOR_noise_tmp data/SEGTHOR_noise
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir data/SEGTHOR_noise_tmp \
		--shape 256 256 --retain 10 --test_with_labels True
	mv data/SEGTHOR_noise_tmp data/SEGTHOR_noise

# # Combine all slices into SEGTHOR_all
# data/slice_segthor_all: data/slice_segthor_train data/slice_segthor_affine data/slice_segthor_elastic data/slice_segthor_noise
# 	$(info $(green)Combining all SEGTHOR slices into SEGTHOR_all$(reset))
# 	rm -rf data/SEGTHOR_all_tmp data/SEGTHOR_all
# 	mkdir data/SEGTHOR_all_tmp
# 	cp -r data/SEGTHOR_train/* data/SEGTHOR_all_tmp
# 	cp -r data/SEGTHOR_affine/* data/SEGTHOR_all_tmp
# 	cp -r data/SEGTHOR_elastic/* data/SEGTHOR_all_tmp
# 	cp -r data/SEGTHOR_noise/* data/SEGTHOR_all_tmp
# 	mv data/SEGTHOR_all_tmp data/SEGTHOR_all


