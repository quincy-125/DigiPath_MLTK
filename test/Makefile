SCRIPT = ../src/python/digipath_mltk.py
RUN_DIR = ../../run_dir/
RUN_FILES = ../data/run_files
RESULTS_DIR = $(RUN_DIR)/results

MKDIR_P = mkdir -p

env_setup: create_run_dir copy_run_files

create_run_dir:
	$(MKDIR_P) $(RESULTS_DIR)

copy_run_files:
	cp $(RUN_FILES)/*.yml $(RUN_DIR)

test_im_2_tfr:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file image_files_to_tfrecord.yml

test_tfr_2_mthumb:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file tfrecord_to_masked_thumb.yml

