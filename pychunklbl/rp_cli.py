"""
./toolkit.py -- main()

parameterized calling of the toolkit with the same method names as cli.py
using yml files in data/run_files

Usage:
python3 rp_cli.py -run_directory ./ -run_file run_parameters_file.yml

"""
from pychunklbl import toolkit

def image_2_tfrecord(run_parameters):
    toolkit.run_imfile_to_tfrecord(run_parameters)

def tfrecord_2_masked_thumb(run_parameters):
    toolkit.write_tfrecord_marked_thumbnail_image(run_parameters)

def wsi_2_patches_dir(run_parameters):
    toolkit.image_file_to_patches_directory_for_image_level(run_parameters)

def write_mask_preview(run_parameters):
    toolkit.write_mask_preview_set(run_parameters)

def registration_functions(run_parameters):
    toolkit.run_registration_pairs(run_parameters)

def annotation_functions(run_parameters):
    toolkit.run_annotated_patches(run_parameters)

SELECT = {"wsi_to_patches": image_2_tfrecord,
          "tfrecord_2_masked_thumb": tfrecord_2_masked_thumb,
          'wsi_2_patches_dir': wsi_2_patches_dir,
          'wrte_mask_preview_set': write_mask_preview,
          'registration_to_dir': registration_functions,
          'registration_to_tfrecord': registration_functions,
          'annotations_to_dir': annotation_functions,
          'annotations_to_tfrecord': annotation_functions}

def main():
    import sys

    run_directory, run_file = toolkit.get_run_directory_and_run_file(sys.argv[1:])
    run_parameters = toolkit.get_run_parameters(run_directory, run_file)

    SELECT[run_parameters['method']](run_parameters)

if __name__ == "__main__":
    main()
