"""
main function for ./openslied_2_tfrecord.py

Usage:
python3 digipath_mltk.py -run_directory ./ -run_file run_parameters_file.yml

"""

def image_2_tfrecord(run_parameters):
    from openslide_2_tfrecord import run_imfile_to_tfrecord
    run_imfile_to_tfrecord(run_parameters)

def tfrecord_2_masked_thumb(run_parameters):
    from openslide_2_tfrecord import write_tfrecord_masked_thumbnail
    write_tfrecord_masked_thumbnail(run_parameters)

SELECT = {"image_2_tfrecord": image_2_tfrecord,
          "tfrecord_2_masked_thumb": tfrecord_2_masked_thumb}

def main():
    import sys
    from openslide_2_tfrecord import get_run_directory_and_run_file, get_run_parameters
    run_directory, run_file = get_run_directory_and_run_file(sys.argv[1:])
    run_parameters = get_run_parameters(run_directory, run_file)

    SELECT[run_parameters['method']](run_parameters)

if __name__ == "__main__":
    main()
