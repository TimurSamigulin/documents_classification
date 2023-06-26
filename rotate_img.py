import os
import argparse
from app.prepocessing.preproc_data import create_dataframe_from_data
from app.prepocessing.rotate_image import rotate_images_in_dataframe

def rotate_images(dir_path, new_images_dir_path):
    current_directory = os.getcwd()
    files_df = create_dataframe_from_data(dir_path)
    files_df.to_csv(f'{current_directory}/rotate_image_df.csv', sep=';')
    files_df = rotate_images_in_dataframe(files_df, new_images_dir_path)
    files_df.to_csv(f'{current_directory}/rotate_image_df.csv', sep=';')
    print('Ротация изображений завершена!')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning model")
    parser.add_argument("dir_path", help="directory with data")
    parser.add_argument("new_images_dir_path", help="directory for prep images and text")

    args = parser.parse_args()
    rotate_images(args.dir_path, args.new_images_dir_path)
