import os
import argparse
from app.prepocessing.image_ocr import create_text_from_images
from app.prepocessing.preproc_data import create_dataframe_from_data
from app.model.learn_lmv3 import learn_model
from app.prepocessing.rotate_image import rotate_images_in_dataframe

def start_learning(dir_path, new_images_dir_path):
    current_directory = os.getcwd()
    files_df = create_dataframe_from_data(dir_path)
    files_df.to_csv(f'{current_directory}/files_df.csv', sep=';')
    files_df = rotate_images_in_dataframe(files_df, new_images_dir_path)
    files_df.to_csv(f'{current_directory}/files_df.csv', sep=';')
    files_df = create_text_from_images(files_df, new_images_dir_path)
    files_df.to_csv(f'{current_directory}/files_df.csv', sep=';')
    learn_model(files_df, accelerator='cpu', devices=[0], batch_size=8)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning model")
    parser.add_argument("dir_path", help="directory with data")
    parser.add_argument("new_images_dir_path", help="directory for prep images and text")

    args = parser.parse_args()
    start_learning(args.dir_path, args.new_images_dir_path)
