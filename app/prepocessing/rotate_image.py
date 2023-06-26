import os
import cv2
import imutils
import pytesseract
from pytesseract import Output, TesseractError

import pandas as pd

from preproc_data import create_dataframe_from_data

def rotate_image(image_path, image_name, new_image_path, lang='osd'):
    try:
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pytesseract.image_to_osd(rgb, output_type=Output.DICT, lang=lang)

        if results['rotate'] != 0:
            print(f"[INFO] {image_name} detected orientation: {results['orientation']}")
            rotated = imutils.rotate_bound(image, angle=results["rotate"])
            
            new_dir = image_path.split('/')[-2]
            result_path = f'{new_image_path}/{new_dir}'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            
            result_path += f'/{image_name}'

            cv2.imwrite(result_path, rotated)  # Сохранение изображения
            return result_path
    except TesseractError as te:
        # print(f'[ERROR] {te}')
        if lang == 'osd':
            return rotate_image(image_path, image_name, new_image_path, lang='rus')

    return None


def rotate_images_in_dataframe(files_df, new_images_dir_path):
    print('Проверка изображений на правильную ориентацию')
    files_df_len = len(files_df)
    print(f'Всего изображение в наборе данных: {files_df_len}')
    
    new_images_dir_path = os.path.join(new_images_dir_path, 'rotated_images')

    if not os.path.exists(new_images_dir_path):
        os.makedirs(new_images_dir_path)
        
    for index, row in files_df.iterrows():
        # if (index % 50 == 0):
        print(f'Проверенна ротация у {index}/{files_df_len} изображений')
            
        rotated_image_path = rotate_image(row['image_path'], row['image_name'], new_images_dir_path)
        if rotated_image_path is not None:
            files_df.at[index, 'image_path'] = rotated_image_path
    
    return files_df


if __name__ == '__main__':
    dir_path = '/home/tsamigulin/data/test_documents'
    new_images_dir_path = '/home/tsamigulin/data/test_prep_documents'
    files_df = create_dataframe_from_data(dir_path)
    files_df = rotate_images_in_dataframe(files_df, new_images_dir_path)
    print(files_df.loc[50:])