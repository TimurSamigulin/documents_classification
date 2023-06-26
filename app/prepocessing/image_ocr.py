import json
import os

import easyocr
import pandas as pd

# from preproc_data import create_dataframe_from_data

def create_bounding_box(bbox_data):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)
    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))
    
    return [left, top, right, bottom]


def create_text_from_images(files_df, new_images_dir_path):
    reader = easyocr.Reader(["ru"], gpu=True)
    files_df_len = len(files_df)
    bad_index = []
    
    ocr_dir = os.path.join(new_images_dir_path, 'ocr_docs')
    if not os.path.exists(ocr_dir):
        os.makedirs(ocr_dir)
    
    for index, row in files_df.iterrows():
        print(f'Обработанно OCR изображений - {index}/{files_df_len}')
        
        try:
            ocr_result = reader.readtext(str(row['image_path']))
        except Exception as e:
            print(e)
            bad_index.append(str(row['image_path']))
            
        ocr_page = []
        for bbox, word, confidence in ocr_result:
            ocr_page.append({
                "word": word, "bounding_box": create_bounding_box(bbox)
            })
        
        new_dir = row['image_path'].split('/')[-2]
        result_path = f'{ocr_dir}/{new_dir}'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        result_path += f"/{row['image_name'].split('/')[0]}.json"
        # print(result_path)
        with open(result_path, "w") as f:
            json.dump(ocr_page, f)
            
        files_df.at[index, 'ocr_path'] = result_path
        
    return files_df


# if __name__ == '__main__':
#     dir_path = '/home/tsamigulin/data/test_documents'
#     new_images_dir_path = '/home/tsamigulin/data/test_prep_documents'
#     files_df = create_dataframe_from_data(dir_path)
#     files_df = create_text_from_images(files_df, new_images_dir_path)
#     print(files_df.loc[50:])