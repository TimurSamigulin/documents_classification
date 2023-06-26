import os
import pandas as pd


def get_files(dir_path, extensions=None):
    # Функция для получения всех файлов с требуемым расширением в директории и всех её поддиректориях
    extensions_empty = False
    if not extensions:
        extensions_empty = True
    result = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # Проверяем, что расширение файла входит в список требуемых расширений
            if extensions_empty or file.endswith(tuple(extensions)):
                result.append((os.path.join(root, file), os.path.basename(root)))
                
    return result


def get_file_name(file_path):
    return os.path.basename(file_path)


def get_file_extension(file_path):
    return os.path.splitext(file_path)[1]

def create_dataframe_from_data(dir_path):
    extensions = ('.jpg', '.png', '.jpeg', '.JPG')
    files = get_files(dir_path, extensions)
    files_df = pd.DataFrame(files, columns=['image_path', 'category'])
    files_df['image_name'] = files_df['image_path'].apply(get_file_name)
    files_df['image_extension'] = files_df['image_path'].apply(get_file_extension)
    
    return files_df


if __name__ == '__main__':
    dir_path = '/home/tsamigulin/data/test_documents'
    files_df = create_dataframe_from_data(dir_path)
    print(files_df)