# Установка зависимостей

Создать виртуальное окружение через консоль находясь в папке проекта используя python3.8.10 или pythob3.8.13. 

    python3.8.10 -m venv .venv

Активировать виртуальное окружение 

    . .venv/bin/activate

Установить зависимости

    pip install -r requirements.txt

Установить tesseract для распознавания текста 

    sudo apt install tesseract-ocr
    sudo apt install libtesseract-dev

# Запуск обучения

Через консоль, находясь в корне проекта запустить команду

    python main.py <path_to_data> <path_to_prep_data>

где <path_to_data> - путь до папки в которой хранятся изобрежения. Структура должна быть такой:

```
dir_with_image
├── category1
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   ├── ...
│   ├── imgN.jpg
├── category2
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   ├── ...
│   ├── imgN.jpg
...
├── categoryN
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   ├── ...
│   ├── imgN.jpg
```

<path_to_prep_data> - путь до папки, куда будут сохраняться извлеченные данные, это текст с изображений и перевернутые изображения. <b>Не используйте ту же папку, где лежат исходные изображения, укажите другую папку</b>

# Результаты

После завершения обучения в папке documents_classification/logs/lightning_logs повятся логи. Вложенные папки version_N означают номер запуска модуля, если он будет запускаться не один раз. 

В папке checkpoints будут храниться веса модели. 

В файле metrics.csv логи изменения метрик во время обучения.