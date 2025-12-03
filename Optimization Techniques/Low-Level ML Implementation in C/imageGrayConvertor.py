from PIL import Image
import numpy as np
import os
import csv

cats_folder = './Cats'
dogs_folder = './Dogs'
gray_cats_folder = './Gray_Cats'
gray_dogs_folder = './Gray_Dogs'
output_file = './imageData.csv'

MAX_CATS = 1000
MAX_DOGS = 1000

os.makedirs(gray_cats_folder, exist_ok=True)
os.makedirs(gray_dogs_folder, exist_ok=True)

output_data = []

cats_processed = 0
dogs_processed = 0

print("=" * 50)
print("Görüntü İşleme Başladı")
print("=" * 50)

for input_folder, gray_folder, label, max_count, animal_name in [
    (cats_folder, gray_cats_folder, 1, MAX_CATS, "Kedi"),
    (dogs_folder, gray_dogs_folder, -1, MAX_DOGS, "Köpek")
]:
    print(f"\n{animal_name} resimleri işleniyor...")

    image_files = [f for f in os.listdir(input_folder)
                   if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]


    processed_count = 0

    for file_name in image_files[:max_count]:
        try:
            image_path = os.path.join(input_folder, file_name)
            with Image.open(image_path) as img:
                resized_image = img.resize((50, 50))
                gray_image = resized_image.convert('L')

                # İsteğe bağlı: Gri resimleri kaydet
                # gray_image.save(os.path.join(gray_folder, file_name))

                image_array = np.array(gray_image)
                normalized_array = image_array / 255.0
                flattened_array = normalized_array.flatten()
                flattened_array = np.append(flattened_array, label)
                output_data.append(flattened_array)

                processed_count += 1

        except Exception as e:
            print(f" ERROR: {e}")
            continue

    if label == 1:
        cats_processed = processed_count
    else:
        dogs_processed = processed_count
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output_data)

print("=" * 50)
print(f"\nVeriler '{output_file}' dosyasina kaydedildi.")
print(f"Gri görüntüler: '{gray_cats_folder}' ve '{gray_dogs_folder}'")