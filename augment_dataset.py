import os
import re
import random
from PIL import Image, ImageEnhance, ImageOps

DATASET_DIR = r"c:\Users\HP\Desktop\yazlab3\dataset"
# Her bir orijinal resimden kaç adet farklı varyasyon üretileceği
AUGMENTATIONS_PER_IMAGE = 3

def get_max_index(folder_path, prefix):
    max_index = -1
    pattern = re.compile(rf"^{prefix}_PHOTO_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
    return max_index if max_index >= 0 else 0

def augment_image(image):
    """Resme rastgele çevirme, döndürme veya parlaklık işlemi uygular."""
    aug_type = random.choice(['flip', 'rotate', 'brightness', 'color', 'zoom'])
    
    if aug_type == 'flip':
        # Yatay eksende ters çevir (ayna görüntüsü)
        return ImageOps.mirror(image)
    
    elif aug_type == 'rotate':
        # -15 ile 15 derece arası rastgele döndür
        angle = random.uniform(-15, 15)
        # Siyah kenarları önlemek için resmi biraz büyütüp kırpabiliriz ama basit rotate iş görür
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    
    elif aug_type == 'brightness':
        # Parlaklığı %30 azalt veya %30 artır
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
        
    elif aug_type == 'color':
        # Renk doygunluğunu değiştir
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(0.5, 1.5)
        return enhancer.enhance(factor)
        
    elif aug_type == 'zoom':
        # %10-20 arası yakınlaştırma (Crop ve Resize)
        width, height = image.size
        zoom_factor = random.uniform(0.8, 0.9)
        
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image.resize((width, height), Image.Resampling.LANCZOS)

def process_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"Hata: {DATASET_DIR} bulunamadı.")
        return

    categories = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for category in categories:
        folder_path = os.path.join(DATASET_DIR, category)
        prefix = f"data_{category}"
        
        # Sadece orijinal resimlerin listesini alalım (yeni üretilenleri tekrar üretmemek için)
        original_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not original_images:
            continue
            
        current_index = get_max_index(folder_path, prefix) + 1
        print(f"\n[{category}] Kategorisinde {len(original_images)} orijinal resim bulundu. Çoğaltılıyor...")
        
        augmented_count = 0
        for filename in original_images:
            filepath = os.path.join(folder_path, filename)
            
            try:
                with Image.open(filepath) as img:
                    # RGB formatına dönüştür (PNG'lerde şeffaflık hatasını önlemek için)
                    img = img.convert("RGB")
                    
                    for _ in range(AUGMENTATIONS_PER_IMAGE):
                        aug_img = augment_image(img)
                        
                        new_filename = f"{prefix}_PHOTO_{current_index}.jpg"
                        new_filepath = os.path.join(folder_path, new_filename)
                        
                        aug_img.save(new_filepath, "JPEG", quality=90)
                        current_index += 1
                        augmented_count += 1
            except Exception as e:
                print(f"Hata ({filename}): {e}")
                
        print(f"[{category}] Tamamlandı! {augmented_count} yeni resim üretildi.")

if __name__ == "__main__":
    print("Veri Çoğaltma (Data Augmentation) Başlıyor...")
    process_dataset()
    print("\nTüm işlemler başarıyla tamamlandı!")
