import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. API AYARLARI
# ==========================================
app = FastAPI(title="Araba Sınıflandırma API", description="C# uygulamasından gelen resimleri PyTorch ile sınıflandırır.")

# CORS ayarları (.NET uygulamasından bu API'ye sorunsuz HTTP isteği atabilmek için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tüm kaynaklara izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu") # Web sunucusunda GPU şart olmadığı için CPU'ya sabitliyoruz

# ==========================================
# 2. MODELİN RAM'E YÜKLENMESİ (Sadece 1 Kere)
# ==========================================
# Bu sınıfların sırası, eğitimdeki klasörlerin ALFABETİK sırasıyla BİREBİR AYNI olmalıdır.
class_names = [
    'ACIK_TEKERLEKLI', 
    'HATCHBACK', 
    'MICRO', 
    'PICK_UP', 
    'SEDAN', 
    'STATION_WAGON', 
    'SUV', 
    'VAN'
]

# Boş model iskeletini oluştur
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, len(class_names))
)

# Eğittiğimiz ağırlıkları içine yükle
MODEL_PATH = 'araba_modeli_pytorch.pth'
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Modeli tahmin moduna al
    print(f"✅ Model ({MODEL_PATH}) başarıyla API'ye yüklendi ve hazır!")
except Exception as e:
    print(f"⚠️ DİKKAT: '{MODEL_PATH}' bulunamadı. Önce eğitimi tamamladığından emin ol.")

# ⚠️ KRİTİK: Bu pipeline, Kaggle'daki eğitim kodunun 'val' transform'u ile BİREBİR AYNI olmalı!
# Kaggle eğitim val transform: Resize(256,256) → CenterCrop(224) → ToTensor → Normalize
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),      # Önce 256x256'ya büyüt
    transforms.CenterCrop(224),         # Sonra ortadan 224x224 kes
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. API UÇ NOKTASI (ENDPOINT)
# ==========================================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Gelen byte verisini resme çevir
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Resmi modele hazır tensör yapısına dönüştür
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Modeli çalıştırıp sonucu al
        with torch.no_grad():
            output = model(input_batch)
            
        # Çıkan sayısal değerleri % (yüzdelik) olasılığa çevir
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # En yüksek olasılıklı sınıfı bul
        top_prob, top_catid = torch.topk(probabilities, 1)
        predicted_class = class_names[top_catid]
        
        # C# tarafında grafiği (Chart.js) çizdirebilmek için tüm olasılıkları sözlük yapalım
        prob_dict = {}
        for i, class_name in enumerate(class_names):
            prob_dict[class_name] = round(probabilities[i].item() * 100, 2)
            
        # C# uygulamasına bu JSON verisini döndür
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(top_prob.item() * 100, 2),
            "probabilities": prob_dict
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("🌐 Python API ayağa kalkıyor... Adres: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
