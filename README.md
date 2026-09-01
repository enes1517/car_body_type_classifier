# 🚗 Araç Gövde Tipi Sınıflandırma Projesi (Car Body Type Classification)

Bu proje, görüntü işleme ve derin öğrenme teknikleri kullanılarak araçların gövde tiplerini (Sedan, SUV, Hatchback vb.) sınıflandırmayı amaçlayan kapsamlı bir yapay zeka sistemidir. 

Proje, modelin eğitilmesi, test edilmesi, REST API olarak sunulması ve hem modern bir Python (FastAPI) web arayüzü hem de C# (ASP.NET Core) uygulaması üzerinden kullanıcıya ulaştırılması gibi uçtan uca tüm aşamaları barındırmaktadır.

---

## 🌟 Öne Çıkan Özellikler

- **Gelişmiş Derin Öğrenme Modeli:** Transfer Learning yöntemiyle **ResNet-50** mimarisi kullanılmış ve gövde tipi sınıflandırması için özel olarak modifiye edilmiştir.
- **Ezberleme Önleyici (Anti-Overfitting) Eğitim:** Eğitim sırasında Random Erasing, Color Jitter, Gaussian Blur gibi ağır veri çoğaltma (Data Augmentation) teknikleri kullanılarak modelin ezberlemesi engellenmiştir. Label Smoothing ve Weight Decay gibi modern teknikler içerir.
- **Modern Web Arayüzü (Python):** FastAPI kullanılarak geliştirilmiş, "Glassmorphism" tasarıma sahip, Chart.js entegrasyonlu ve sürükle-bırak destekli harika bir arayüz.
- **C# ASP.NET Core Entegrasyonu:** Eğitilen model `.onnx` formatına dönüştürülerek C# ortamında (ONNX Runtime) çalıştırılabilir modern bir MVC web uygulaması oluşturulmuştur.
- **Çok Sınıflı Tespit:** 8 farklı araç gövde tipi desteklenmektedir: `AÇIK TEKERLEKLİ (F1)`, `HATCHBACK`, `MICRO`, `PICK-UP`, `SEDAN`, `STATION WAGON`, `SUV`, `VAN`.

---

## 📂 Proje Yapısı

```text
📦 yazlab3
 ┣ 📂 CarClassificationApp    # C# ASP.NET Core Web Uygulaması (.onnx modeli kullanır)
 ┣ 📂 dataset                 # Eğitim ve doğrulama için veri setinin bulunduğu klasör
 ┣ 📜 api_servis.py           # FastAPI tabanlı, modern web arayüzüne sahip model sunum dosyası
 ┣ 📜 kaggle_resnet50_super.py # Kaggle ve yerel ortamlar için gelişmiş ResNet-50 eğitim betiği
 ┣ 📜 test_manuel.py          # Modeli dışarıdan verilen rastgele fotoğraflarla test etme betiği
 ┣ 📜 araba_modeli_resnet50 (10).pth # Eğitilmiş PyTorch model ağırlıkları
 ┣ 📜 araba_modeli.onnx       # C# vs. diğer ortamlarda kullanım için dışa aktarılmış ONNX modeli
 ┣ 📜 calculate_f1.py         # Test sonuçları üzerinden F1 skorunu hesaplayan yardımcı betik
 ┣ 📜 plot_f1.py              # Sonuç grafiklerini (Accuracy/Loss vs) çizen betik
 ┣ 📜 PredictionScript.py     # Modeli hızlıca bir resim üzerinde tahmin yaptırmak için komut dosyası
 ┗ 📜 rapor_latex.tex         # Projenin LaTeX formatında akademik raporu
```

---

## 🧠 Model Mimarisi ve Eğitim Stratejisi

Model **PyTorch** framework'ü kullanılarak geliştirilmiştir. 

**Mimari Detayları:**
- Önceden ImageNet verisiyle eğitilmiş `ResNet-50` (`IMAGENET1K_V2` ağırlıkları) iskelet olarak kullanılmıştır.
- Son Katman (Fully Connected Layer) tamamen değiştirilmiş; yoğun ezberleme (overfitting) ihtimaline karşı **2 adet Dropout katmanı** (`p=0.6` ve `p=0.4`) ve araya `BatchNorm1d` katmanı eklenerek normalize edilmiştir.

**Anti-Overfitting (Ezber Bozma) Stratejileri:**
- **Aggressive Data Augmentation:** Modelin far, amblem veya arka plan gibi spesifik pikselleri ezberlemesini önlemek için: 
  - `RandomResizedCrop`, `ColorJitter`, `RandomRotation`, `RandomAffine`, `GaussianBlur` 
  - Görüntü üzerine rastgele siyah kutular koyan `RandomErasing`.
- **Label Smoothing:** Modelin tahminlerinde aşırı güvenli (kendinden %100 emin) olmasını engelleyerek yumuşatılmış kayıp fonksiyonu (CrossEntropyLoss + Label Smoothing).
- **Early Stopping & LR Scheduler:** Validation Loss artmaya başladığında eğitimi erken durdurur ve takıldığı noktalarda öğrenme oranını (Learning Rate) otomatik olarak düşürür.

---

## 🚀 Kurulum ve Çalıştırma Yönergeleri

### 1. Python Web Arayüzünü Başlatma (FastAPI)
Modeli modern ve görsel bir arayüzde doğrudan test etmek için bu yöntemi kullanabilirsiniz.

Gerekli kütüphaneleri yükleyin:
```bash
pip install torch torchvision fastapi uvicorn python-multipart Pillow
```

Sunucuyu başlatın:
```bash
python api_servis.py
```
> Ardından tarayıcınızdan **`http://127.0.0.1:8000`** adresine giderek sistemi kullanabilirsiniz.

### 2. C# .NET Uygulamasını Başlatma
Eğer modeli C# ekosisteminde denemek isterseniz:

1. `CarClassificationApp` klasörünün içerisine gidin.
2. Konsolu veya Visual Studio'yu açın.
3. Uygulamayı derleyin ve çalıştırın:
```bash
cd CarClassificationApp
dotnet run
```
> Bu uygulama, tahminleme işlemleri için `Assets` klasöründeki `araba_modeli.onnx` dosyasını kullanır.

### 3. Modeli Yeniden Eğitmek
Kendi verinizle veya modeli sıfırdan eğitmek için:
```bash
python kaggle_resnet50_super.py
```
*Not: Bu betik hem lokal ortamı hem de Kaggle çalışma ortamını otomatik tanır ve dosya yollarını ona göre ayarlayıp eğitimi başlatır.*

### 4. Manuel Toplu Test
Sınıflandırılmış klasörlerde bulunan test resimleri üzerindeki başarıyı konsolda görmek için:
```bash
python test_manuel.py
```

---

## 📊 Performans ve Metrikler

Eğitim süreci sonunda (`kaggle_resnet50_super.py`), model çıktısı olarak aşağıdaki raporlar otomatik oluşturulur:
- `1_Loss_Grafik.png` ve `2_Accuracy_Grafik.png`: Eğitim ve doğrulama grafikleri.
- `3_Siniflandirma_Raporu.txt`: Precision, Recall, Macro/Weighted F1-Score detayları.
- `4_Normalized_Confusion_Matrix.png`: Sınıflar arası karışıklığı gösteren Isı Haritası (Heatmap).

---

## 🛠️ Kullanılan Teknolojiler

- **Yapay Zeka & Derin Öğrenme:** PyTorch, Torchvision, ONNX
- **Veri Manipülasyonu & Analiz:** Numpy, Scikit-learn, Matplotlib, Seaborn
- **Web Geliştirme (Python):** FastAPI, Uvicorn
- **Web Geliştirme (C#):** ASP.NET Core, ONNX Runtime
- **Arayüz Tasarımı (UI):** HTML5, Vanilla CSS (Glassmorphism), Chart.js, Bootstrap 5
