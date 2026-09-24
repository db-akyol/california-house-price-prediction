# California Konut Fiyatı Tahmini

1990 California nüfus sayımı verisini kullanarak bölgelerin **medyan konut değerini (`median_house_value`)** tahmin eden bir regresyon projesidir. Eksik veri doldurma, kategorik değişken kodlama ve dağılım dönüşümlerinden (Yeo-Johnson + Box-Cox) oluşan bir ön işleme hattı kurulur; ardından 9 farklı regresyon modeli aynı ölçütlerle (MAE / RMSE / R²) karşılaştırılır. Amaç, klasik lineer modeller ile ağaç tabanlı ensemble yöntemler arasındaki performans farkını ve hedef/öznitelik dönüşümünün model başarısına etkisini ortaya koymaktır.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-017CEE)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Box--Cox-8CAAE6?logo=scipy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Veri Seti

| | |
|---|---|
| **Kaynak** | [Kaggle – California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data) |
| **Dosya** | `21-housing.csv` |
| **Satır sayısı** | 20.640 |
| **Sütun sayısı** | 10 (9 sayısal + 1 kategorik) |
| **Hedef değişken** | `median_house_value` |
| **Eksik veri** | `total_bedrooms` sütununda 207 kayıt |

**Öznitelikler**

| Sütun | Açıklama |
|---|---|
| `longitude`, `latitude` | Bölgenin coğrafi koordinatları |
| `housing_median_age` | Bölgedeki konutların medyan yaşı |
| `total_rooms` | Toplam oda sayısı |
| `total_bedrooms` | Toplam yatak odası sayısı (eksik veri içerir) |
| `population` | Bölge nüfusu |
| `households` | Hane sayısı |
| `median_income` | Medyan hane geliri |
| `ocean_proximity` | Okyanusa yakınlık (kategorik) |
| `median_house_value` | **Hedef** – medyan konut değeri |

`ocean_proximity` dağılımı: `<1H OCEAN` (9.136), `INLAND` (6.551), `NEAR OCEAN` (2.658), `NEAR BAY` (2.290), `ISLAND` (5).

---

## Yöntem / İş Akışı

### 1. Keşifsel Veri Analizi (EDA)
- `info()`, `describe()`, `isnull().sum()` ile veri tipi, temel istatistik ve eksik değer kontrolü.
- 9 sayısal değişkenin dağılımı 3×3 grid halinde **histogram + KDE** ile görselleştirildi (`seaborn.histplot`).
- Korelasyon matrisi çıkarıldı. En güçlü sinyal **`median_income` ↔ `median_house_value` = 0.688**; ayrıca `total_rooms`, `total_bedrooms`, `population` ve `households` arasında 0.85–0.98 aralığında yüksek çoklu doğrusal bağlantı (multicollinearity) tespit edildi.

### 2. Veri Temizleme ve Hazırlık
- `total_bedrooms` içindeki 207 eksik değer **medyan** ile dolduruldu (ortalama yerine medyan: değişken sağa çarpık).
- `ocean_proximity` **one-hot encoding** ile sayısallaştırıldı (`pd.get_dummies`, `drop_first=True` → dummy trap'ten kaçınma).
- Öznitelik/hedef ayrımı yapıldı ve veri **%70 eğitim / %30 test** olarak bölündü (`random_state=15`).

### 3. Temel (Baseline) Modelleme
Ortak bir `evaluate_model()` fonksiyonu ile MAE, RMSE ve R² hesaplanarak 9 model hem eğitim hem test kümesinde değerlendirildi:

`LinearRegression` · `Lasso` · `Ridge` · `KNeighborsRegressor` · `DecisionTreeRegressor` · `RandomForestRegressor` · `AdaBoostRegressor` · `GradientBoostingRegressor` · `XGBRegressor`

Eğitim/test skorları birlikte raporlanarak **overfitting** doğrudan gözlemlendi (ör. Decision Tree eğitimde R² = 1.0, testte 0.645).

### 4. Dağılım Dönüşümü (Feature & Target Transformation)
- **Öznitelikler:** `PowerTransformer(method="yeo-johnson")` — eğitim setinde `fit_transform`, test setinde yalnızca `transform` (veri sızıntısı önlendi).
- **Hedef:** `scipy.stats.boxcox` ile `y_train` normalleştirildi; tahminler `inverse_boxcox()` fonksiyonuyla orijinal dolar ölçeğine geri döndürülerek metrikler **karşılaştırılabilir ölçekte** hesaplandı.
- Dönüşüm öncesi/sonrası dağılımlar `plot_all_histograms()` yardımcı fonksiyonuyla yan yana görselleştirildi.
- Tüm modeller dönüştürülmüş veri üzerinde yeniden eğitilip test edildi.

---

## Sonuçlar

### Dönüşüm öncesi (ham öznitelikler + ham hedef) — Test seti

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **XGBoost Regressor** | **48.194** | 31.983 | **0.827** |
| Random Forest Regressor | 49.259 | 31.978 | 0.819 |
| Gradient Boost Regressor | 55.700 | 38.552 | 0.769 |
| Decision Tree | 69.034 | 44.151 | 0.645 |
| Linear Regression | 69.423 | 49.827 | 0.641 |
| Lasso | 69.423 | 49.827 | 0.641 |
| Ridge | 69.423 | 49.830 | 0.641 |
| AdaBoost Regressor | 86.399 | 73.253 | 0.444 |
| K Neighbors Regressor | 100.161 | 77.487 | 0.253 |

### Dönüşüm sonrası (Yeo-Johnson öznitelikler + Box-Cox hedef) — Test seti

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **XGBoost Regressor** | **49.058** | 31.488 | **0.821** |
| Random Forest Regressor | 50.869 | 32.392 | 0.807 |
| Gradient Boost Regressor | 57.317 | 38.517 | 0.756 |
| K Neighbors Regressor | 63.443 | 42.877 | 0.700 |
| Linear Regression | 68.057 | 47.856 | 0.655 |
| Ridge | 68.060 | 47.857 | 0.655 |
| Decision Tree | 69.188 | 44.400 | 0.644 |
| AdaBoost Regressor | 82.559 | 60.441 | 0.493 |
| Lasso | 98.637 | 71.333 | 0.276 |

> RMSE ve MAE değerleri dolar cinsindendir (binlik ayraçla gösterilmiştir).

### Öne çıkan bulgular

- **En iyi model: XGBoost Regressor** — ham veride **R² = 0.827 / RMSE = 48.194 $**, dönüştürülmüş veride **R² = 0.821 / RMSE = 49.058 $**.
- Ağaç tabanlı ensemble modeller (XGBoost, Random Forest), lineer modellere kıyasla R² değerini yaklaşık **0.64 → 0.82** seviyesine taşıdı.
- Dönüşümden **en çok fayda gören model KNN** oldu: R² 0.253 → 0.700. Bu, mesafe tabanlı algoritmaların ölçek ve çarpıklığa ne kadar duyarlı olduğunu net biçimde gösteriyor.
- Varsayılan `alpha` değeriyle çalışan **Lasso**, dönüştürülmüş (ve dolayısıyla küçük ölçekli) hedef üzerinde katsayıları fazla cezalandırarak ciddi performans kaybına uğradı (0.641 → 0.276).
- Ağaç tabanlı modeller ölçekten bağımsız çalıştığı için dönüşüm bu modellerde anlamlı bir kazanç sağlamadı.

---

## Kurulum ve Çalıştırma

```bash
# 1) Depoyu klonlayın
git clone https://github.com/<kullanici-adi>/California-House-Price-Data.git
cd California-House-Price-Data

# 2) (Önerilir) Sanal ortam oluşturun
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3) Bağımlılıkları kurun
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy jupyter

# 4) Notebook'u açın
jupyter notebook california_house_price.ipynb
```

Notebook'u baştan sona çalıştırmak için: **Kernel → Restart & Run All**. Veri seti (`21-housing.csv`) depo içinde yer aldığından ek indirme gerekmez.

---

## Dosya Yapısı

```
California-House-Price-Data/
├── california_house_price.ipynb   # EDA, ön işleme, modelleme ve karşılaştırma
├── 21-housing.csv                 # Veri seti (20.640 satır × 10 sütun)
├── .gitignore
├── LICENSE                        # MIT
└── README.md
```

---

## Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakınız.
