# alldata-models — Kapsamlı Dokümantasyon

> İki başlıklı (2-head) çok sınıflı zaman serisi sınıflandırma sistemi.
> **Zaman serisi türünü** (base type) ve **anomali türünü** (anomaly type) aynı anda tahmin eder.
> tsfresh EfficientFCParameters ile 777 istatistiksel özellik çıkarılır; LightGBM / XGBoost / MLP ile eğitilir.

---

## İçindekiler

1. [Proje Amacı](#1-proje-amacı)
2. [Dizin Yapısı](#2-dizin-yapısı)
3. [Veri Kaynakları — Hangi Klasörden Kaç Örnek](#3-veri-kaynakları)
4. [Özellik Çıkarımı (tsfresh)](#4-özellik-çıkarımı)
5. [Model Mimarisi](#5-model-mimarisi)
6. [Eğitim Stratejisi](#6-eğitim-stratejisi)
7. [Sonuçlar](#7-sonuçlar)
   - 7.1 Genel Özet
   - 7.2 Validation Skorları (Tüm Modeller)
   - 7.3 Base Type — Sınıf Bazında Doğruluk
   - 7.4 Anomaly Type — Sınıf Bazında Doğruluk
   - 7.5 Kombinasyon Grid'i (4×7 Full-Match Matrisi)
   - 7.6 Tüm Kategoriler — Detaylı Breakdown Tablosu
   - 7.7 Full-Match Dağılımı
8. [Teknik Notlar ve Çözülen Sorunlar](#8-teknik-notlar)
9. [Nasıl Çalıştırılır](#9-nasıl-çalıştırılır)

---

## 1. Proje Amacı

`Generated Data` dizini altında üretilmiş sentetik zaman serilerinden yararlanarak:

- **Head 1 — Base Type (4 sınıf):** Serinin temel istatistiksel karakterini belirle
  `stationary` | `deterministic_trend` | `stochastic_trend` | `volatility`

- **Head 2 — Anomaly Type (7 sınıf):** Seride hangi anomali türü var?
  `none` | `collective_anomaly` | `contextual_anomaly` | `mean_shift` | `point_anomaly` | `trend_shift` | `variance_shift`

**Full-match metriği:** Her iki baş aynı anda doğruysa başarılı sayılır.
Tek bir seri → 28 teorik kombinasyondan birine atanır (veri setinde 23 kombinasyon mevcut).

---

## 2. Dizin Yapısı

```
alldata-models/
├── config.py          # Tüm parametreler, label tanımları, klasör yolları
├── processor.py       # CSV okuma, tsfresh özellik çıkarımı, chunk işleme
├── trainer.py         # Model eğitimi, değerlendirme, sonuç kaydetme
├── main.py            # Ana giriş noktası (processor + trainer orchestration)
├── unzip_all.py       # Generated Data altındaki ZIP'leri açar (idempotent)
├── processed_data/    # (geçici) X.npy, y_base.npy, y_anomaly.npy, feature_names.json
├── results/
│   └── training_results.json   # Tüm metrikler, val skorları, per-combination breakdown
└── README.md          # Bu dosya
```

---

## 3. Veri Kaynakları

### Örnekleme Mantığı

Her kaynak klasör için:
1. `get_leaf_dirs()` — Doğrudan CSV içeren en derin alt klasörler bulunur
   (örn. `stationary/short/`, `stationary/medium/`, `stationary/long/`)
2. `sample_from_leaves()` — Her yaprak klasörden eşit sayıda CSV seçilir
   → `short / medium / long` dengesini otomatik sağlar
3. **`SAMPLES_PER_CATEGORY = 350`** — Her kategoriden en fazla 350 örnek

Toplam veri seti: **~13,493 seri** → 39 kaynak kategori

---

### 3.1 Tekli Kategoriler (10 adet)

`Generated Data/` altında doğrudan bulunan klasörler:

| Klasör Adı             | Base Label           | Anomaly Label        | Açıklama                                |
|------------------------|----------------------|----------------------|-----------------------------------------|
| `stationary`           | stationary           | none                 | Anomalisiz durağan seri                 |
| `deterministic_trend`  | deterministic_trend  | none                 | Anomalisiz deterministik trend          |
| `Stochastic Trend`     | stochastic_trend     | none                 | Anomalisiz stokastik trend              |
| `Volatility`           | volatility           | none                 | Anomalisiz volatilite                   |
| `collective_anomaly`   | stationary           | collective_anomaly   | Durağan taban + toplu anomali           |
| `contextual_anomaly`   | stationary           | contextual_anomaly   | Durağan taban + bağlamsal anomali       |
| `mean_shift`           | stationary           | mean_shift           | Durağan taban + ortalama kayması        |
| `point_anomaly`        | stationary           | point_anomaly        | Durağan taban + nokta anomali           |
| `trend_shift`          | stationary           | trend_shift          | Durağan taban + trend kayması           |
| `variance_shift`       | stationary           | variance_shift       | Durağan taban + varyans kayması         |

---

### 3.2 Kombinasyon Kategorileri (29 adet)

`Generated Data/Combinations/` altında bulunan iç içe klasörler:

#### Cubic Base (4 kombinasyon)

| Yol (Combinations altında)                                  | Base Label           | Anomaly Label      |
|-------------------------------------------------------------|----------------------|--------------------|
| `Cubic Base/Cubic Base/Cubic + Mean Shift`                  | deterministic_trend  | mean_shift         |
| `Cubic Base/Cubic Base/Cubic + Point Anomaly`               | deterministic_trend  | point_anomaly      |
| `Cubic Base/Cubic Base/Cubic + Variance Shift`              | deterministic_trend  | variance_shift     |
| `Cubic Base/Cubic Base/cubic_collective_anomaly`             | deterministic_trend  | collective_anomaly |

#### Damped Base (4 kombinasyon)

| Yol (Combinations altında)                                  | Base Label           | Anomaly Label      |
|-------------------------------------------------------------|----------------------|--------------------|
| `Damped Base/Damped Base/Damped + Collective Anomaly`       | deterministic_trend  | collective_anomaly |
| `Damped Base/Damped Base/Damped + Mean Shift`               | deterministic_trend  | mean_shift         |
| `Damped Base/Damped Base/Damped + Point Anomaly`            | deterministic_trend  | point_anomaly      |
| `Damped Base/Damped Base/Damped + Variance Shift`           | deterministic_trend  | variance_shift     |

#### Exponential Base (4 kombinasyon)

| Yol (Combinations altında)                                      | Base Label           | Anomaly Label      |
|-----------------------------------------------------------------|----------------------|--------------------|
| `Exponential Base/Exponential Base/Exponential + Mean Shift`    | deterministic_trend  | mean_shift         |
| `Exponential Base/Exponential Base/exponential_collective_anomaly` | deterministic_trend | collective_anomaly |
| `Exponential Base/Exponential Base/exponential_point_anomaly`   | deterministic_trend  | point_anomaly      |
| `Exponential Base/Exponential Base/exponential_variance_shift`  | deterministic_trend  | variance_shift     |

#### Linear Base (5 kombinasyon)

| Yol (Combinations altında)                                  | Base Label           | Anomaly Label      |
|-------------------------------------------------------------|----------------------|--------------------|
| `Linear Base/Linear Base/Linear + Collective Anomaly`       | deterministic_trend  | collective_anomaly |
| `Linear Base/Linear Base/Linear + Mean Shift`               | deterministic_trend  | mean_shift         |
| `Linear Base/Linear Base/Linear + Point Anomaly`            | deterministic_trend  | point_anomaly      |
| `Linear Base/Linear Base/Linear + Trend Shift`              | deterministic_trend  | trend_shift        |
| `Linear Base/Linear Base/Linear + Variance Shift`           | deterministic_trend  | variance_shift     |

#### Quadratic Base (4 kombinasyon)

| Yol (Combinations altında)                                      | Base Label           | Anomaly Label      |
|-----------------------------------------------------------------|----------------------|--------------------|
| `Quadratic Base/Quadratic Base/Quadratic + Collective anomaly`  | deterministic_trend  | collective_anomaly |
| `Quadratic Base/Quadratic Base/Quadratic + Mean Shift`          | deterministic_trend  | mean_shift         |
| `Quadratic Base/Quadratic Base/Quadratic + Point Anomaly`       | deterministic_trend  | point_anomaly      |
| `Quadratic Base/Quadratic Base/Quadratic + Variance Shift`      | deterministic_trend  | variance_shift     |

#### Stochastic Trend + Anomali (4 kombinasyon)

| Yol (Combinations altında)                                               | Base Label      | Anomaly Label      |
|--------------------------------------------------------------------------|-----------------|---------------------|
| `Stochastic Trend + Collective Anomaly`                                  | stochastic_trend | collective_anomaly |
| `Stochastic Trend + Mean Shift`                                          | stochastic_trend | mean_shift         |
| `Stochastic Trend + Point Anomaly`                                       | stochastic_trend | point_anomaly      |
| `Stochastic Trend + Variance Shift/Stochastic Trend + Variance Shift`    | stochastic_trend | variance_shift     |


#### Volatility + Anomali (4 kombinasyon)

| Yol (Combinations altında)          | Base Label | Anomaly Label      |
|-------------------------------------|------------|--------------------|
| `Volatility + Collective Anomaly`   | volatility | collective_anomaly |
| `Volatility + Mean Shift`           | volatility | mean_shift         |
| `Volatility + Point Anomaly`        | volatility | point_anomaly      |
| `Volatility + Variance Shift`       | volatility | variance_shift     |

---

### 3.3 Veri Seti Özeti

| Parametre             | Değer                              |
|-----------------------|------------------------------------|
| Kaynak kategori sayısı | 39 (10 tekli + 29 kombinasyon)    |
| Kategori başına max   | 350 örnek                          |
| Toplam seri           | ~13,493                            |
| Min seri uzunluğu     | 50 zaman adımı                     |
| Train / Val / Test    | 70% / 10% / 20%                    |
| Train seti            | ~9,445 seri                        |
| Validation seti       | ~1,349 seri                        |
| Test seti             | 2,699 seri                         |
| Gözlemlenen kombinasyon | 23 (teorik max 28)              |

---

## 4. Özellik Çıkarımı

### 4.1 tsfresh EfficientFCParameters

Her zaman serisinden **777 istatistiksel özellik** çıkarılır.

```python
from tsfresh.feature_extraction import EfficientFCParameters

tsfresh_extract(
    combined_df,
    column_id='id',
    column_sort='time',
    column_value='value',
    default_fc_parameters=EfficientFCParameters(),
    disable_progressbar=True,
    n_jobs=4,
)
```

Özellik kategorileri (seçilmiş örnekler):

| Kategori                   | Özellik Örnekleri                                                   |
|----------------------------|---------------------------------------------------------------------|
| İstatistiksel momentler    | mean, variance, skewness, kurtosis                                  |
| Otokorelasyon              | ACF değerleri (lag 1..40), PACF değerleri                           |
| Fourier / frekans          | FFT katsayıları, spectral energy, spectral centroid                 |
| Lineer trend               | lineer regresyon eğimi, R², arta kalan istatistikler                |
| Entropiler                 | permutation entropy, approximate entropy, sample entropy            |
| Karmaşıklık                | CID metric, Hjorth parametreleri                                     |
| Periyodisite               | number of peaks, peak prominence, CWT katsayıları                  |
| Eşik bazlı                 | ratio beyond r sigma, count above/below mean                        |
| Zaman alan                 | absolute sum of changes, mean absolute change, longest strike above |

Eksik değerler `tsfresh.utilities.dataframe_functions.impute()` ile doldurulur.

### 4.2 Chunk İşleme (Bellek Yönetimi)

Büyük veri setleri için RAM taşmasını önlemek amacıyla chunk tabanlı işleme kullanılır:

```
CHUNK_SIZE = 4000 seri

Chunk 1: seri 0–3999    → chunk_000.npy (diske yazılır, RAM temizlenir)
Chunk 2: seri 4000–7999 → chunk_001.npy
...
Son chunk: kalan seriler → chunk_XXX.npy
Final: tüm chunk'lar np.vstack() ile birleştirilir → X.npy
Geçici dosyalar silinir.
```

Her chunk arasında `gc.collect()` çağrılarak Python çöp toplayıcısı zorlanır.

### 4.3 Ölçekleme

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
```

---

## 5. Model Mimarisi

### 5.1 İki Bağımsız Sınıflandırıcı (2-Head)

Aynı özellik matrisi `X` üzerinde **iki ayrı** bağımsız model eğitilir:

```
X (13,493 × 777)
    │
    ├─► Head 1: BASE TYPE classifier
    │       4 sınıf: stationary / deterministic_trend / stochastic_trend / volatility
    │
    └─► Head 2: ANOMALY TYPE classifier
            7 sınıf: none / collective_anomaly / contextual_anomaly /
                     mean_shift / point_anomaly / trend_shift / variance_shift
```

Her head için üç model aday olarak eğitilir; **validation F1 (macro)** en yüksek olan seçilir.

### 5.2 Eğitilen Modeller

#### LightGBM
```python
LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',    # sınıf dengesizliğine karşı
    n_jobs=-1,
)
```

#### XGBoost
```python
XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    n_jobs=-1,
)
```

#### MLP (sklearn)
```python
MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),   # 3 gizli katman
    max_iter=600,
    early_stopping=True,
    validation_fraction=0.1,
)
```

> **Not:** `MLPClassifier` `class_weight` parametresini desteklemez;
> sınıf dengesi LightGBM'de `class_weight='balanced'` ile, örneklemede ise
> her kategoriden eşit sayıda CSV seçerek sağlanır.

### 5.3 Model Seçim Kriteri

```
En iyi model = argmax(val_F1_macro) üzerinden seçilir
→ Seçilen model test setinde değerlendirilir
```

---

## 6. Eğitim Stratejisi

### 6.1 Veri Bölme

```
Toplam ~13,493 seri
    ↓ stratify=y_base ile ayır
├── Test  (20%) : 2,699 seri  — sabit, hiç dokunulmaz
└── Geride kalanlar (80%)
        ↓ stratify=y_base ile ayır
        ├── Validation (10% toplam = 12.5% of 80%) : ~1,349 seri
        └── Train      (70% toplam)                : ~9,445 seri
```

`stratify=y_base` kullanılarak her bölümde base type dağılımı korunur.

### 6.2 Sınıf Dengesi

İki mekanizma birlikte çalışır:

| Mekanizma | Uygulama |
|-----------|----------|
| **Eşit örnekleme** | Her kategori klasöründen en fazla 350 örnek; yaprak klasörlerde (short/medium/long) eşit dağılım |
| **class_weight='balanced'** | LightGBM'de seyrek sınıflara daha yüksek ağırlık verilir |

### 6.3 Full-Match Metriği

```python
base_correct    = (base_pred == base_true)
anomaly_correct = (anomaly_pred == anomaly_true)

full_match   = sum(base_correct & anomaly_correct)   # her ikisi doğru
base_only    = sum(base_correct & ~anomaly_correct)  # sadece base doğru
anomaly_only = sum(~base_correct & anomaly_correct)  # sadece anomaly doğru
no_match     = sum(~base_correct & ~anomaly_correct) # her ikisi yanlış
```

---

## 7. Sonuçlar

### 7.1 Genel Özet

| Metrik                          | Değer              |
|---------------------------------|--------------------|
| **Full Match (test)**           | **86.66%** (2339/2699) |
| Base Type Accuracy (test)       | 97.04%             |
| Anomaly Type Accuracy (test)    | 88.37%             |
| Base Type F1-macro (test)       | 0.9526             |
| Anomaly Type F1-macro (test)    | 0.8974             |
| En İyi Base Model               | XGBoost            |
| En İyi Anomaly Model            | LightGBM           |
| Test seti büyüklüğü             | 2,699 seri         |
| Özellik sayısı                  | 777                |

---

### 7.2 Validation Skorları — Tüm Modeller

#### Head 1: Base Type

| Model     | Val F1 (macro) | Val Accuracy |
|-----------|---------------|--------------|
| **XGBoost**   | **0.9580**    | **0.9741**   |
| LightGBM  | 0.9482        | 0.9689       |
| MLP       | 0.8858        | 0.9222       |

→ **Seçilen: XGBoost** (Val F1 = 0.9580)
→ Test F1 = 0.9526 | Test Acc = 0.9704

#### Head 2: Anomaly Type

| Model     | Val F1 (macro) | Val Accuracy |
|-----------|---------------|--------------|
| **LightGBM**  | **0.8830**    | **0.8726**   |
| XGBoost   | 0.8828        | 0.8741       |
| MLP       | 0.7356        | 0.7222       |

→ **Seçilen: LightGBM** (Val F1 = 0.8830)
→ Test F1 = 0.8974 | Test Acc = 0.8837

---

### 7.3 Base Type — Sınıf Bazında Full-Match Doğruluğu

> Not: Bu değerler test setindeki full-match oranlarından hesaplanmıştır.
> Head 1 tek başına %97 doğrulukla çalışır; aşağıdaki tablo her iki başın aynı anda
> doğru olduğu durumu yansıtır.

| Sınıf                | Doğru | Toplam | Oran   |
|----------------------|-------|--------|--------|
| deterministic_trend  | 1483  | 1527   | 97.1%  |
| stochastic_trend     |  278  |  349   | 79.7%  |
| stationary           |  345  |  473   | 72.9%  |
| volatility           |  233  |  350   | 66.6%  |


---

### 7.4 Anomaly Type — Sınıf Bazında Full-Match Doğruluğu

| Anomali Türü       | Doğru | Toplam | Oran   |
|--------------------|-------|--------|--------|
| contextual_anomaly |   62  |    62  | 100.0% |
| trend_shift        |  123  |   134  |  91.8% |
| variance_shift     |  508  |   563  |  90.2% |
| point_anomaly      |  495  |   562  |  88.1% |
| mean_shift         |  508  |   583  |  87.1% |
| collective_anomaly |  452  |   527  |  85.8% |
| none               |  191  |   268  |  71.3% |


---

### 7.5 Kombinasyon Grid'i — Full-Match Matrisi (4 × 7)

Hücre formatı: `doğru / toplam = oran%`
`N/A` = veri setinde bu kombinasyon mevcut değil

```
                      none       collective  contextual  mean_shift  point_anom  trend_sh   var_shift
                    ─────────────────────────────────────────────────────────────────────────────────
stationary          49/72=68%   46/73=63%   62/62=100%  46/63=73%   39/68=57%   53/64=83%  50/71=70%
deterministic_trend 47/55=85%   331/331=100% N/A        356/383=93% 348/348=100% 70/70=100% 331/340=97%
stochastic_trend    53/70=76%   57/69=83%   N/A         44/66=67%   57/67=85%   N/A         67/77=87%
volatility          42/71=59%   18/54=33%   N/A         62/71=87%   51/79=65%   N/A         60/75=80%
```


---

### 7.6 Tüm Kategoriler — Detaylı Breakdown Tablosu

Test setindeki 23 kombinasyon, düşükten yükseğe sıralanmış:

```
Kategori                                              Doğru  Yanlış Toplam    Oran
──────────────────────────────────────────────────────────────────────────────────
volatility + collective_anomaly                          18      36     54   33.3%
stationary + point_anomaly                               39      29     68   57.4%
volatility + none                                        42      29     71   59.2%
stationary + collective_anomaly                          46      27     73   63.0%
volatility + point_anomaly                               51      28     79   64.6%
stochastic_trend + mean_shift                            44      22     66   66.7%
stationary + none                                        49      23     72   68.1%
stationary + variance_shift                              50      21     71   70.4%
stationary + mean_shift                                  46      17     63   73.0%
stochastic_trend + none                                  53      17     70   75.7%
volatility + variance_shift                              60      15     75   80.0%
stochastic_trend + collective_anomaly                    57      12     69   82.6%
stationary + trend_shift                                 53      11     64   82.8%
stochastic_trend + point_anomaly                         57      10     67   85.1%
deterministic_trend + none                               47       8     55   85.5%
stochastic_trend + variance_shift                        67      10     77   87.0%
volatility + mean_shift                                  62       9     71   87.3%
deterministic_trend + mean_shift                        356      27    383   93.0%
deterministic_trend + variance_shift                    331       9    340   97.4%
deterministic_trend + collective_anomaly                331       0    331  100.0%
deterministic_trend + point_anomaly                     348       0    348  100.0%
stationary + contextual_anomaly                          62       0     62  100.0%
deterministic_trend + trend_shift                        70       0     70  100.0%
──────────────────────────────────────────────────────────────────────────────────
TOPLAM                                                 2339     360   2699   86.7%
```

---

### 7.7 Full-Match Dağılımı

```
Full Match   (her ikisi doğru) : 2339 / 2699  (86.66%)
Sadece Base  (anomaly yanlış)  :  280 / 2699  (10.37%)
Sadece Anomali (base yanlış)   :   46 / 2699  ( 1.70%)
No Match     (her ikisi yanlış):   34 / 2699  ( 1.26%)
```


---

## 8. Teknik Notlar

### 8.1 Çözülen Sorunlar

#### MLPClassifier `class_weight` Hatası
```
TypeError: MLPClassifier.__init__() got an unexpected keyword argument 'class_weight'
```
sklearn'in MLP implementasyonu `class_weight`'i desteklemez.
Çözüm: `class_weight` yalnızca LightGBM'de bırakıldı; MLP'den kaldırıldı.

#### tsfresh `n_jobs=-1` Bellek Hatası
```
ValueError: Number of processes must be at least 1
```
`n_jobs=-1` büyük veri setinde tüm CPU'ları kullanmaya çalışırken RAM taşması nedeniyle
süreç sayısı sıfıra düşüyordu.
Çözüm: `n_jobs=4` sabit değerine getirildi.

#### Chunk İşleme Sırasında RAM Taşması
```
numpy.core._exceptions._ArrayMemoryError
```
`CHUNK_SIZE=8000` iken chunk 15'te RAM doldu.
Çözüm: `CHUNK_SIZE=4000`, her chunk diske `.npy` olarak yazılıp RAM'den silinir,
`gc.collect()` zorla çalıştırılır.

#### Stochastic Trend + Variance Shift — ZIP Sorunu
Klasör CSV değil, 105 adet ZIP içeriyordu.
`get_leaf_dirs()` CSV bulamadığından kategoriyi atlıyordu.
Çözüm: `unzip_all.py` yazıldı ve çalıştırıldı.

#### MinimalFCParameters Deney Başarısızlığı
10 özelliğe geçildiğinde Full Match: **87.78% → 62.12%** düştü.
EfficientFCParameters'a (777 özellik) geri dönüldü.

### 8.2 Denenen Parametre Kombinasyonları

| Deney | SAMPLES_PER_CATEGORY | Özellik Seti | Toplam Seri | Full Match |
|-------|----------------------|--------------|-------------|------------|
| Başlangıç | 1000 | EfficientFC | ~37K | ~87.78% |
| MinimalFC testi | 1000 | MinimalFC (10 özellik) | ~37K | **62.12%** |
| Bellek optimizasyonu | 350 | EfficientFC | ~13.5K | **86.66%** |


---

## 9. Nasıl Çalıştırılır

### Ön Koşullar

```bash
pip install tsfresh lightgbm xgboost scikit-learn numpy pandas tqdm
```

### Adım 0: ZIP Açma (gerekiyorsa)

`Generated Data` altında ZIP olan klasörler varsa:

```bash
python unzip_all.py
```

Idempotent'tir — zaten açılmış CSV'leri atlar.

### Adım 1+2: İşleme + Eğitim (tam akış)

```bash
python main.py
```

### Sadece Eğitim (işlenmiş veri varsa)

```bash
python main.py --train
```

### Veriyi Yeniden İşle (cache'i zorla sil)

```bash
python main.py --force
```

### Beklenen Süreler

| Adım | Süre (yaklaşık) |
|------|-----------------|
| CSV okuma (~13.5K seri) | ~2-5 dk |
| tsfresh EfficientFC (4 chunk × ~3400 seri) | ~20-40 dk |
| Model eğitimi (3 model × 2 head) | ~5-15 dk |
| **Toplam** | **~30-60 dk** |

### Çıktılar

| Dosya | İçerik |
|-------|--------|
| `processed_data/X.npy` | Özellik matrisi (N × 777) |
| `processed_data/y_base.npy` | Base type etiketleri (int) |
| `processed_data/y_anomaly.npy` | Anomaly type etiketleri (int) |
| `processed_data/feature_names.json` | 777 özellik adları listesi |
| `results/training_results.json` | Tüm metrikler, model skorları, per-combination breakdown |

---

## Etiket İndeksleri

### Base Labels
| İndeks | Etiket               |
|--------|----------------------|
| 0      | stationary           |
| 1      | deterministic_trend  |
| 2      | stochastic_trend     |
| 3      | volatility           |

### Anomaly Labels
| İndeks | Etiket               |
|--------|----------------------|
| 0      | none                 |
| 1      | collective_anomaly   |
| 2      | contextual_anomaly   |
| 3      | mean_shift           |
| 4      | point_anomaly        |
| 5      | trend_shift          |
| 6      | variance_shift       |
