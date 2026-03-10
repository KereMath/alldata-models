# alldata-models — Calisma Plani

## Amac

`C:\Users\user\Desktop\Generated Data\` altindaki TUM veri tiplerini kullanarak,
bir zaman serisinin hem **base tipini** hem **anomali tipini** ayni anda tahmin eden
iki baslikli (two-head) bir multi-class model egitmek.

`combinations-thesis` ile ayni mimari, ama cok daha genis veri kapsamı.

---

## Veri Kaynaği

Dizin: `C:\Users\user\Desktop\Generated Data\`

### Ust-Level Klasorler (11 adet)

| Klasor | Icerik | Alt-klasor Yapisi |
|--------|--------|-------------------|
| `stationary` | Duragan seriler (anomalisiz) | 12 yaprak: (ar/arma/ma/wn) x (short/medium/long) |
| `deterministic_trend` | Deterministik trend (anomalisiz) | 12 yaprak: (AR/ARMA/MA/White_Noise) x (short/med/long) |
| `Stochastic Trend` | Birim koklu surecler (anomalisiz) | 15 yaprak: (ARI/ARIMA/IMA/RW/RWD) x (short/med/long) |
| `Volatility` | Heteroskedastik (anomalisiz) | 12 yaprak: (APARCH/ARCH/EGARCH/GARCH) x (short/med/long) |
| `collective_anomaly` | Stok. taban + kolektif anomali | 12 yaprak: (AR/ARMA/MA/WN) x (short/med/long) |
| `contextual_anomaly` | Stok. taban + baglamssal anomali | 12 yaprak |
| `mean_shift` | Stok. taban + ortalama kayisi | 12 yaprak |
| `point_anomaly` | Stok. taban + nokta anomalisi | 12 yaprak |
| `trend_shift` | Stok. taban + trend kayisi | 12 yaprak |
| `variance_shift` | Stok. taban + varyans kayisi | 12 yaprak |
| `Combinations` | Taban + anomali kombinasyonlari | 28 yaprak (asagida detay) |

### Combinations Yapraksari (28 adet)

Deterministik taban (5 tip) x anomali (4-5 tip) = 21 kombinasyon
Stokastik taban + anomali = 3 kombinasyon
Volatility taban + anomali = 4 kombinasyon
Toplam: **28 kombinasyon yapragi**

| Taban | Anomali Tipleri |
|-------|----------------|
| Cubic / Damped / Exponential / Linear / Quadratic | collective, mean_shift, point, variance (+ trend_shift sadece Linear) |
| Stochastic Trend | collective, mean_shift, point |
| Volatility | collective, mean_shift, point, variance |

---

## Ornekleme Plani

**Ilke:** Her kategoriden esit sayida ornek AL. Her kategori icindeki yapraklardan
da esit sayida al.

**Darbogazı:** En kucuk yaprak = 1000 CSV (bazi Combinations yapraklari)

### Karar: Her kategoriden 1000 ornek

| Tip | Yaprak Sayisi | Yaprak Basina | Kategori Toplami |
|-----|---------------|---------------|-----------------|
| stationary | 12 | 83 | 1,000 |
| deterministic_trend | 12 | 83 | 1,000 |
| Stochastic Trend | 15 | 67 | 1,000 |
| Volatility | 12 | 83 | 1,000 |
| collective_anomaly | 12 | 83 | 1,000 |
| contextual_anomaly | 12 | 83 | 1,000 |
| mean_shift | 12 | 83 | 1,000 |
| point_anomaly | 12 | 83 | 1,000 |
| trend_shift | 12 | 83 | 1,000 |
| variance_shift | 12 | 83 | 1,000 |
| Her Combinations yapragi (28 adet) | 1 | 1,000 | 1,000 x 28 |

**TOPLAM: 38 kategori x 1,000 = 38,000 ornek**

---

## Label Semasi (2 Head)

### Head 1 — base_type (4 sinif)

| Sinif | Hangi Klasorler |
|-------|----------------|
| `stationary` | stationary/, collective_anomaly/, contextual_anomaly/, mean_shift/, point_anomaly/, trend_shift/, variance_shift/ |
| `deterministic_trend` | deterministic_trend/, Combinations/Cubic Base/, Combinations/Damped Base/, Combinations/Exponential Base/, Combinations/Linear Base/, Combinations/Quadratic Base/ |
| `stochastic_trend` | Stochastic Trend/, Combinations/Stochastic Trend + .../ |
| `volatility` | Volatility/, Combinations/Volatility + .../ |

> Not: Tekli anomali klasorlerinin altindaki AR/ARMA/MA/White_Noise
> duragan stokastik surecler oldugu icin base_type = `stationary`.

### Head 2 — anomaly_type (7 sinif)

| Sinif | Hangi Klasorler |
|-------|----------------|
| `none` | stationary/, deterministic_trend/, Stochastic Trend/, Volatility/ |
| `collective_anomaly` | collective_anomaly/, Combinations/*/collective/ |
| `contextual_anomaly` | contextual_anomaly/ |
| `mean_shift` | mean_shift/, Combinations/*/mean_shift/ |
| `point_anomaly` | point_anomaly/, Combinations/*/point/ |
| `trend_shift` | trend_shift/, Combinations/*/trend_shift/ |
| `variance_shift` | variance_shift/, Combinations/*/variance/ |

### Full-Match Metrik

Her iki head de dogru tahmin ettiyse = Full Match.
Hedef: combinations-thesis'teki %91.7'yi askin bir oran.

---

## Egitim Mekanizmasi (combinations-thesis'ten ayni mimari)

### combinations-thesis'te nasil egitimlendi?

**processor.py:**
- Klasorleri tara, klasor adından (base_type, anomaly_type) labellarini cikar
- Her CSV'den `data` kolonunu oku
- `tsfresh.EfficientFCParameters()` ile toplu ozellik cikarimi (uzun suruyor)
- Her kombinasyon icin MAX_SAMPLES_PER_COMBO=800 ile dengele
- X.npy, y_base.npy, y_anomaly.npy olarak kaydet

**trainer.py:**
- X.npy yukle, StandardScaler uygula
- Train/Val/Test bolme: %70 / %10 / %20 (stratify=y_base)
- 3 model dene: LightGBM, XGBoost, MLP
- Val F1 (macro)'e gore en iyiyi sec
- Test seti uzerinde: F1, Accuracy, classification_report
- Full-Match degerlendirmesi: ikisi de dogru mu?
- Kombinasyon bazinda detay raporu
- Sonuclari results/training_results.json'a kaydet

**Model parametreleri (aynen alinacak):**
```
LightGBM:  n_estimators=500, lr=0.05, max_depth=8, num_leaves=63, subsample=0.8
XGBoost:   n_estimators=500, lr=0.05, max_depth=8, subsample=0.8
MLP:       hidden=(200,100,50), max_iter=600, early_stopping=True
```

### alldata-models icin degisecekler

| Parametre | combinations-thesis | alldata-models |
|-----------|--------------------|--------------------|
| Veri kaynaği | Combinations/ | Generated Data/ (tum klasorler) |
| BASE_LABELS | 5 (cubic/damped/exp/linear/quad) | 4 (stationary/det_trend/stoch_trend/volatility) |
| ANOMALY_LABELS | 5 (anomali tipleri) | 7 (none + 6 anomali tipi) |
| Kategori sayisi | 21 kombinasyon | 38 kategori |
| Max ornek/kategori | 800 | 1,000 |
| Toplam ornek | ~16,800 | ~38,000 |
| tsfresh parametresi | EfficientFCParameters | EfficientFCParameters (ayni) |

---

## Dosya Yapisi (olusturulacaklar)

```
alldata-models/
├── plan.md               <- bu dosya
├── config.py             <- yollar, labellar, parametreler
├── processor.py          <- veri tarama, label esleme, tsfresh
├── trainer.py            <- egitim, deger., kayit (ayni mantik)
├── main.py               <- orchestrator
├── processed_data/       <- X.npy, y_base.npy, y_anomaly.npy (otomatik)
├── results/              <- training_results.json (otomatik)
└── trained_models/       <- (isteğe bagli kayit)
```

---

## Beklenen Sure

| Adim | Sure |
|------|------|
| Veri tarama + tsfresh ozellik cikarimi (38K seri) | ~20-30 dk |
| Model egitimi (LightGBM + XGBoost + MLP x 2 head) | ~5-10 dk |
| **Toplam** | **~30-40 dk** |

> Not: tsfresh EfficientFCParameters 38K seri icin agir olabilir.
> Gerekirse MinimalFCParameters ile hizlandiririz.

---

## Sonraki Adim

Plan onaylandiktan sonra:
1. `config.py` yaz
2. `processor.py` yaz (klasor → label mapping)
3. `trainer.py` yaz (combinations-thesis'ten uyarla)
4. `main.py` yaz
5. Calistir ve sonuclari karsilastir
