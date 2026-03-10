"""
alldata-models - Ana Calistirma Scripti
Kullanim:
    python main.py           # isleme + egitim
    python main.py --force   # islemi yeniden yap (cache sil)
    python main.py --train   # sadece egitim (isleme atla)
"""
import sys
import numpy as np
from pathlib import Path
from config import PROCESSED_DATA_DIR


def main():
    force      = '--force' in sys.argv
    train_only = '--train' in sys.argv
    data_ready = (PROCESSED_DATA_DIR / 'X.npy').exists()

    if train_only and data_ready:
        X = np.load(PROCESSED_DATA_DIR / 'X.npy')
        print(f">> Adim 1/2: --train modu, mevcut veri kullaniliyor -> {X.shape}")
    elif force or not data_ready:
        print(">> Adim 1/2: Veri isleniyor (tsfresh)...")
        from processor import process_and_save
        process_and_save()
    else:
        X = np.load(PROCESSED_DATA_DIR / 'X.npy')
        print(f">> Adim 1/2: Onceden islenmis veri bulundu -> {X.shape}")

    print("\n>> Adim 2/2: Model egitimi basliyor...")
    from trainer import train_and_evaluate
    train_and_evaluate()

    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
