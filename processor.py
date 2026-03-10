"""
alldata-models - Veri Isleme
Generated Data altindaki tum kategorilerden dengeli ornek alir,
tsfresh ile ozellik cikarir, X.npy / y_base.npy / y_anomaly.npy kaydeder.
Chunk processing ile buyuk veri setlerinde bellek sorunu olmaz.
"""
import os
import gc
import random
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
from tsfresh import extract_features as tsfresh_extract
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters

from config import (
    GENERATED_DATA_DIR, COMBINATIONS_DIR, PROCESSED_DATA_DIR,
    SINGLE_CATEGORIES, COMBO_LEAVES,
    BASE_LABELS, ANOMALY_LABELS,
    SAMPLES_PER_CATEGORY, MIN_SERIES_LENGTH, RANDOM_STATE,
)

warnings.filterwarnings('ignore')
random.seed(RANDOM_STATE)

CHUNK_SIZE = 4000   # Her chunk'ta max seri sayisi (bellek limiti icin)


# ---------------------------------------------------------------
# Yaprak klasor bulucu
# ---------------------------------------------------------------
def get_leaf_dirs(directory: Path) -> List[Path]:
    """
    Dogrudan CSV iceren (metadata haric) klasorleri dondurur.
    CSV bulunan klasore indikten sonra daha derin gitme.
    Boylece short/medium/long gibi son seviyeler otomatik bulunur.
    """
    result = []

    def _walk(d: Path):
        try:
            items = list(d.iterdir())
        except PermissionError:
            return
        has_csvs = any(
            f.is_file() and f.suffix == '.csv' and f.name != 'metadata.csv'
            for f in items
        )
        if has_csvs:
            result.append(d)
        else:
            for item in items:
                if item.is_dir():
                    _walk(item)

    _walk(directory)
    return sorted(result)


def sample_from_leaves(leaves: List[Path], total: int) -> List[Path]:
    """
    Her yapraktan esit sayida CSV sec (floor(total / n_yaprak)).
    short/medium/long dengesi otomatik saglanir.
    """
    if not leaves:
        return []
    per_leaf = max(1, total // len(leaves))
    result = []
    for leaf in sorted(leaves):
        csvs = sorted([
            p for p in leaf.iterdir()
            if p.is_file() and p.suffix == '.csv' and p.name != 'metadata.csv'
        ])
        if len(csvs) > per_leaf:
            csvs = random.sample(csvs, per_leaf)
        result.extend(csvs)
    return result


# ---------------------------------------------------------------
# CSV okuma
# ---------------------------------------------------------------
def read_series(csv_path: Path) -> np.ndarray:
    """CSV'den zaman serisini oku. Kolon adi farkliliklarina karsi savunmali."""
    try:
        df = pd.read_csv(csv_path)
        for col in ('data', 'value', 'values', 'y'):
            if col in df.columns:
                return df[col].dropna().values.astype(float)
        # Ilk numerik kolon
        num_cols = df.select_dtypes(include=[float, int]).columns
        if len(num_cols) > 0:
            return df[num_cols[0]].dropna().values.astype(float)
    except Exception:
        pass
    return np.array([])


# ---------------------------------------------------------------
# Tum kategorilerden ornek toplama
# ---------------------------------------------------------------
def collect_all_samples() -> List[Tuple[Path, str, str]]:
    """
    Her kategoriden dengeli ornek topla.
    Returns: [(csv_path, base_label, anomaly_label), ...]
    """
    all_items: List[Tuple[Path, str, str]] = []
    category_stats = []

    print("\n--- Tekli kategoriler ---")
    for folder_name, base_lbl, anomaly_lbl in SINGLE_CATEGORIES:
        root = GENERATED_DATA_DIR / folder_name
        if not root.exists():
            print(f"  [WARN] Bulunamadi: {root}")
            continue

        leaves = get_leaf_dirs(root)
        csvs   = sample_from_leaves(leaves, SAMPLES_PER_CATEGORY)

        for c in csvs:
            all_items.append((c, base_lbl, anomaly_lbl))

        per_leaf = SAMPLES_PER_CATEGORY // len(leaves) if leaves else 0
        print(f"  {folder_name:<25}  {len(leaves):2d} yaprak  "
              f"x {per_leaf:3d}/yaprak  = {len(csvs):4d} ornek")
        category_stats.append((folder_name, len(csvs)))

    print("\n--- Kombinasyon yapraklari ---")
    for path_parts, base_lbl, anomaly_lbl in COMBO_LEAVES:
        leaf_dir = COMBINATIONS_DIR
        for part in path_parts:
            leaf_dir = leaf_dir / part

        if not leaf_dir.exists():
            print(f"  [WARN] Bulunamadi: {leaf_dir}")
            continue

        leaves = get_leaf_dirs(leaf_dir)
        csvs   = sample_from_leaves(leaves, SAMPLES_PER_CATEGORY)

        combo_label = " / ".join(path_parts[-2:] if len(path_parts) > 1 else path_parts)
        n_leaves = len(leaves)
        print(f"  {combo_label:<55}  {n_leaves} yaprak  {len(csvs):4d} ornek")
        category_stats.append((combo_label, len(csvs)))

        for c in csvs:
            all_items.append((c, base_lbl, anomaly_lbl))

    total_cats    = len(category_stats)
    total_samples = sum(n for _, n in category_stats)
    print(f"\nToplam: {total_cats} kategori, {total_samples} ornek")
    return all_items


# ---------------------------------------------------------------
# Chunk bazli tsfresh
# ---------------------------------------------------------------
def _extract_chunk(series_list: List[pd.DataFrame], id_offset: int) -> pd.DataFrame:
    """
    Verilen seri listesini tek chunk olarak tsfresh'e gonder.
    id_offset: bu chunk'taki ilk serinin global ID'si.
    """
    # ID'leri 0'dan baslat (tsfresh icin), sonra concat'a gerek yok
    local_dfs = []
    for local_id, df in enumerate(series_list):
        local_dfs.append(pd.DataFrame({
            'id':    local_id,
            'time':  df['time'].values,
            'value': df['value'].values,
        }))
    combined = pd.concat(local_dfs, ignore_index=True)

    X_chunk = tsfresh_extract(
        combined,
        column_id='id',
        column_sort='time',
        column_value='value',
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=True,
        n_jobs=4,
    )
    impute(X_chunk)
    return X_chunk


# ---------------------------------------------------------------
# Ana isleme fonksiyonu
# ---------------------------------------------------------------
def process_and_save():
    print("=" * 62)
    print("  alldata-models  -  Veri Isleme (tsfresh EfficientFC, chunked)")
    print("=" * 62)

    items = collect_all_samples()
    random.shuffle(items)

    # CSV'leri oku
    series_list: List[pd.DataFrame] = []
    labels:      List[Tuple[str, str]] = []
    failed = 0

    print("\nCSV'ler okunuyor...")
    for csv_path, base_lbl, anomaly_lbl in tqdm(items, desc="Okuma"):
        data = read_series(csv_path)
        if len(data) < MIN_SERIES_LENGTH:
            failed += 1
            continue
        series_list.append(pd.DataFrame({
            'time':  np.arange(len(data)),
            'value': data,
        }))
        labels.append((base_lbl, anomaly_lbl))

    sid = len(series_list)
    print(f"Basarili: {sid}  |  Basarisiz/atlanan: {failed}")
    if not series_list:
        print("HATA: Hic ornek islenmedi!")
        return

    # tsfresh - chunk bazli
    n_chunks = (sid + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"\ntsfresh EfficientFC - {sid} seri, {n_chunks} chunk x {CHUNK_SIZE}")
    print("Her chunk ~5-10 dk surebilir...\n")

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    chunk_files = []
    col_names   = None

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, sid)
        print(f"  Chunk {chunk_idx+1}/{n_chunks}: seri {start}-{end} ({end-start} adet)...")
        chunk_df = _extract_chunk(series_list[start:end], start)
        if col_names is None:
            col_names = chunk_df.columns.tolist()
        chunk_path = PROCESSED_DATA_DIR / f'chunk_{chunk_idx:03d}.npy'
        np.save(chunk_path, chunk_df[col_names].values)
        chunk_files.append(chunk_path)
        print(f"  Chunk {chunk_idx+1} kaydedildi: {chunk_df.shape} -> {chunk_path.name}")
        del chunk_df
        gc.collect()

    print(f"\nChunk'lar birlestiriliyor ({len(chunk_files)} dosya)...")
    X = np.vstack([np.load(p) for p in chunk_files])
    # Gecici chunk dosyalarini sil
    for p in chunk_files:
        p.unlink()
    y_base    = np.array([BASE_LABELS.index(l[0])    for l in labels], dtype=int)
    y_anomaly = np.array([ANOMALY_LABELS.index(l[1]) for l in labels], dtype=int)

    # Kaydet
    np.save(PROCESSED_DATA_DIR / 'X.npy',         X)
    np.save(PROCESSED_DATA_DIR / 'y_base.npy',    y_base)
    np.save(PROCESSED_DATA_DIR / 'y_anomaly.npy', y_anomaly)
    with open(PROCESSED_DATA_DIR / 'feature_names.json', 'w') as f:
        json.dump(col_names, f)

    print(f"\nKaydedildi -> {PROCESSED_DATA_DIR}")
    print(f"Ozellik matrisi : {X.shape}")
    print(f"Base dagilimi   : {dict(zip(BASE_LABELS,    np.bincount(y_base,    minlength=len(BASE_LABELS))))}")
    print(f"Anomaly dagilimi: {dict(zip(ANOMALY_LABELS, np.bincount(y_anomaly, minlength=len(ANOMALY_LABELS))))}")

    return X, y_base, y_anomaly


if __name__ == "__main__":
    process_and_save()
