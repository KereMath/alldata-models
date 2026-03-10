"""
Generated Data altindaki tum ZIP dosyalarini ayni klasore cikarir.
Cikartilmis CSV'ler zaten varsa o ZIP'i atlar.
"""
import zipfile
from pathlib import Path
from tqdm import tqdm

BASE = Path(r"C:\Users\user\Desktop\Generated Data")

zips = sorted(BASE.rglob("*.zip"))
print(f"Toplam ZIP: {len(zips)}")

skipped  = 0
extracted = 0

for zpath in tqdm(zips, desc="Cikartiliyor"):
    dest = zpath.parent
    try:
        with zipfile.ZipFile(zpath) as zf:
            csvs_inside = [n for n in zf.namelist() if n.endswith(".csv")]
            # Hepsi zaten varsa atla
            all_exist = all((dest / n).exists() for n in csvs_inside)
            if all_exist:
                skipped += 1
                continue
            zf.extractall(dest)
            extracted += 1
    except Exception as e:
        print(f"\n[HATA] {zpath.name}: {e}")

print(f"\nTamamlandi: {extracted} ZIP cikartildi, {skipped} atlanildi (zaten mevcut).")
