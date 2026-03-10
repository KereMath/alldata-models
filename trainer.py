"""
alldata-models - Model Egitimi
Iki ayri multi-class classifier:
  1) base_type    -> stationary / deterministic_trend / stochastic_trend / volatility
  2) anomaly_type -> none / collective / contextual / mean_shift / point / trend_shift / variance
Full-match: her ikisi de dogru tahmin edilmis mi?
"""
import numpy as np
import json
import warnings
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR,
    BASE_LABELS, ANOMALY_LABELS,
    TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE,
)

warnings.filterwarnings('ignore')


def load_data():
    X         = np.load(PROCESSED_DATA_DIR / 'X.npy')
    y_base    = np.load(PROCESSED_DATA_DIR / 'y_base.npy')
    y_anomaly = np.load(PROCESSED_DATA_DIR / 'y_anomaly.npy')
    return X, y_base, y_anomaly


def _build_models():
    return {
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric='mlogloss', verbosity=0,
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(200, 100, 50), max_iter=600,
            early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def _train_task(task_name, labels, X_tr, y_tr, X_val, y_val, X_te, y_te):
    print(f"\n{'='*58}")
    print(f"  TASK: {task_name}  ({len(labels)} sinif)")
    print(f"  {labels}")
    print(f"{'='*58}")

    models     = _build_models()
    val_scores = {}

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred_val = model.predict(X_val)
        f1  = f1_score(y_val, pred_val, average='macro')
        acc = accuracy_score(y_val, pred_val)
        val_scores[name] = {'f1': f1, 'acc': acc, 'model': model}
        print(f"  {name:<12} Val F1={f1:.4f}  Acc={acc:.4f}")

    best_name  = max(val_scores, key=lambda k: val_scores[k]['f1'])
    best_model = val_scores[best_name]['model']

    pred_test = best_model.predict(X_te)
    test_f1   = f1_score(y_te, pred_test, average='macro')
    test_acc  = accuracy_score(y_te, pred_test)

    print(f"\n  >>> En iyi: {best_name}  |  Test F1={test_f1:.4f}  Acc={test_acc:.4f}")
    print(classification_report(y_te, pred_test, target_names=labels, digits=4))

    return {
        'best_model_name': best_name,
        'test_f1':   round(test_f1,  4),
        'test_acc':  round(test_acc, 4),
        'val_scores': {k: {'f1': round(v['f1'], 4), 'acc': round(v['acc'], 4)}
                       for k, v in val_scores.items()},
        'predictions': pred_test.tolist(),
        'true_labels': y_te.tolist(),
        'label_names': labels,
    }, best_model


def train_and_evaluate():
    print("\n" + "="*58)
    print("  alldata-models  -  Model Egitimi")
    print("="*58)

    X, y_base, y_anomaly = load_data()
    print(f"Veri boyutu: {X.shape}  |  NaN: {np.isnan(X).sum()}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train / Val / Test bolme
    val_ratio_adj = VALIDATION_SIZE / (1 - TEST_SIZE)

    (X_tmp, X_te,
     yb_tmp, yb_te,
     ya_tmp, ya_te) = train_test_split(
        X, y_base, y_anomaly,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_base,
    )
    (X_tr, X_val,
     yb_tr, yb_val,
     ya_tr, ya_val) = train_test_split(
        X_tmp, yb_tmp, ya_tmp,
        test_size=val_ratio_adj, random_state=RANDOM_STATE, stratify=yb_tmp,
    )

    print(f"Train: {len(X_tr)}  |  Val: {len(X_val)}  |  Test: {len(X_te)}")

    # Olcekleme
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_te)

    # Task 1: Base type (4 sinif)
    base_res, base_model = _train_task(
        "BASE TYPE", BASE_LABELS,
        X_tr_s, yb_tr, X_val_s, yb_val, X_te_s, yb_te,
    )

    # Task 2: Anomaly type (7 sinif)
    anomaly_res, anomaly_model = _train_task(
        "ANOMALY TYPE", ANOMALY_LABELS,
        X_tr_s, ya_tr, X_val_s, ya_val, X_te_s, ya_te,
    )

    # Full-match degerlendirmesi
    bp = np.array(base_res['predictions'])
    bt = np.array(base_res['true_labels'])
    ap = np.array(anomaly_res['predictions'])
    at = np.array(anomaly_res['true_labels'])
    n  = len(bt)

    base_ok    = (bp == bt)
    anomaly_ok = (ap == at)
    full_match   = int(np.sum(base_ok & anomaly_ok))
    base_only    = int(np.sum(base_ok & ~anomaly_ok))
    anomaly_only = int(np.sum(~base_ok & anomaly_ok))
    no_match     = int(np.sum(~base_ok & ~anomaly_ok))

    print(f"\n{'='*58}")
    print("  KOMBINASYON DEGERLENDIRMESI  (test seti)")
    print(f"{'='*58}")
    print(f"  Full Match   (ikisi de dogru) : {full_match:4d} / {n}  ({100*full_match/n:.2f}%)")
    print(f"  Sadece Base  dogru            : {base_only:4d} / {n}  ({100*base_only/n:.2f}%)")
    print(f"  Sadece Anomali dogru          : {anomaly_only:4d} / {n}  ({100*anomaly_only/n:.2f}%)")
    print(f"  No Match     (ikisi de yanlis): {no_match:4d} / {n}  ({100*no_match/n:.2f}%)")
    print(f"\n  Base Accuracy   : {100*np.mean(base_ok):.2f}%")
    print(f"  Anomaly Accuracy: {100*np.mean(anomaly_ok):.2f}%")

    # Kategori bazinda detay
    print(f"\n{'='*58}")
    print("  Kategori Bazinda Full-Match Orani")
    print(f"{'='*58}")

    combo_stats = defaultdict(lambda: {'total': 0, 'full': 0})
    for i in range(n):
        key = f"{BASE_LABELS[bt[i]]} + {ANOMALY_LABELS[at[i]]}"
        combo_stats[key]['total'] += 1
        if base_ok[i] and anomaly_ok[i]:
            combo_stats[key]['full'] += 1

    for key in sorted(combo_stats.keys()):
        s    = combo_stats[key]
        rate = 100 * s['full'] / s['total'] if s['total'] > 0 else 0
        print(f"  {key:<45}  {s['full']:3d}/{s['total']:3d}  ({rate:.1f}%)")

    # Kaydet
    RESULTS_DIR.mkdir(exist_ok=True)
    combo_eval = {
        'full_match':       full_match,
        'full_match_pct':   round(100 * full_match / n, 2),
        'base_only':        base_only,
        'anomaly_only':     anomaly_only,
        'no_match':         no_match,
        'total':            n,
        'base_accuracy':    round(float(np.mean(base_ok)),    4),
        'anomaly_accuracy': round(float(np.mean(anomaly_ok)), 4),
        'per_combination':  {
            k: {'full': v['full'], 'total': v['total'],
                'rate': round(100 * v['full'] / v['total'], 1)}
            for k, v in combo_stats.items()
        },
    }

    all_results = {
        'base_type':        {k: v for k, v in base_res.items()    if k != 'predictions'},
        'anomaly_type':     {k: v for k, v in anomaly_res.items() if k != 'predictions'},
        'combination_eval': combo_eval,
    }

    out_file = RESULTS_DIR / 'training_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSonuclar kaydedildi: {out_file}")

    return all_results


if __name__ == "__main__":
    train_and_evaluate()
