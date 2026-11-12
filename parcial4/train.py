"""Entrena 2 representaciones visuales x 3 esquemas (desde cero, transfer learning,
fine tuning), compara métricas y guarda el mejor modelo para la aplicación.

Uso:  python train.py            (o  python train.py --quick  para una prueba corta)
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from src.models import SCHEMES, build_model
from src.preprocessing import CLASSES, FEATURES, Preprocessor, engineer, labels, load_clean
from src.representations import REPRESENTATIONS, make_images

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "credit_train.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
SEED = 42
BATCH = 1024
EPOCHS = {"scratch": 10, "transfer": 8, "finetune": 8}
LR = {"scratch": 2e-3, "transfer": 3e-3, "finetune": 3e-4}
PATIENCE = 4


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_co": precision_score(y, pred, zero_division=0),
        "recall_co": recall_score(y, pred),
        "f1_co": f1_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, prob),
    }


@torch.no_grad()
def predict(model, x, rep, order, device):
    model.eval()
    out = []
    for i in range(0, len(x), 1024):
        imgs = make_images(x[i:i + 1024], rep, order)
        out.append(torch.softmax(model(imgs), 1)[:, 1].float().cpu())
    return torch.cat(out).numpy()


def train_one(model, rep, scheme, data, order, device, weights, epochs):
    xtr, ytr, xva, yva = data
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR[scheme], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler(enabled=False)
    best, best_state, wait, hist = -1, None, 0, []
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for ep in range(1, epochs + 1):
        model.train()
        if scheme == "transfer":  # BatchNorm congelada en el backbone
            model.net.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm2d) else None)
        t0, tot, n = time.time(), 0.0, 0
        perm = torch.randperm(len(xtr), generator=g).to(device)
        for i in range(0, len(xtr), BATCH):
            idx = perm[i:i + BATCH]
            imgs = make_images(xtr[idx], rep, order)
            with torch.autocast(device.type, enabled=False):
                loss = loss_fn(model(imgs), ytr[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(idx)
            n += len(idx)
        sched.step()
        prob = predict(model, xva, rep, order, device)
        yv = yva.cpu().numpy()
        with torch.no_grad():
            vl = loss_fn(torch.log(torch.tensor(np.stack([1 - prob, prob], 1)).clamp_min(1e-7)).to(device),
                         yva).item()
        m = metrics(yv, prob)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_loss": vl, **{f"val_{k}": v for k, v in m.items()}})
        print(f"  [{rep}/{scheme}] ep {ep:02d} loss {tot / n:.4f} val_loss {vl:.4f} "
              f"AUC {m['roc_auc']:.4f} F1m {m['f1_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)
        if m["roc_auc"] > best + 1e-4:
            best, best_state, wait = m["roc_auc"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("  parada temprana")
                break
    model.load_state_dict(best_state)
    return model, hist


def plot_all(results, histories, probs_test, y_test, best_key):
    FIGS.mkdir(parents=True, exist_ok=True)
    # Curvas de entrenamiento
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    for c, scheme in enumerate(SCHEMES):
        for rep, ls in zip(REPRESENTATIONS, ["-", "--"]):
            h = pd.DataFrame(histories[f"{rep}__{scheme}"])
            axes[0, c].plot(h.epoch, h.train_loss, ls, label=f"{rep} train")
            axes[0, c].plot(h.epoch, h.val_loss, ls, alpha=.6, label=f"{rep} val")
            axes[1, c].plot(h.epoch, h.val_roc_auc, ls, label=rep)
        axes[0, c].set_title(SCHEMES[scheme])
        axes[1, c].set_xlabel("época")
    axes[0, 0].set_ylabel("pérdida")
    axes[1, 0].set_ylabel("ROC-AUC validación")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "curvas_entrenamiento.png", dpi=120)
    plt.close(fig)

    # Curvas ROC en prueba
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, prob in probs_test.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, lw=2.5 if key == best_key else 1, label=f"{key} (AUC {roc_auc_score(y_test, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC — conjunto de prueba")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "roc_prueba.png", dpi=120)
    plt.close(fig)

    # Comparación de métricas
    df = results[results.split == "test"].set_index("modelo")
    cols = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc"]
    ax = df[cols].plot.bar(figsize=(12, 5), rot=30)
    ax.set_ylim(0.4, 0.85)
    ax.set_title("Métricas en prueba por modelo")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGS / "comparacion_metricas.png", dpi=120)
    plt.close(ax.figure)

    # Matriz de confusión del mejor modelo
    cm = confusion_matrix(y_test, (probs_test[best_key] >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {best_key}")
    fig.tight_layout()
    fig.savefig(FIGS / "matriz_confusion_mejor.png", dpi=120)
    plt.close(fig)


def main(quick=False):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)
    df = load_clean(DATA)
    feats, y = engineer(df), labels(df)
    print(f"Registros limpios: {len(df)} | Charged Off: {y.mean():.1%}")
    idx = np.arange(len(df))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)
    if quick:
        i_tr, i_va, i_te = i_tr[:4000], i_va[:2000], i_te[:2000]
    prep = Preprocessor().fit(feats.iloc[i_tr])
    X = {k: prep.transform(feats.iloc[v]) for k, v in [("train", i_tr), ("val", i_va), ("test", i_te)]}
    Y = {"train": y[i_tr], "val": y[i_va], "test": y[i_te]}
    order = prep.order_
    print("Tamaños:", {k: len(v) for k, v in Y.items()})

    rows, histories, probs_test = [], {}, {}

    # Línea base tabular (referencia, no visual)
    for name, clf in [("baseline_logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
                      ("baseline_hgb", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))]:
        clf.fit(X["train"], Y["train"])
        for split in ["val", "test"]:
            p = clf.predict_proba(X[split])[:, 1]
            rows.append({"modelo": name, "representacion": "tabular", "esquema": "baseline", "split": split,
                         **metrics(Y[split], p)})
            if split == "test":
                probs_test[name] = p
        print(name, "val AUC", rows[-2]["roc_auc"])

    t = {k: torch.tensor(v, device=device) for k, v in X.items()}
    ty = {k: torch.tensor(v, device=device, dtype=torch.long) for k, v in Y.items()}
    w = len(Y["train"]) / (2 * np.bincount(Y["train"]))
    weights = torch.tensor(w, dtype=torch.float32, device=device)

    trained = {}
    for rep in REPRESENTATIONS:
        for scheme in SCHEMES:
            set_seed()
            key = f"{rep}__{scheme}"
            print(f"\n=== {key} ===", flush=True)
            if scheme == "finetune":  # parte del modelo de transfer learning ya entrenado
                model = copy.deepcopy(trained[f"{rep}__transfer"])
                model.unfreeze_top()
            else:
                model = build_model(scheme)
            ep = 2 if quick else EPOCHS[scheme]
            t0 = time.time()
            model, hist = train_one(model, rep, scheme, (t["train"], ty["train"], t["val"], ty["val"]),
                                    order, device, weights, ep)
            secs = time.time() - t0
            trained[key] = model
            histories[key] = hist
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            for split in ["val", "test"]:
                p = predict(model, t[split], rep, order, device)
                rows.append({"modelo": key, "representacion": rep, "esquema": scheme, "split": split,
                             "params_entrenables": n_params, "segundos": round(secs, 1), "epocas": len(hist),
                             **metrics(Y[split], p)})
                if split == "test":
                    probs_test[key] = p

    results = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / "metricas.csv", index=False)
    (RESULTS / "historial.json").write_text(json.dumps(histories, indent=1), encoding="utf-8")

    visual = results[(results.split == "val") & (results.esquema != "baseline")]
    visual = visual.sort_values(["roc_auc", "f1_macro"], ascending=False)
    best_key = visual.iloc[0]["modelo"]
    rep, scheme = best_key.split("__")
    print("\nMejor modelo (ROC-AUC validación):", best_key)
    print(results[results.split == "test"].drop(columns="split").round(4).to_string(index=False))
    plot_all(results, histories, probs_test, Y["test"], best_key)

    if not quick:
        MODELS.mkdir(exist_ok=True)
        best = trained[best_key].cpu()
        torch.save(best.state_dict(), MODELS / "best_model.pt")
        joblib.dump(prep, MODELS / "preprocessor.joblib")
        test_row = results[(results.modelo == best_key) & (results.split == "test")].iloc[0]
        info = {"key": best_key, "representation": rep, "scheme": scheme, "features": FEATURES,
                "classes": CLASSES, "order": prep.order_.tolist(), "selection": "max ROC-AUC en validación",
                "test_metrics": {k: float(test_row[k]) for k in ["accuracy", "balanced_accuracy", "f1_macro",
                                                                  "f1_co", "recall_co", "precision_co", "roc_auc"]}}
        (MODELS / "model_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Guardado en", MODELS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
