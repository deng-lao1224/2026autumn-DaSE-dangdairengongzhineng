
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data-Project1" / "data-Project1"
OUT_DIR = Path(__file__).resolve().parent / "results"
MAX_FEATURES = 12000
BATCH_SIZE = 256
EPOCHS = 24


def load_raw():
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")
    texts = train_df["text"].astype(str).tolist()
    labels = train_df["target"].to_numpy()
    test_texts = test_df["text"].astype(str).tolist()
    return texts, labels, test_texts


def strip_mail_headers(text: str, keep_subject: bool) -> str:
    """去掉信头。keep_subject=True 时保留 Subject，因为主题句本身是分类线索。"""
    lines = text.splitlines()
    body = []
    subjects = []
    in_header = True
    for line in lines:
        if in_header:
            if line.strip() == "":
                in_header = False
                continue
            if keep_subject and line.lower().startswith("subject:"):
                subjects.append(line.split(":", 1)[1])
            continue
        if line.lstrip().startswith(">"):
            continue
        body.append(line)
    cleaned = "\n".join(subjects + body).strip()
    return cleaned if cleaned else text


def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_text(text: str, header_mode: str, drop_stop: bool) -> str:
    if header_mode == "drop_all":
        text = strip_mail_headers(text, keep_subject=False)
    elif header_mode == "keep_subject":
        text = strip_mail_headers(text, keep_subject=True)
    text = basic_clean(text)
    if drop_stop:
        tokens = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]
        text = " ".join(tokens)
    return text


def make_vectorizer(kind: str):
    token = r"(?u)\b[a-zA-Z]{2,}\b"
    common = dict(
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.9,
        token_pattern=token,
        lowercase=True,
    )
    if kind == "bow":
        return CountVectorizer(**common)
    if kind == "tfidf_l2":
        return TfidfVectorizer(norm="l2", sublinear_tf=False, ngram_range=(1, 1), **common)
    if kind == "tfidf_l1":
        return TfidfVectorizer(norm="l1", sublinear_tf=False, ngram_range=(1, 1), **common)
    if kind == "tfidf_sublinear":
        return TfidfVectorizer(norm="l2", sublinear_tf=True, ngram_range=(1, 1), **common)
    if kind == "tfidf_bigram":
        return TfidfVectorizer(norm="l2", sublinear_tf=True, ngram_range=(1, 2), **common)
    raise ValueError(kind)
#5种向量化的方式，配上不同的清洗方式，得到8种向量化，分别计算验证集的macro-F1，选择最好的，送进模型

def batch_loss(clf, X, y, classes, loss_name: str) -> float:
    if loss_name == "log_loss":
        proba = clf.predict_proba(X)
        return float(log_loss(y, proba, labels=classes))
    scores = clf.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    total = 0.0
    for i, c in enumerate(classes):
        y_bin = np.where(y == c, 1.0, -1.0)
        total += np.maximum(0.0, 1.0 - y_bin * scores[:, i]).mean()
    return float(total / len(classes))


def snapshot(clf):
    return clf.coef_.copy(), clf.intercept_.copy()


def restore(clf, state):
    coef, intercept = state
    clf.coef_ = coef
    clf.intercept_ = intercept


def train_sgd(X_tr, y_tr, X_va, y_va, loss_name, alpha, epochs=EPOCHS):
    classes = np.unique(y_tr)
    clf = SGDClassifier(
        loss=loss_name,
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        random_state=SEED,
        average=False,
        tol=None,
    )
    rng = np.random.RandomState(SEED)
    n = X_tr.shape[0]
    steps = []
    epochs_log = []
    best_state = None
    best_f1 = -1.0
    best_epoch = -1
    step_id = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_batch_losses = []#新建一个空列表，只存放这一轮里每一批的损失。下一轮会重新建一个。
        for start in range(0, n, BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            if step_id == 0:
                clf.partial_fit(X_tr[batch], y_tr[batch], classes=classes)#partial_fit 是 sklearn 的函数，意思是用这一批数据更新一次已有权重，而不是把前面学过的内容清掉重来。
            else:
                clf.partial_fit(X_tr[batch], y_tr[batch])
            loss = batch_loss(clf, X_tr[batch], y_tr[batch], classes, loss_name)
            step_id += 1
            epoch_batch_losses.append(loss)
            steps.append(
                {
                    "step": step_id,
                    "epoch": epoch,
                    "batch_loss": loss,
                }
            )#内层的 for start 到这里结束。这一轮的所有批次都学完了，接下来用验证集检查这一轮好不好。
        y_hat = clf.predict(X_va)
        val_acc = float(accuracy_score(y_va, y_hat))
        val_f1 = float(f1_score(y_va, y_hat, average="macro"))
        train_loss = float(np.mean(epoch_batch_losses))
        val_loss = batch_loss(clf, X_va, y_va, classes, loss_name)
        #再用同一个损失公式，对整份验证集算一次损失。验证集不参与 partial_fit。如下
        epochs_log.append(
            {
                "epoch": epoch,
                "train_batch_loss_mean": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = snapshot(clf)
    restore(clf, best_state)
    y_hat = clf.predict(X_va)
    summary = {
        "val_accuracy": float(accuracy_score(y_va, y_hat)),
        "val_macro_f1": float(f1_score(y_va, y_hat, average="macro")),
        "best_epoch": int(best_epoch),
        "n_steps": int(step_id),
    }
    return clf, steps, epochs_log, summary


def train_mlp(X_tr, y_tr, X_va, y_va, hidden, alpha, epochs=12):
    classes = np.unique(y_tr)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=BATCH_SIZE,
        learning_rate_init=0.001,
        max_iter=1,
        warm_start=False,
        random_state=SEED,
        early_stopping=False,
    )
    steps = []
    epochs_log = []
    best = None
    best_f1 = -1.0
    best_epoch = -1
    rng = np.random.RandomState(SEED)
    n = X_tr.shape[0]
    step_id = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_losses = []
        for start in range(0, n, BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            Xb = X_tr[batch]
            yb = y_tr[batch]
            if hasattr(Xb, "toarray"):
                Xb = Xb.toarray()
            clf.partial_fit(Xb, yb, classes=classes)
            proba = clf.predict_proba(Xb)
            loss = float(log_loss(yb, proba, labels=classes))
            step_id += 1
            epoch_losses.append(loss)
            steps.append({"step": step_id, "epoch": epoch, "batch_loss": loss})
        y_hat = clf.predict(X_va)
        val_f1 = float(f1_score(y_va, y_hat, average="macro"))
        val_acc = float(accuracy_score(y_va, y_hat))
        Xv = X_va.toarray() if hasattr(X_va, "toarray") else X_va
        val_loss = float(log_loss(y_va, clf.predict_proba(Xv), labels=classes))
        epochs_log.append(
            {
                "epoch": epoch,
                "train_batch_loss_mean": float(np.mean(epoch_losses)),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best = (
                [c.copy() for c in clf.coefs_],
                [b.copy() for b in clf.intercepts_],
            )
    clf.coefs_ = best[0]
    clf.intercepts_ = best[1]
    y_hat = clf.predict(X_va)
    summary = {
        "val_accuracy": float(accuracy_score(y_va, y_hat)),
        "val_macro_f1": float(f1_score(y_va, y_hat, average="macro")),
        "best_epoch": int(best_epoch),
        "n_steps": int(step_id),
    }
    return clf, steps, epochs_log, summary


def fit_nb(X_tr, y_tr, X_va, y_va, alpha):
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_va)
    return clf, {
        "val_accuracy": float(accuracy_score(y_va, y_hat)),
        "val_macro_f1": float(f1_score(y_va, y_hat, average="macro")),
        "best_epoch": 1,
        "n_steps": 1,
    }


def fit_linearsvc(X_tr, y_tr, X_va, y_va, C):
    clf = LinearSVC(C=C, dual="auto", random_state=SEED, max_iter=2000)
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_va)
    return clf, {
        "val_accuracy": float(accuracy_score(y_va, y_hat)),
        "val_macro_f1": float(f1_score(y_va, y_hat, average="macro")),
        "best_epoch": int(np.max(clf.n_iter_)),
        "n_steps": int(np.max(clf.n_iter_)),
    }


def evaluate_split(name, clf, X_va, y_va):
    pred = clf.predict(X_va)
    return {
        "name": name,
        "val_accuracy": float(accuracy_score(y_va, pred)),
        "val_macro_f1": float(f1_score(y_va, pred, average="macro")),
    }


def plot_curves(steps, epochs_log, title, out_path: Path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    step_x = [r["step"] for r in steps]
    step_y = [r["batch_loss"] for r in steps]
    window = 8
    kernel = np.ones(window) / window
    smooth = np.convolve(step_y, kernel, mode="valid")
    axes[0].plot(step_x, step_y, color="#8fb8d6", linewidth=0.8, label="逐步损失")
    axes[0].plot(
        step_x[window - 1 :],
        smooth,
        color="#1f4e79",
        linewidth=1.6,
        label=f"{window} 步滑动平均",
    )
    axes[0].set_xlabel("训练步")
    axes[0].set_ylabel("当前 batch 损失")
    axes[0].set_title("训练逐步损失")
    axes[0].legend(frameon=False, fontsize=8)

    ep = [r["epoch"] for r in epochs_log]
    axes[1].plot(ep, [r["train_batch_loss_mean"] for r in epochs_log], marker="o", label="训练损失")
    axes[1].plot(ep, [r["val_loss"] for r in epochs_log], marker="s", label="验证损失")
    ax2 = axes[1].twinx()
    ax2.plot(
        ep,
        [r["val_macro_f1"] for r in epochs_log],
        color="#c47b2b",
        marker="^",
        linewidth=1.2,
        label="验证 macro-F1",
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("损失")
    ax2.set_ylabel("验证 macro-F1")
    axes[1].set_title("按 epoch 的损失与验证指标")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bars(rows, out_path: Path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    names = [r["name"] for r in rows]
    vals = [r["val_macro_f1"] for r in rows]
    colors = ["#1f4e79" if i == int(np.argmax(vals)) else "#8fb8d6" for i in range(len(vals))]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("验证集 macro-F1")
    ax.set_xlim(min(vals) - 0.05, max(vals) + 0.02)
    ax.set_title("同一验证集上的模型对比")
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texts, labels, test_texts = load_raw()
    idx = np.arange(len(texts))
    idx_tr, idx_va = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=SEED
    )
    print(
        f"labeled={len(texts)} train_split={len(idx_tr)} val_split={len(idx_va)} test={len(test_texts)}",
        flush=True,
    )

    # 阶段 1：固定 SGD 逻辑回归，只改文本和归一化
    feature_rows = []
    feature_cache = {}
    configs = [
        ("原文+词袋", "raw", False, "bow"),
        ("原文+TFIDF-L2", "raw", False, "tfidf_l2"),
        ("原文+TFIDF-L1", "raw", False, "tfidf_l1"),
        ("原文+TFIDF-sublinear", "raw", False, "tfidf_sublinear"),
        ("原文+TFIDF-bigram", "raw", False, "tfidf_bigram"),
        ("去信头含主题+TFIDF-sublinear", "drop_all", False, "tfidf_sublinear"),
        ("保留主题去信头+TFIDF-sublinear", "keep_subject", False, "tfidf_sublinear"),
        ("保留主题去停用词+TFIDF-sublinear", "keep_subject", True, "tfidf_sublinear"),
    ]
    for name, header_mode, drop_stop, kind in configs:
        prepared = [prepare_text(texts[i], header_mode, drop_stop) for i in range(len(texts))]
        vec = make_vectorizer(kind)
        X_tr = vec.fit_transform([prepared[i] for i in idx_tr])
        X_va = vec.transform([prepared[i] for i in idx_va])#X_tr、X_va 就是这两份文本对应的稀疏矩阵，一行一篇文档，一列一个词。
        y_tr = labels[idx_tr]
        y_va = labels[idx_va]
        # 表示对比时固定同一个线性分类器，避免把优化过程的差异混进特征比较。
        _, summary = fit_linearsvc(X_tr, y_tr, X_va, y_va, C=1.0)#_ 是 Python 里表示“这个返回值先接住、后面不用”的变量名。真正留下的是 summary
        row = {
            "name": name,
            "kind": kind,
            "header_mode": header_mode,
            "drop_stop": drop_stop,
            **summary,
        }
        feature_rows.append(row)
        feature_cache[name] = (prepared, kind)
        print("feature", name, summary, flush=True)

    feature_rows.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    best_feat = feature_rows[0]
    prepared, kind = feature_cache[best_feat["name"]]
    vec = make_vectorizer(kind)
    X_tr = vec.fit_transform([prepared[i] for i in idx_tr])
    X_va = vec.transform([prepared[i] for i in idx_va])
    y_tr = labels[idx_tr]
    y_va = labels[idx_va]
    print("best feature", best_feat["name"], flush=True)

    # 阶段 2：锁定文本表示，比较模型
    model_logs = {}
    candidates = []

    for alpha in (0.1, 0.5, 1.0):
        clf, summary = fit_nb(X_tr, y_tr, X_va, y_va, alpha)
        name = f"MultinomialNB(alpha={alpha})"
        candidates.append({"name": name, "family": "nb", "alpha": alpha, **summary, "clf": clf})
        print(name, summary, flush=True)

    n_tr = X_tr.shape[0]
    alpha_grid = (1e-4, 1.0 / n_tr, 1.0 / (2.0 * n_tr))
    for alpha in alpha_grid:
        clf, steps, epochs_log, summary = train_sgd(
            X_tr, y_tr, X_va, y_va, "log_loss", alpha=alpha, epochs=EPOCHS
        )
        name = f"SGD-LogReg(alpha={alpha:.3g})"
        candidates.append(
            {
                "name": name,
                "family": "sgd_log",
                "alpha": alpha,
                "loss": "log_loss",
                **summary,
                "clf": clf,
            }
        )
        model_logs[name] = (steps, epochs_log)
        print(name, summary, flush=True)

    for alpha in alpha_grid:
        clf, steps, epochs_log, summary = train_sgd(
            X_tr, y_tr, X_va, y_va, "hinge", alpha=alpha, epochs=EPOCHS
        )
        name = f"SGD-SVM(alpha={alpha:.3g})"
        candidates.append(
            {
                "name": name,
                "family": "sgd_hinge",
                "alpha": alpha,
                "loss": "hinge",
                **summary,
                "clf": clf,
            }
        )
        model_logs[name] = (steps, epochs_log)
        print(name, summary, flush=True)

    for C in (0.5, 1.0, 2.0):
        clf, summary = fit_linearsvc(X_tr, y_tr, X_va, y_va, C)
        name = f"LinearSVC(C={C:g})"
        candidates.append({"name": name, "family": "linearsvc", "C": C, **summary, "clf": clf})
        print(name, summary, flush=True)

    #SGD-SVM和LinearSVC都是线性支持向量机，前者损失是损失是普通 hinge，后者损失为squared hinge（hinge再平方）

    print("training MLP", flush=True)
    mlp, steps, epochs_log, summary = train_mlp(
        X_tr, y_tr, X_va, y_va, hidden=(64,), alpha=1e-4, epochs=8
    )
    name = "MLP(64)"
    candidates.append({"name": name, "family": "mlp", "alpha": 1e-4, **summary, "clf": mlp})
    model_logs[name] = (steps, epochs_log)
    print(name, summary, flush=True)

    ranked = sorted(candidates, key=lambda r: (r["val_macro_f1"], r["val_accuracy"]), reverse=True)
    best = ranked[0]
    print("BEST", best["name"], best["val_macro_f1"], best["val_accuracy"], flush=True)

    # 逐步损失：优先画最终入选且有逐步日志的模型；LinearSVC 另附同强度 SGD 曲线
    curve_name = best["name"] if best["name"] in model_logs else None
    if curve_name is None:
        hinge_ones = [c for c in ranked if c["family"] == "sgd_hinge"]
        curve_name = hinge_ones[0]["name"]
    steps, epochs_log = model_logs[curve_name]
    plot_curves(
        steps,
        epochs_log,
        f"{curve_name} 的训练过程",
        OUT_DIR / "loss_curve.png",
    )
    pd.DataFrame(steps).to_csv(OUT_DIR / "loss_steps.csv", index=False)
    pd.DataFrame(epochs_log).to_csv(OUT_DIR / "loss_epochs.csv", index=False)

    bar_rows = []
    seen = set()
    for row in ranked:
        family = row["family"]
        if family in seen:
            continue
        seen.add(family)
        bar_rows.append(row)
    plot_bars(bar_rows, OUT_DIR / "model_compare.png")

    pred = best["clf"].predict(X_va)
    report = classification_report(y_va, pred, digits=3, output_dict=True)
    cm = confusion_matrix(y_va, pred)
    confused = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                confused.append({"true": int(i), "pred": int(j), "count": int(cm[i, j])})
    confused.sort(key=lambda r: r["count"], reverse=True)

    # 锁定超参后，在全部带标签数据上重训，再预测测试集
    full_vec = make_vectorizer(kind)
    X_full = full_vec.fit_transform(prepared)
    X_test = full_vec.transform(
        [
            prepare_text(t, best_feat["header_mode"], best_feat["drop_stop"])
            for t in test_texts
        ]
    )
    final_clf = refit_final(best, X_full, labels)
    test_pred = final_clf.predict(X_test).astype(int)
    pred_path = Path(__file__).resolve().parent / "prediction.csv"
    pd.DataFrame(test_pred).to_csv(pred_path, index=False, header=False)


    #它把这次实验的设置和验证集分数整理成一个字典，写成 results/metrics.json，方便之后写报告时核对
    public = []#public = [] 是在循环开始前准备一个空列表。后面每处理完一个模型，就用 public.append(item) 往里面加一条。
    for row in ranked:
        item = {k: v for k, v in row.items() if k != "clf"}
        public.append(item)
    payload = {
        "seed": SEED,
        "n_labeled": int(len(texts)),
        "n_train_split": int(len(idx_tr)),
        "n_val_split": int(len(idx_va)),
        "n_test": int(len(test_texts)),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "best_feature": best_feat,
        "feature_rows": feature_rows,
        "models": public,
        "selected_model": best["name"],
        "selected_val_accuracy": best["val_accuracy"],
        "selected_val_macro_f1": best["val_macro_f1"],
        "curve_model": curve_name,
        "curve_epochs": epochs_log,
        "top_confusions": confused[:8],
        "val_classification_report": report,
        "prediction_path": str(pred_path),
        "prediction_count": int(len(test_pred)),
    }
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", pred_path, "n=", len(test_pred), flush=True)


def refit_final(best, X_full, y_full):
    """验证集只负责选定超参。最终模型在全部带标签样本上重训，不再看验证集。"""
    family = best["family"]
    if family == "nb":
        clf = MultinomialNB(alpha=best["alpha"])
        clf.fit(X_full, y_full)
        return clf
    if family == "linearsvc":
        clf = LinearSVC(C=best["C"], dual="auto", random_state=SEED, max_iter=3000)
        clf.fit(X_full, y_full)
        return clf
    if family == "mlp":
        return fit_mlp_fixed_epochs(
            X_full, y_full, hidden=(64,), alpha=best["alpha"], epochs=best["best_epoch"]
        )
    return fit_sgd_fixed_epochs(
        X_full, y_full, best["loss"], best["alpha"], best["best_epoch"]
    )

#fixed_epochs 就是把轮数固定成 best["best_epoch"]。
def fit_mlp_fixed_epochs(X, y, hidden, alpha, epochs):
    classes = np.unique(y)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=BATCH_SIZE,
        learning_rate_init=0.001,
        random_state=SEED,
    )
    rng = np.random.RandomState(SEED)
    n = X.shape[0]
    started = False
    for _epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            Xb = X[batch].toarray() if hasattr(X, "toarray") else X[batch]
            if not started:
                clf.partial_fit(Xb, y[batch], classes=classes)
                started = True
            else:
                clf.partial_fit(Xb, y[batch])
    return clf


def fit_sgd_fixed_epochs(X, y, loss_name, alpha, epochs):
    classes = np.unique(y)
    clf = SGDClassifier(
        loss=loss_name,
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        random_state=SEED,
        tol=None,
    )
    rng = np.random.RandomState(SEED)
    n = X.shape[0]
    step_id = 0
    for _epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            if step_id == 0:
                clf.partial_fit(X[batch], y[batch], classes=classes)
            else:
                clf.partial_fit(X[batch], y[batch])
            step_id += 1
    return clf


if __name__ == "__main__":
    main()
