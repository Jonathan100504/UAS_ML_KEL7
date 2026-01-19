import os
from pathlib import Path
from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from utils_nlp import (
    load_slang, load_stopwords, fetch_reviews_resumable, preprocess_and_split,
    prepare_hf_dataset, train_indobert, load_indobert, predict_indobert,
    run_bertopic_and_plots, simple_word_tokenize,
    AmbiguityDetector, analyze_reviews_with_features, ASPECT_KEYWORDS
)


from wordcloud import WordCloud

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

app = Flask(__name__)
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "static" / "plots"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DEFAULTS = {
    "lang": "id", "country": "id",
    "date_min": "2020-01-01", "date_max": "2026-12-31",
    "target_reviews": 2000, "label_mode": "binary_strict",
    "epochs": 4, "batch_size": 16, "learning_rate": "2e-5",
    "weight_decay": 0.01, "max_length": 256, "run_bertopic": "on"
}

def _ts() -> str: return datetime.now().strftime('%Y%m%d_%H%M%S')

def _save_plot(fig, name_prefix: str) -> str:
    fname = f"{name_prefix}_{_ts()}.png"
    path = PLOT_DIR / fname
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return f"/static/plots/{fname}"

# ---------- Label distribution chart ----------
def _plot_label_distribution(counts: dict, label_mode: str) -> str:
    is_3 = (label_mode == "3class")
    order = ["negative", "neutral", "positive"] if is_3 else ["negative", "positive"]
    xticks = ["Negatif", "Netral", "Positif"] if is_3 else ["Negatif", "Positif"]
    colors = ["#E74C3C", "#BDC3C7", "#27AE60"] if is_3 else ["#E74C3C", "#27AE60"]
    vals = [int(counts.get(k, 0)) for k in order]

    fig = plt.figure(figsize=(6.5, 3.6))
    ax = fig.add_subplot(111)
    bars = ax.bar(range(len(vals)), vals, tick_label=xticks, color=colors)
    ax.set_title("Distribusi Label (setelah preprocessing)")
    ax.set_xlabel("Label Sentimen")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ymax = max(vals) if vals else 0
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width()/2.0,
                v + (0.02*ymax if ymax > 0 else 1),
                f"{v}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    return _save_plot(fig, "label_distribution")

# ---------- WordCloud helpers ----------
def _plot_wordcloud_from_text(text: str, title: str, name_prefix: str) -> str:
    wc = WordCloud(
        width=1200, height=600, background_color="white",
        collocations=False, prefer_horizontal=0.9, max_words=300
    ).generate(text if isinstance(text, str) else "")
    fig = plt.figure(figsize=(9.5, 5.2))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    return _save_plot(fig, name_prefix)

# ---------- NEW: Learning curves ----------
def _plot_training_curves(history: dict) -> dict:
    """
    history = {
      'epoch': [...],
      'train_loss': [...],          # optional
      'eval_loss': [...],           # optional
      'eval_accuracy': [...],       # optional
      'eval_f1_macro': [...],       # optional
    }
    """
    paths = {}

    # Loss curves
    if history.get("epoch") and (history.get("train_loss") or history.get("eval_loss")):
        fig = plt.figure(figsize=(6.5, 3.6))
        if history.get("train_loss"):
            plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
        if history.get("eval_loss"):
            plt.plot(history["epoch"], history["eval_loss"], marker="o", label="Eval Loss")
        plt.title("Learning Curve — Loss per Epoch")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(alpha=0.3); plt.legend()
        paths["loss"] = _save_plot(fig, "curve_loss")

    # Metrics curves
    if history.get("epoch") and (history.get("eval_accuracy") or history.get("eval_f1_macro")):
        fig = plt.figure(figsize=(6.5, 3.6))
        if history.get("eval_accuracy"):
            plt.plot(history["epoch"], history["eval_accuracy"], marker="o", label="Eval Accuracy")
        if history.get("eval_f1_macro"):
            plt.plot(history["epoch"], history["eval_f1_macro"], marker="o", label="Eval F1 Macro")
        plt.title("Evaluation Curve — Metrics per Epoch")
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.grid(alpha=0.3); plt.legend()
        paths["metrics"] = _save_plot(fig, "curve_metrics")

    return paths
# --------------------------------------

def _plot_ambiguity_breakdown(ambiguity_summary: dict) -> str:
    """Plot ambiguity types breakdown"""
    types = ["low_confidence", "high_entropy", "close_margin", "contrastive_keywords"]
    counts = [ambiguity_summary.get(t, 0) for t in types]
    labels = ["Low Conf", "High Entropy", "Close Margin", "Contrastive KW"]
    colors = ["#FF6B6B", "#FFA500", "#FFD93D", "#6BCB77"]
    
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90
    )
    ax.set_title("Ambiguity Breakdown — Tipe Ketidakjelasan", fontsize=12, fontweight="bold")
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    return _save_plot(fig, "ambiguity_breakdown")

def _plot_aspects_frequency(aspects_freq: dict, total_reviews: int) -> str:
    """Plot aspects frequency"""
    if not aspects_freq:
        return ""
    
    aspects = list(aspects_freq.keys())
    freqs = list(aspects_freq.values())
    percentages = [100 * f / total_reviews for f in freqs]
    
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    bars = ax.barh(aspects, percentages, color="#6BCB77")
    ax.set_xlabel("Persentase Review (%)")
    ax.set_title("Frekuensi Aspek (ABSA) dalam Review")
    ax.grid(axis="x", alpha=0.3)
    
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2, 
                f"{pct:.1f}%", ha="left", va="center", fontsize=9)
    
    return _save_plot(fig, "aspects_frequency")

def _plot_confidence_distribution(confidence_stats: dict, confidences: list) -> str:
    """Plot confidence score distribution"""
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    
    ax.hist(confidences, bins=30, color="#4ECDC4", edgecolor="black", alpha=0.7)
    ax.axvline(confidence_stats["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean: {confidence_stats['mean']:.3f}")
    ax.axvline(confidence_stats["median"], color="orange", linestyle="--", linewidth=2, label=f"Median: {confidence_stats['median']:.3f}")
    
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Jumlah Review")
    ax.set_title("Distribusi Confidence Score Prediksi")
    ax.legend()
    ax.grid(alpha=0.3)
    
    return _save_plot(fig, "confidence_distribution")

# --------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", defaults=DEFAULTS)

@app.route("/analyze", methods=["POST"])
def analyze():
    app_id = request.form.get("app_id", "").strip()
    lang = request.form.get("lang", DEFAULTS["lang"]).strip()
    country = request.form.get("country", DEFAULTS["country"]).strip()
    date_min = request.form.get("date_min", DEFAULTS["date_min"]).strip()
    date_max = request.form.get("date_max", DEFAULTS["date_max"]).strip()
    target_reviews = int(request.form.get("target_reviews", DEFAULTS["target_reviews"]))
    label_mode = request.form.get("label_mode", DEFAULTS["label_mode"])

    epochs = int(request.form.get("epochs", DEFAULTS["epochs"]))
    batch_size = int(request.form.get("batch_size", DEFAULTS["batch_size"]))
    learning_rate_str = request.form.get("learning_rate", DEFAULTS["learning_rate"])
    learning_rate = float(learning_rate_str)  # Convert string like "2e-5" to float
    weight_decay = float(request.form.get("weight_decay", DEFAULTS["weight_decay"]))
    max_length = int(request.form.get("max_length", DEFAULTS["max_length"]))
    run_bertopic_flag = (request.form.get("run_bertopic", DEFAULTS["run_bertopic"]) == "on")

    if not app_id:
        return render_template("index.html", defaults=DEFAULTS, error="App ID wajib diisi.")

    slang_map = load_slang(BASE_DIR / "slangwords.txt")
    stopwords_set = load_stopwords(BASE_DIR / "stopwords.txt")

    # crawl/resume
    csv_path = DATA_DIR / f"{app_id.replace('.','_')}_{lang}_{country}.csv"
    df_raw = fetch_reviews_resumable(
        app_id, lang, country, target_reviews, csv_path, batch_size=200, sleep_sec=1.0
    )
    if df_raw.empty:
        return render_template("index.html", defaults=DEFAULTS,
                               error="Tidak ada data. Coba target lebih kecil / ganti lang/country.")

    # PREPROCESS — tanpa min word count & tanpa balancing
    df, removed, pstats, examples = preprocess_and_split(
        df_raw, date_min, date_max, slang_map, stopwords_set,
        label_mode=label_mode, min_words=0, balance=False
    )

    need_classes = 3 if label_mode == "3class" else 2
    if df["label"].nunique() < need_classes:
        return render_template("index.html", defaults=DEFAULTS,
                               error="Label tidak cukup beragam untuk mode yang dipilih.")

    # ---- Charts: distribusi label + wordcloud ----
    label_counts = df["label"].value_counts().to_dict()
    label_plot_file = _plot_label_distribution(label_counts, label_mode)

    wordcloud_files = []
    all_text = " ".join(df["final_text"].tolist())
    wordcloud_files.append({
        "title": "WordCloud — Semua Label",
        "img": _plot_wordcloud_from_text(all_text, "WordCloud — Semua Label", "wc_all")
    })
    label_titles = {
        "negative": "WordCloud — Negatif",
        "neutral": "WordCloud — Netral",
        "positive": "WordCloud — Positif",
    }
    for lab in ["negative", "neutral", "positive"]:
        if lab not in df["label"].unique(): continue
        text_lab = " ".join(df[df["label"] == lab]["final_text"].tolist())
        wordcloud_files.append({
            "title": label_titles[lab],
            "img": _plot_wordcloud_from_text(text_lab, label_titles[lab], f"wc_{lab}")
        })

    # SPLIT
    y = df["label"]; X_text = df["final_text"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    split_info = {"train_size": int(len(X_train)),
                  "valid_size": int(len(X_test)),
                  "test_size": 0}
    label_order = ["negative","neutral","positive"] if label_mode=="3class" else ["negative","positive"]

    # TRAIN / LOAD
    model_tag = f"{app_id.replace('.', '_')}_{lang}_{country}_{label_mode}"
    model_out = MODEL_DIR / f"indobert_{model_tag}"
    id2label = {i: lab for i, lab in enumerate(sorted(df['label'].unique()))}
    label2id = {v: k for k, v in id2label.items()}

    # NEW: placeholder for curves
    curve_paths = {}

    try:
        ds_train, ds_test, tokenizer = prepare_hf_dataset(
            X_train, y_train, X_test, y_test, label2id, max_length=max_length
        )

        # token preview
        token_preview = []
        sample_texts = list(X_train)[:2]
        for txt in sample_texts:
            toks = simple_word_tokenize(txt)
            token_preview.append({"text": txt, "tokens": toks[:30]})

        need_train = not (model_out / "config.json").exists()
        pre_report, epoch_reports, history = None, [], {}

        if need_train and epochs > 0:
            pre_report, epoch_reports, history = train_indobert(
                ds_train, ds_test, tokenizer, id2label, label2id,
                out_dir=str(model_out),
                epochs=epochs, batch_size=batch_size, lr=learning_rate,
                weight_decay=weight_decay, max_length=max_length
            )
            curve_paths = _plot_training_curves(history)

        # jika belum ada model & epochs==0 → minimal 1 epoch
        if not (model_out / "config.json").exists() and epochs == 0:
            pre_report, epoch_reports, history = train_indobert(
                ds_train, ds_test, tokenizer, id2label, label2id,
                out_dir=str(model_out),
                epochs=1, batch_size=batch_size, lr=learning_rate,
                weight_decay=weight_decay, max_length=max_length
            )
            curve_paths = _plot_training_curves(history)

        model, tokenizer = load_indobert(str(model_out))
        y_pred, logits_array = predict_indobert(
            model, tokenizer, list(X_test), max_length=max_length, return_logits=True
        )
        best_params_str = f"epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, wd={weight_decay}, max_len={max_length}"

        # NEW: ABSA & Ambiguity Analysis
        detector = AmbiguityDetector(
            conf_threshold=0.65,
            entropy_threshold=0.8,
            margin_threshold=0.15
        )
        analysis_results = analyze_reviews_with_features(
            texts=list(X_test),
            y_true=list(y_test),
            y_pred_labels=y_pred,
            logits_array=logits_array,
            id2label=id2label,
            detector=detector
        )

    except Exception as e:
        return render_template("index.html", defaults=DEFAULTS,
                               error=f"Gagal train/load IndoBERT: {type(e).__name__}: {e}")

    # METRICS
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, labels=label_order)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred, labels=label_order)
    fig_cm = plt.figure()
    plt.imshow(cm, cmap="Blues"); plt.title(f"Confusion Matrix — IndoBERT ({' / '.join(label_order)})")
    plt.colorbar(); plt.xticks(range(len(label_order)), [s.title()[:3] for s in label_order])
    plt.yticks(range(len(label_order)), [s.title()[:3] for s in label_order])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plot_cm_path = _save_plot(fig_cm, "cm_indobert")

    # BERTopic opsional
    topics_info = None
    if run_bertopic_flag:
        try:
            topics_info = run_bertopic_and_plots(
                texts=df["final_text"].tolist(),
                out_dir=PLOT_DIR, prefix=f"{app_id.replace('.', '_')}_{lang}_{country}"
            )
        except Exception:
            topics_info = None

    # NEW: Generate ABSA & Ambiguity visualizations
    ambiguity_breakdown_plot = _plot_ambiguity_breakdown(analysis_results["ambiguity_summary"])
    aspects_plot = ""
    if analysis_results["aspects_frequency"]:
        aspects_plot = _plot_aspects_frequency(
            analysis_results["aspects_frequency"], 
            len(X_test)
        )
    confidence_dist_plot = _plot_confidence_distribution(
        analysis_results["confidence_stats"],
        [r["confidence"] for r in analysis_results["detailed_results"]]
    )

    # Extract ambiguous reviews untuk highlight di tabel
    ambiguous_reviews = [r for r in analysis_results["detailed_results"] if r["is_ambiguous"]]
    ambiguous_review_idxs = {r["idx"] for r in ambiguous_reviews}

    # Ambiguity summary stats
    ambiguity_summary = analysis_results["ambiguity_summary"]
    ambiguity_summary["ambiguous_ratio"] = (
        100 * ambiguity_summary["total_ambiguous"] / len(X_test) 
        if len(X_test) > 0 else 0
    )

    # PREVIEW — tambahkan confidence score & ambiguity status
    # Buat mapping dari X_test ke indices dalam analysis_results
    preview = []
    test_indices_set = {i for i in range(len(X_test))}
    
    for i, (idx, row) in enumerate(df[["content","score","label","final_text"]].head(20).iterrows()):
        rec = row.to_dict()
        # Cek apakah ini di dalam test set
        if i in test_indices_set and i < len(analysis_results["detailed_results"]):
            det = analysis_results["detailed_results"][i]
            rec["confidence"] = f"{det['confidence']:.3f}"
            rec["is_ambiguous"] = det["is_ambiguous"]
            rec["ambiguity_types"] = det["ambiguity_types"]
            rec["aspects"] = [asp for asp, found in det["aspects"].items() if found]
        else:
            rec["confidence"] = "N/A"
            rec["is_ambiguous"] = False
            rec["ambiguity_types"] = []
            rec["aspects"] = []
        preview.append(rec)

    return render_template(
        "result_bert.html",
        app_id=app_id, lang=lang, country=country,
        date_min=date_min, date_max=date_max,
        target_reviews=target_reviews, rows=len(df),
        label_mode=label_mode,
        removed=removed, pstats=pstats,
        best_params=best_params_str,
        acc=f"{acc:.4f}", f1_macro=f"{f1_macro:.4f}", f1_weighted=f"{f1_weighted:.4f}",
        report=report, plot_file=plot_cm_path,
        topics_info=topics_info,
        pre_report=pre_report, epoch_reports=epoch_reports,
        split_info=split_info,
        examples=examples,
        token_preview=token_preview,
        preview=preview,
        label_plot_file=label_plot_file,
        wordcloud_files=wordcloud_files,
        # NEW
        curve_loss_file=curve_paths.get("loss"),
        curve_metrics_file=curve_paths.get("metrics"),
        # ABSA & Ambiguity
        ambiguity_breakdown_plot=ambiguity_breakdown_plot,
        aspects_plot=aspects_plot,
        confidence_dist_plot=confidence_dist_plot,
        ambiguity_summary=ambiguity_summary,
        confidence_stats=analysis_results["confidence_stats"],
        ambiguous_reviews=ambiguous_reviews[:30],  # top 30 ambiguous
        ambiguous_review_idxs=ambiguous_review_idxs,
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
