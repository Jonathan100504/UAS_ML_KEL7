import re, time, json, random, os, inspect
from pathlib import Path
import numpy as np
import pandas as pd
from google_play_scraper import reviews, Sort
from scipy.special import softmax

# HF / IndoBERT
from datasets import Dataset, Features, Value, ClassLabel
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
import evaluate
import torch

from sklearn.metrics import classification_report, accuracy_score, f1_score

# BERTopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from umap import UMAP

SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- Shim accelerate lama ----
try:
    from accelerate import Accelerator
    if "keep_torch_compile" not in inspect.signature(Accelerator.unwrap_model).parameters:
        _orig = Accelerator.unwrap_model
        def _shim(self, model, *a, **kw):
            kw.pop("keep_torch_compile", None)
            return _orig(self, model, *a, **kw)
        Accelerator.unwrap_model = _shim
except Exception:
    pass
# --------------------------------

# ========= Common =========
def load_slang(path: Path):
    if not path or not path.exists(): return {}
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return {k.lower(): v.lower() for k, v in d.items()}
    except:
        pass
    mapping = {}
    for line in text.splitlines():
        if "," in line:
            a,b = line.split(",",1)
            mapping[a.lower().strip()] = b.lower().strip()
    return mapping

def load_stopwords(path: Path):
    if not path or not path.exists(): return set()
    return set([w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()])

def fetch_reviews_resumable(app_id, lang, country, target, save_csv: Path, batch_size=200, sleep_sec=1.0):
    if save_csv.exists():
        df = pd.read_csv(save_csv)
        seen = set(df["reviewId"].astype(str))
    else:
        df = pd.DataFrame(columns=["reviewId","content","score","at","thumbsUpCount","appVersion"])
        seen = set()
    token = None
    while len(df) < target:
        batch, token = reviews(
            app_id, lang=lang, country=country,
            sort=Sort.NEWEST, count=batch_size, continuation_token=token
        )
        if not batch: break
        rows = []
        for r in batch:
            rid = str(r.get("reviewId"))
            if rid in seen: continue
            seen.add(rid)
            rows.append({
                "reviewId": rid,
                "content": r.get("content"),
                "score": r.get("score"),
                "at": r.get("at").strftime("%Y-%m-%d") if r.get("at") else None,
                "thumbsUpCount": r.get("thumbsUpCount"),
                "appVersion": r.get("appVersion"),
            })
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(save_csv, index=False)
        if token is None: break
        time.sleep(sleep_sec)
    return df

def _squeeze_repeats(t: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1', t)

def _clean_text(t):
    if not isinstance(t, str):
        t = "" if t is None else str(t)
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"(?<=\w)[\-\._#]+(?=\w)", "", t)
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = _squeeze_repeats(t)
    return t

def _normalize_slang(t, slang_map):
    return " ".join([slang_map.get(w, w) for w in t.split()])

def _remove_stopwords(t, stopwords_set):
    return " ".join([w for w in t.split() if w not in stopwords_set])

def simple_word_tokenize(text: str):
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z0-9]+", text.lower())

def simple_tokenize(text: str):
    if not isinstance(text, str):
        return []
    return [tok for tok in text.split() if tok]

def preprocess_and_split(
    df, date_min, date_max, slang_map, stopwords_set,
    label_mode="binary", min_words=0, balance=False
):
    """
    Dikurangi: tidak ada filter min word count & tidak ada balancing.
    """
    steps = {}
    steps["raw_total"] = int(len(df))

    # drop NaN content/score
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["content", "score"])
    steps["after_dropna"] = int(len(df))
    steps["missing_removed"] = int(before - len(df))

    # filter tanggal
    df = df[(df["at"]>=pd.to_datetime(date_min)) & (df["at"]<=pd.to_datetime(date_max))].copy()
    steps["after_date_filter"] = int(len(df))

    # label mapping
    score = pd.to_numeric(df["score"], errors="coerce")
    if label_mode == "3class":
        df["label"] = score.apply(lambda x: "positive" if x>=4 else ("negative" if x<=2 else "neutral"))
        keep = score.isin([1,2,3,4,5])
    elif label_mode == "binary":
        df["label"] = score.apply(lambda x: "positive" if x>=4 else ("negative" if x<=2 else None))
        keep = score.isin([1,2,4,5])
    elif label_mode == "binary_strict":
        df["label"] = score.apply(lambda x: "positive" if x==5 else ("negative" if x==1 else None))
        keep = score.isin([1,5])
    else:
        raise ValueError("label_mode invalid")

    df = df[keep].copy().dropna(subset=["label"])
    steps["after_label_keep"] = int(len(df))
    steps["label_counts_before_balance"] = df["label"].value_counts().to_dict()

    # cleansing → normalisasi slang → stopword → token
    df["clean"]      = df["content"].map(_clean_text)
    df["normalized"] = df["clean"].apply(lambda t: _normalize_slang(t, slang_map))
    df["final_text"] = df["normalized"].apply(lambda t: _remove_stopwords(t, stopwords_set))
    df["tokens"]     = df["final_text"].apply(simple_tokenize)

    # deduplikasi teks
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["final_text"])
    steps["after_dedup"] = int(len(df))
    steps["removed_duplicates"] = int(before_dedup - len(df))

    # hitung removed_total setelah dedup
    removed_total = int(steps["raw_total"] - len(df))

    examples = {
        "clean_examples": df[["content","clean"]].head(5).to_dict(orient="records"),
        "norm_examples":  df[["clean","normalized"]].head(5).to_dict(orient="records"),
        "stop_examples":  df[["normalized","final_text"]].head(5).to_dict(orient="records")
    }
    pstats = {"missing_removed": steps.get("missing_removed", 0), "steps": steps}
    return df, removed_total, pstats, examples

# ========= IndoBERT =========
INDOBERT_CKPT = "indobenchmark/indobert-base-p1"

def prepare_hf_dataset(X_train, y_train, X_test, y_test, label2id, max_length=256):
    ytr = [label2id[y] for y in y_train]
    yte = [label2id[y] for y in y_test]
    feat = Features({"text": Value("string"), "label": ClassLabel(num_classes=len(label2id), names=list(label2id.keys()))})
    ds_train = Dataset.from_dict({"text": list(X_train), "label": ytr}, features=feat)
    ds_test  = Dataset.from_dict({"text": list(X_test),  "label": yte}, features=feat)

    tokenizer = AutoTokenizer.from_pretrained(INDOBERT_CKPT)

    def tok(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    ds_train = ds_train.map(tok, batched=True)
    ds_test  = ds_test.map(tok, batched=True)
    return ds_train, ds_test, tokenizer

# NEW: callback untuk merekam log/riwayat per-epoch
class HistoryCallback(TrainerCallback):
    def __init__(self):
        self.records = []

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {}).copy()
        # hanya simpan jika ada info penting
        if any(k in logs for k in ("loss", "eval_loss", "eval_accuracy", "eval_f1_macro", "learning_rate")):
            logs["epoch"] = float(state.epoch) if state.epoch is not None else None
            self.records.append(logs)

def _compress_history(records):
    """
    Ambil titik terakhir per epoch untuk train dan eval.
    """
    epoch_set = sorted({r.get("epoch") for r in records if r.get("epoch") is not None})
    hist = {"epoch": [] , "train_loss": [], "eval_loss": [], "eval_accuracy": [], "eval_f1_macro": []}

    for ep in epoch_set:
        # titik terakhir pada epoch tsb
        ep_logs = [r for r in records if r.get("epoch") == ep]
        last = ep_logs[-1] if ep_logs else {}
        hist["epoch"].append(int(round(ep)))
        hist["train_loss"].append(last.get("loss"))
        hist["eval_loss"].append(last.get("eval_loss"))
        hist["eval_accuracy"].append(last.get("eval_accuracy"))
        hist["eval_f1_macro"].append(last.get("eval_f1_macro"))

    return hist

class EpochReportsCallback(TrainerCallback):
    def __init__(self, predict_fn, id2label):
        self.predict_fn = predict_fn
        self.id2label = id2label
        self.reports = []
        self._epoch_counter = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        self._epoch_counter += 1
        epoch_num = self._epoch_counter
        y_true, y_pred = self.predict_fn()
        labs = [self.id2label[i] for i in sorted(self.id2label.keys())]
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        rep = classification_report(y_true, y_pred, target_names=labs)
        self.reports.append({
            "epoch": epoch_num,
            "acc": acc,
            "f1_macro": f1m,
            "report": rep
        })

def train_indobert(ds_train, ds_test, tokenizer, id2label, label2id,
                   out_dir, epochs=4, batch_size=16, lr=2e-5,
                   weight_decay=0.01, max_length=256):
    model = AutoModelForSequenceClassification.from_pretrained(
        INDOBERT_CKPT, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    base_args = dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        seed=SEED,
        logging_steps=50,
        save_total_limit=2,
        # NEW: log per epoch
        logging_strategy="epoch",
    )

    try:
        try:
            from transformers import IntervalStrategy
            eval_kw = {"evaluation_strategy": IntervalStrategy.EPOCH,
                       "save_strategy": IntervalStrategy.EPOCH}
        except Exception:
            eval_kw = {"evaluation_strategy": "epoch", "save_strategy": "epoch"}

        full_args = dict(
            **base_args, **eval_kw,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
        )
        args = TrainingArguments(**full_args)
        es_cb = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-4)
    except TypeError:
        args = TrainingArguments(**base_args)
        es_cb = None

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[cb for cb in [es_cb] if cb is not None]
    )

    # baseline sebelum training
    out0 = trainer.predict(ds_test)
    y_true0 = out0.label_ids
    y_pred0 = out0.predictions.argmax(axis=-1)
    labs = [id2label[i] for i in sorted(id2label.keys())]
    baseline_report = classification_report(y_true0, y_pred0, target_names=labs)

    # Per-epoch report + riwayat kurva
    def _predict_eval():
        out = trainer.predict(ds_test)
        return out.label_ids, out.predictions.argmax(axis=-1)

    epoch_cb = EpochReportsCallback(_predict_eval, id2label)
    hist_cb = HistoryCallback()
    trainer.add_callback(epoch_cb)
    trainer.add_callback(hist_cb)

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    history = _compress_history(hist_cb.records)
    return baseline_report, epoch_cb.reports, history

def load_indobert(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model, tokenizer

@torch.no_grad()
def predict_indobert(model, tokenizer, texts, max_length=256, return_logits=False):
    """
    Prediksi label untuk texts.
    
    Args:
        model: IndoBERT model
        tokenizer: tokenizer
        texts: list of texts
        max_length: max token length
        return_logits: jika True, return (preds, logits). Jika False, hanya return preds.
    
    Return:
        preds (list) atau (preds, logits_array)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    preds = []
    logits_list = []
    
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc)
        logits = out.logits[0].cpu().numpy()  # shape: (num_classes,)
        p = logits.argmax()
        label = model.config.id2label[int(p)]
        preds.append(label)
        logits_list.append(logits)
    
    if return_logits:
        return preds, np.array(logits_list)  # shape: (batch_size, num_classes)
    return preds

# ========= BERTopic =========
def run_bertopic_and_plots(texts, out_dir: Path, prefix: str):
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED,
    )
    topic_model = BERTopic(
        embedding_model=encoder, umap_model=umap_model,
        calculate_probabilities=True, verbose=False,
    )
    topics, probs = topic_model.fit_transform(texts)

    freq = topic_model.get_topic_freq()
    topN = freq.head(10)

    fig = plt.figure(figsize=(8, 4.5))
    plt.barh(topN["Topic"].astype(str), topN["Count"])
    plt.title("Top Topic Frequency (Top 10)")
    plt.xlabel("Count"); plt.ylabel("Topic ID")
    out_bar = out_dir / f"{prefix}_topic_freq.png"
    fig.tight_layout(); fig.savefig(out_bar, dpi=120, bbox_inches="tight"); plt.close(fig)

    top_keywords = []
    for t in topN["Topic"].tolist():
        words = [w for w, _ in (topic_model.get_topic(t) or [])][:5]
        top_keywords.append({"topic_id": int(t), "keywords": ", ".join(words)})

    return {
        "freq_image": f"/static/plots/{out_bar.name}",
        "top_keywords": top_keywords,
        "n_topics": int(freq[freq['Topic'] != -1].shape[0])
    }

# ========= NEW: ABSA & Confidence Score =========

# Common keywords untuk ABSA (dapat diperluas)
ASPECT_KEYWORDS = {
    "performance": ["cepat", "lambat", "lag", "freeze", "crash", "error", "kecepatan", "performa", "responsif", "loading"],
    "ui_design": ["ui", "design", "interface", "tampilan", "layout", "fitur", "menu", "button", "user experience"],
    "battery": ["baterai", "battery", "daya", "charger", "charge", "power", "consumption"],
    "stability": ["stabil", "crash", "error", "bug", "masalah", "gangguan", "issue", "problem"],
    "functionality": ["fungsi", "fitur", "work", "tidak bisa", "error", "failed", "tidak jalan"],
    "ux_experience": ["mudah", "sulit", "nyaman", "ribet", "kerumitan", "complicated", "user friendly"],
    "pricing": ["harga", "mahal", "murah", "price", "cost", "premium", "payment"],
    "support": ["support", "bantuan", "customer service", "help", "responsif", "reply"]
}

def extract_aspects(text: str) -> dict:
    """
    Ekstraksi aspek dari review berdasarkan keyword matching.
    Return: dict dengan aspect nama sebagai key, nilai boolean.
    """
    text_lower = text.lower()
    aspects = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        found = any(kw in text_lower for kw in keywords)
        aspects[aspect] = found
    return aspects

def calculate_confidence_and_entropy(logits):
    """
    Dari logits (shape: (batch_size, num_classes)):
    - Hitung softmax probabilities
    - Hitung confidence (max probability)
    - Hitung entropy (ukuran ketidakpastian)
    """
    probs = softmax(logits, axis=-1)  # shape: (batch_size, num_classes) - gunakan scipy softmax
    confidence = np.max(probs, axis=-1)   # shape: (batch_size,)
    
    # Shannon entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)  # shape: (batch_size,)
    
    return probs, confidence, entropy

class AmbiguityDetector:
    """
    Multi-komponen ambiguity detector:
    1. Low confidence: confidence < threshold
    2. High entropy: entropy > threshold
    3. Close margin: perbedaan top-2 probability kecil
    4. Contrastive keywords: mix antara positive/negative keywords
    """
    def __init__(self, 
                 conf_threshold=0.65,
                 entropy_threshold=0.8,
                 margin_threshold=0.15,
                 pos_keywords=None,
                 neg_keywords=None):
        self.conf_threshold = conf_threshold
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        
        self.pos_keywords = pos_keywords or ["bagus", "mantap", "keren", "ok", "suka", "best", "excellent", "good", "great", "awesome", "perfect", "love", "hebat", "cemerlang"]
        self.neg_keywords = neg_keywords or ["jelek", "buruk", "tidak bagus", "masalah", "error", "crash", "lag", "lambat", "tidak bisa", "bad", "worse", "poor", "terrible", "hate", "worst", "malah", "kecewa"]
    
    def detect(self, text: str, logits, label_idx: int, id2label: dict, probs=None, confidence=None, entropy=None):
        """
        Deteksi ambiguity dan return dict dengan info detail.
        
        Args:
            text: review text
            logits: model logits (1D array untuk 1 sample)
            label_idx: predicted label index
            id2label: mapping dari label index ke nama
            probs: softmax probabilities (opsional, akan dihitung jika None)
            confidence: confidence score (opsional)
            entropy: entropy (opsional)
        
        Return:
            {
                "is_ambiguous": bool,
                "confidence": float,
                "entropy": float,
                "margin": float,
                "has_contrastive": bool,
                "ambiguity_types": list of str
            }
        """
        if probs is None:
            probs = softmax(logits)
        if confidence is None:
            confidence = np.max(probs)
        if entropy is None:
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Calculate margin (perbedaan top-2 probability)
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        
        # Deteksi contrastive keywords
        text_lower = text.lower()
        has_pos = any(kw in text_lower for kw in self.pos_keywords)
        has_neg = any(kw in text_lower for kw in self.neg_keywords)
        has_contrastive = has_pos and has_neg
        
        # Tentukan ambiguity type
        ambiguity_types = []
        if confidence < self.conf_threshold:
            ambiguity_types.append("low_confidence")
        if entropy > self.entropy_threshold:
            ambiguity_types.append("high_entropy")
        if margin < self.margin_threshold:
            ambiguity_types.append("close_margin")
        if has_contrastive:
            ambiguity_types.append("contrastive_keywords")
        
        is_ambiguous = len(ambiguity_types) > 0
        
        return {
            "is_ambiguous": is_ambiguous,
            "confidence": float(confidence),
            "entropy": float(entropy),
            "margin": margin,
            "has_contrastive": has_contrastive,
            "ambiguity_types": ambiguity_types,
            "aspects": extract_aspects(text)
        }

def analyze_reviews_with_features(texts, y_true, y_pred_labels, logits_array, 
                                  id2label, detector: AmbiguityDetector = None):
    """
    Analisis review lengkap dengan ABSA, confidence, dan ambiguity.
    
    Args:
        texts: list of review texts
        y_true: true labels (array)
        y_pred_labels: predicted labels (array)
        logits_array: model logits (shape: (batch_size, num_classes))
        id2label: mapping label index to name
        detector: AmbiguityDetector instance
    
    Return:
        {
            "detailed_results": list of dict (per review),
            "ambiguity_summary": dict dengan breakdown tipe ambiguity,
            "aspects_frequency": dict dengan frekuensi aspek,
            "confidence_stats": dict dengan statistik confidence
        }
    """
    if detector is None:
        detector = AmbiguityDetector()
    
    probs_all, confidence_all, entropy_all = calculate_confidence_and_entropy(logits_array)
    
    detailed_results = []
    ambiguity_counter = {
        "low_confidence": 0,
        "high_entropy": 0,
        "close_margin": 0,
        "contrastive_keywords": 0,
        "total_ambiguous": 0
    }
    aspects_counter = {asp: 0 for asp in ASPECT_KEYWORDS.keys()}
    
    for i, text in enumerate(texts):
        logits = logits_array[i]
        probs = probs_all[i]
        confidence = confidence_all[i]
        entropy = entropy_all[i]
        
        pred_idx = np.argmax(logits)
        pred_label = id2label[pred_idx]
        true_label = y_true[i]
        
        amb_info = detector.detect(
            text=text,
            logits=logits,
            label_idx=pred_idx,
            id2label=id2label,
            probs=probs,
            confidence=confidence,
            entropy=entropy
        )
        
        if amb_info["is_ambiguous"]:
            ambiguity_counter["total_ambiguous"] += 1
            for amb_type in amb_info["ambiguity_types"]:
                ambiguity_counter[amb_type] += 1
        
        for asp, found in amb_info["aspects"].items():
            if found:
                aspects_counter[asp] += 1
        
        detailed_results.append({
            "idx": i,
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "is_correct": (pred_label == true_label),
            "confidence": confidence,
            "entropy": entropy,
            "margin": amb_info["margin"],
            "is_ambiguous": amb_info["is_ambiguous"],
            "ambiguity_types": amb_info["ambiguity_types"],
            "has_contrastive": amb_info["has_contrastive"],
            "aspects": amb_info["aspects"],
            "probs_dict": {id2label[j]: float(p) for j, p in enumerate(probs)}
        })
    
    confidence_stats = {
        "mean": float(np.mean(confidence_all)),
        "median": float(np.median(confidence_all)),
        "min": float(np.min(confidence_all)),
        "max": float(np.max(confidence_all)),
        "std": float(np.std(confidence_all))
    }
    
    return {
        "detailed_results": detailed_results,
        "ambiguity_summary": ambiguity_counter,
        "aspects_frequency": {k: v for k, v in aspects_counter.items() if v > 0},
        "confidence_stats": confidence_stats
    }
