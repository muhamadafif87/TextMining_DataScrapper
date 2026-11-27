import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from svm.preprocessing import preprocess_text
except Exception:
    preprocess_text = None

# ---------------------------------------------------------------------------
# Konfigurasi Path (Sesuai struktur TextMining_DataScrapper)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

METADATA_PATH = MODELS_DIR / "model_metadata.json"
SPLIT_PATH = OUTPUT_DIR / "dataset_split.csv"
REVIEW_DATA_PATH = OUTPUT_DIR / "cleaned_data_auto_label_review.csv"
CLEANED_DATA_PATH = BASE_DIR / "cleaned_data.csv"

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
    "tidak", "tak", "ada", "akan", "dalam", "atau", "juga", "saja", "bisa",
    "sudah", "belum", "oleh", "sebagai", "karena", "jadi", "kalau", "kami",
    "kita", "saya", "aku", "kamu", "dia", "mereka", "ya", "yg", "dgn", "utk",
    "gak", "ga", "nya", "apa", "banget", "lagi", "sih", "deh", "nih", "tapi",
    "jika", "bahwa", "sama", "hanya", "para", "bagi", "lebih", "seperti",
    "the", "and", "http", "https", "amp", "rt", "co",
}
TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")

SENTIMENT_MAP = {
    "positif": "Positif",
    "positive": "Positif",
    "senang": "Positif",
    "bahagia": "Positif",
    "joy": "Positif",
    "negatif": "Negatif",
    "negative": "Negatif",
    "marah": "Negatif",
    "jijik": "Negatif",
    "sedih": "Negatif",
    "takut": "Negatif",
    "anger": "Negatif",
    "disgust": "Negatif",
    "sadness": "Negatif",
    "fear": "Negatif",
    "netral": "Netral",
    "neutral": "Netral",
    "terkejut": "Netral",
    "surprise": "Netral",
}

st.set_page_config(
    page_title="Analisis Sentimen & Emosi Perpres 111/2025",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Loader dengan Cache Otomatis (Refresh saat file berubah)
# ---------------------------------------------------------------------------
def _mtime(path: Path) -> float:
    return os.path.getmtime(path) if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_json(path_str: str, mtime: float) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Memuat dataset...")
def load_dataset(
    split_str: str,
    split_mtime: float,
    review_str: str,
    review_mtime: float,
    cleaned_str: str,
    cleaned_mtime: float,
) -> pd.DataFrame | None:
    split_path = Path(split_str)
    if not split_path.exists():
        return None

    df_split = pd.read_csv(split_path, dtype={"tweet_id": "string"})

    # Gabungkan dengan teks & kolom sentiment_label (jika ada) dari CSV sumber
    text_source = None
    for candidate in (Path(review_str), Path(cleaned_str)):
        if candidate.exists():
            tmp = pd.read_csv(candidate, dtype={"tweet_id": "string"}, low_memory=False)
            if "tweet_id" in tmp.columns:
                text_source = tmp
                break

    if text_source is not None:
        text_col = next(
            (c for c in ("cleaned_text", "clean_text", "raw_text") if c in text_source.columns),
            None,
        )
        merge_cols = ["tweet_id"]
        if text_col:
            merge_cols.append(text_col)
        if "sentiment_label" in text_source.columns and "sentiment_label" not in df_split.columns:
            merge_cols.append("sentiment_label")

        if len(merge_cols) > 1:
            merged = df_split.merge(
                text_source[merge_cols].drop_duplicates("tweet_id"),
                on="tweet_id",
                how="left",
            )
            if text_col:
                merged["teks_analisis"] = merged[text_col].fillna("").astype(str)
                if text_col == "raw_text" and preprocess_text is not None:
                    merged["teks_analisis"] = merged["teks_analisis"].map(preprocess_text)
            else:
                merged["teks_analisis"] = ""
            return merged

    df_split["teks_analisis"] = ""
    return df_split


def get_sentiment_series(df: pd.DataFrame) -> pd.Series:
    """Ambil label sentimen (Positif, Negatif, Netral) dari kolom sentiment_label atau mapping target_label."""
    mapped_target = (
        df["target_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(SENTIMENT_MAP)
        .fillna("Netral")
    )

    if "sentiment_label" in df.columns:
        raw_sent = df["sentiment_label"].fillna("").astype(str).str.strip().str.lower()
        mapped_sent = raw_sent.map(SENTIMENT_MAP)
        return mapped_sent.fillna(mapped_target)

    return mapped_target


@st.cache_data(show_spinner="Menghitung frekuensi kata...")
def word_frequencies(texts: tuple) -> dict:
    counter = Counter()
    for text in texts:
        for token in TOKEN_RE.findall(str(text).lower()):
            if token not in STOPWORDS_ID:
                counter[token] += 1
    return dict(counter)


# ---------------------------------------------------------------------------
# Halaman 1: Dashboard Overview
# ---------------------------------------------------------------------------
def render_dashboard(df: pd.DataFrame, meta: dict | None) -> None:
    st.header(
        "Analisis Sentimen Publik terhadap Pencantuman Isu LGBTQ+ sebagai Ancaman Nonmiliter dalam Perpres Nomor 111 Tahun 2025"
    )

    st.divider()
    
    st.subheader("Sumber Data")
    st.write("Postingan/Tweet publik di platform X/Twitter dengan query pencarian:")
    st.write('**"ancaman non militer" OR "ancaman nonmiliter" OR "perpres 111" OR "perpres no 111"**')
    st.write("**lgbt OR lgbtq**")
    st.write("")

    data_stats = (meta or {}).get("data_stats", {})
    total_raw = data_stats.get("total_rows")
    if total_raw is None and CLEANED_DATA_PATH.exists():
        total_raw = len(pd.read_csv(CLEANED_DATA_PATH))
    if total_raw is None:
        total_raw = len(df)

    total_clean = data_stats.get("usable_rows", len(df))
    pct_remaining = (total_clean / total_raw * 100) if total_raw else 0.0

    labels = df["target_label"].astype(str)
    classes = sorted(labels.unique())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data Komentar (Awal)", f"{total_raw:,}")
    c2.metric(
        "Data Bersih Tervalidasi",
        f"{total_clean:,}",
        f"{pct_remaining:.1f}% siap latih",
        delta_color="off",
        delta_arrow="off"
    )
    c3.metric("Jumlah Kelas Label", len(classes))
    st.write("")
    
    # Statistik Sentimen (Positif, Negatif, Netral) - Diagram Lingkaran
    st.subheader("Statistik Sentimen Publik")
    sent_series = get_sentiment_series(df)
    sentiment_order = ["Positif", "Negatif", "Netral"]
    sent_counts = sent_series.value_counts()

    sent_df = pd.DataFrame(
        {
            "Sentimen": sentiment_order,
            "Jumlah": [int(sent_counts.get(s, 0)) for s in sentiment_order],
        }
    )
    total_sent = sent_df["Jumlah"].sum()
    sent_df["Persentase"] = (
        (sent_df["Jumlah"] / total_sent * 100) if total_sent > 0 else 0.0
    )

    col_pie, col_stat = st.columns([1.4, 1])
    with col_pie:
        fig_pie = px.pie(
            sent_df,
            names="Sentimen",
            values="Jumlah",
            color="Sentimen",
            color_discrete_map={
                "Positif": "#2ECC71",
                "Negatif": "#E74C3C",
                "Netral": "#95A5A6",
            },
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="label+value+percent",
            hovertemplate="Sentimen: %{label}<br>Jumlah: %{value:,} tweet<br>Persentase: %{percent}<extra></extra>",
        )
        fig_pie.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_stat:
        for _, row in sent_df.iterrows():
            st.metric(
                label=f"Sentimen {row['Sentimen']}",
                value=f"{row['Jumlah']:,} tweet"
            )

    # Distribusi Kelas
    st.subheader("Distribusi Kelas Emosi / Target Label")
    dist = labels.value_counts().rename_axis("kelas").reset_index(name="jumlah")
    dist["persentase"] = dist["jumlah"] / dist["jumlah"].sum() * 100
    dist["label_teks"] = dist.apply(
        lambda r: f"{r['jumlah']:,} ({r['persentase']:.1f}%)", axis=1
    )

    fig_dist = px.bar(
        dist,
        x="kelas",
        y="jumlah",
        color="kelas",
        text="label_teks",
        labels={"kelas": "Kategori Emosi / Sentimen", "jumlah": "Jumlah Komentar"},
    )
    fig_dist.update_traces(textposition="outside")
    fig_dist.update_layout(showlegend=False, height=430)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Top Keywords
    st.subheader("Top Keywords")
    if df["teks_analisis"].str.strip().eq("").all():
        st.info(
            "Kolom teks tidak ditemukan di `output/cleaned_data_auto_label_review.csv` "
            "atau `cleaned_data.csv` untuk menghitung frekuensi kata."
        )
        return

    freq = word_frequencies(tuple(df["teks_analisis"]))
    if not freq:
        st.warning("Tidak ada kata yang dapat dihitung dari dataset.")
        return

    urutan = st.radio(
        "Urutkan berdasarkan frekuensi kata:",
        ["Terbanyak", "Paling Sedikit"],
        horizontal=True,
    )

    if urutan == "Terbanyak":
        items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:15]
        df_kw = pd.DataFrame(items, columns=["kata", "frekuensi"]).sort_values(
            by=["frekuensi", "kata"], ascending=[True, False]
        )
        chart_title = "15 Kata dengan Frekuensi Terbanyak"
    else:
        items = sorted(freq.items(), key=lambda x: (x[1], x[0]))[:15]
        df_kw = pd.DataFrame(items, columns=["kata", "frekuensi"]).sort_values(
            by=["frekuensi", "kata"], ascending=[False, False]
        )
        chart_title = "15 Kata dengan Frekuensi Paling Sedikit"

    fig_top = px.bar(
        df_kw,
        x="frekuensi",
        y="kata",
        orientation="h",
        text="frekuensi",
        title=chart_title,
        labels={"frekuensi": "Kemunculan", "kata": "Kata"},
    )
    fig_top.update_traces(textposition="outside")
    fig_top.update_layout(height=480)
    st.plotly_chart(fig_top, use_container_width=True)


# ---------------------------------------------------------------------------
# Halaman 2: Performa & Detail Model SVM
# ---------------------------------------------------------------------------
def render_model(meta: dict | None) -> None:
    st.header("Performa & Detail Model SVM")

    if meta is None:
        st.warning(
            "File `models/model_metadata.json` belum ditemukan. "
            "Jalankan `python -m svm.train_svm` terlebih dahulu."
        )
        return

    test_metrics = meta.get("test_metrics", {})
    per_class = test_metrics.get("per_class", {})

    avg_mode = st.radio(
        "Metode Rata-Rata Metrik Multikelas:",
        ["macro avg", "weighted avg"],
        horizontal=True,
        format_func=lambda x: "Macro Average (Seimbang antar kelas)" if x == "macro avg" else "Weighted Average (Proporsional jumlah data)",
    )
    avg_stats = per_class.get(avg_mode, {})

    acc = test_metrics.get("accuracy")
    prec = avg_stats.get("precision")
    rec = avg_stats.get("recall")
    f1 = test_metrics.get("macro_f1") if avg_mode == "macro avg" else test_metrics.get("weighted_f1")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (Test Set)", f"{acc * 100:.2f}%" if acc is not None else "-")
    c2.metric("Precision", f"{prec * 100:.2f}%" if prec is not None else "-")
    c3.metric("Recall", f"{rec * 100:.2f}%" if rec is not None else "-")
    c4.metric("F1-Score", f"{f1 * 100:.2f}%" if f1 is not None else "-")

    # Confusion Matrix
    st.subheader("Confusion Matrix (Test Set)")
    scheme = meta.get("scheme", "ekman7")
    cm_path = OUTPUT_DIR / f"svm_{scheme}_confusion_matrix.csv"

    if cm_path.exists():
        cm_df = pd.read_csv(cm_path, index_col=0)
        labels = list(cm_df.columns)
        z_vals = cm_df.values

        fig_cm = go.Figure(
            go.Heatmap(
                z=z_vals,
                x=labels,
                y=list(cm_df.index),
                text=z_vals,
                texttemplate="%{text}",
                colorscale="Blues",
                hovertemplate="Aktual: %{y}<br>Prediksi: %{x}<br>Jumlah: %{z}<extra></extra>",
            )
        )
        fig_cm.update_layout(
            xaxis_title="Prediksi Model SVM",
            yaxis_title="Label Aktual (Manual)",
            yaxis_autorange="reversed",
            height=500,
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info(f"File `{cm_path.name}` belum tersedia.")

    # Rincian Metrik Per Kelas & Info Konfigurasi
    col_detail, col_cfg = st.columns([1.2, 1])

    with col_detail:
        st.subheader("Performa per Kelas")
        class_rows = []
        for cls_name, vals in per_class.items():
            if cls_name in {"accuracy", "macro avg", "weighted avg"} or not isinstance(vals, dict):
                continue
            class_rows.append(
                {
                    "Kelas": cls_name,
                    "Precision": f"{vals.get('precision', 0) * 100:.1f}%",
                    "Recall": f"{vals.get('recall', 0) * 100:.1f}%",
                    "F1-Score": f"{vals.get('f1-score', 0) * 100:.1f}%",
                    "Jumlah Uji": int(vals.get("support", 0)),
                }
            )
        if class_rows:
            st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

    with col_cfg:
        st.subheader("Info Konfigurasi Model")
        cfg = meta.get("best_config", {})
        splits = meta.get("split_summary", {}).get("splits", {})
        split_str = " / ".join(
            f"{k.capitalize()} {v.get('pct', 0):.0f}%"
            for k, v in splits.items()
        ) or "50% / 20% / 30%"

        info_items = [
            ("Skema Klasifikasi", scheme),
            ("Konfigurasi Terpilih", cfg.get("name", "-")),
            ("Tipe / Kernel SVM", cfg.get("model_type", "-")),
            ("Parameter C", str(cfg.get("C", "-"))),
            ("Class Weight", str(cfg.get("class_weight", "-"))),
            ("Word N-Gram", str(cfg.get("word_ngram_range", "-"))),
            ("Max Features (Word / Char)", f"{cfg.get('max_word_features', '-')} / {cfg.get('max_char_features', '-')}"),
            ("Rasio Split (Group-Aware)", split_str),
            ("Cakupan Fit Akhir", meta.get("fit_scope", "-")),
            ("Validation Macro-F1", f"{meta.get('validation_best_macro_f1', 0) * 100:.2f}%"),
            ("Waktu Training (UTC)", str(meta.get("trained_at", "-"))[:19].replace("T", " ")),
        ]
        st.table(pd.DataFrame(info_items, columns=["Parameter", "Nilai"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ["Dashboard Overview", "Performa & Detail Model SVM"],
    )

    meta = load_json(str(METADATA_PATH), _mtime(METADATA_PATH))

    if page == "Dashboard Overview":
        df = load_dataset(
            str(SPLIT_PATH), _mtime(SPLIT_PATH),
            str(REVIEW_DATA_PATH), _mtime(REVIEW_DATA_PATH),
            str(CLEANED_DATA_PATH), _mtime(CLEANED_DATA_PATH),
        )
        if df is None:
            st.error(
                "File `output/dataset_split.csv` belum ditemukan. "
                "Jalankan `python -m svm.train_svm` terlebih dahulu."
            )
            return
        render_dashboard(df, meta)
    else:
        render_model(meta)


if __name__ == "__main__":
    main()