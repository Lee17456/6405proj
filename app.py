"""English Streamlit interface; run from the project root."""
import hashlib
import html
import json
import os
import time

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_CHARS = 400000
MODELS = {"svm": "TF-IDF + SVM", "roberta": "RoBERTa", "mamba": "Mamba", "longformer": "Longformer"}
DATASETS = {"ag_news": "AG News", "imdb": "IMDb", "scotus": "SCOTUS"}
TASKS = {"ag_news": "News topic classification", "imdb": "Movie review sentiment", "scotus": "Legal topic classification"}
EXAMPLES = {
    "ag_news": "The company announced a new computer chip and plans to expand its software business next year.",
    "imdb": "The film starts slowly, but the acting is excellent and the ending is deeply moving. I enjoyed it.",
    "scotus": "The Court considered whether the search violated the protections of the Fourth Amendment.",
}
# Original label values from the API, not zero-based class IDs.
# https://scdb.la.psu.edu/online-codebook/issue-area/
SCOTUS_LABELS = {
    "1": "Criminal Procedure", "2": "Civil Rights", "3": "First Amendment", "4": "Due Process",
    "5": "Privacy", "6": "Attorneys", "7": "Unions", "8": "Economic Activity", "9": "Judicial Power",
    "10": "Federalism", "11": "Interstate Relations", "12": "Federal Taxation",
    "13": "Miscellaneous", "14": "Private Action",
}


def display_label(dataset, label):
    raw = str(label)
    return f"{SCOTUS_LABELS.get(raw, 'Unknown category')} · code {raw}" if dataset == "scotus" else raw


def model_name(run):
    return MODELS.get(run["model"], run["model"])


def study_name(run):
    if "_long_" in run["name"]:
        return "Length study"
    if "_core_" in run["name"]:
        return "Core study"
    return run["name"]


def run_label(run):
    return f"{model_name(run)} · seed {run['seed']} · {study_name(run)}"


def view_label(view):
    if view == "native":
        return "Native input (model-specific limits)"
    if view.startswith("shared") and view[6:].isdigit():
        return f"Shared prefix · {int(view[6:]):,} token budget"
    return view


def ordered_runs(records):
    order = list(MODELS)
    return sorted(records, key=lambda r: (order.index(r["model"]) if r["model"] in order else 99,
                                         0 if "_core_" in r["name"] else 1,
                                         0 if r["seed"] == 42 else 1, r["name"]))


def default_runs(records):
    seen, names = set(), []
    for run in ordered_runs(records):
        if run["model"] not in seen:
            names.append(run["name"])
            seen.add(run["model"])
    return names[:3]


def get(path):
    response = requests.get(API + path, timeout=15)
    response.raise_for_status()
    return response.json()


def connection_error(message, exc):
    st.error(message)
    with st.expander("Connection details"):
        st.code(f"API: {API}\n{exc}", language="text")


def csv_download(label, frame, filename, key):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"), filename,
                       "text/csv", key=key, use_container_width=True)


def number(value, digits=1):
    return "—" if value is None or pd.isna(value) else f"{value:,.{digits}f}"


def signature(dataset, names, text):
    return hashlib.sha256(json.dumps([dataset, names, text], ensure_ascii=False).encode()).hexdigest()


def use_example(dataset):
    st.session_state[f"text_{dataset}"] = EXAMPLES.get(dataset, "")
    remember_text(dataset)


def remember_text(dataset):
    st.session_state[f"draft_{dataset}"] = st.session_state[f"text_{dataset}"]


st.set_page_config(page_title="Text Classification Lab", page_icon="◈", layout="wide")
st.markdown("""
<style>
 .block-container {max-width:1440px; padding:2.8rem 2rem 2rem;}
 [data-testid="stSidebar"] {border-right:1px solid #dce4ef;}
 h1 {font-size:2.25rem !important; letter-spacing:-.065rem; font-weight:750 !important;}
 h2,h3 {letter-spacing:-.025rem;}
 h3 {font-size:1.3rem !important;}
 .eyebrow {color:#42628b; font-size:.73rem; font-weight:700; letter-spacing:.16em; margin-bottom:.5rem;}
 .sidebar-brand {font-size:1.2rem; font-weight:750; letter-spacing:-.03rem; margin-bottom:.3rem;}
 .hero-description {color:#53667e; font-size:1.03rem; margin:0 0 1.4rem;}
 .connection {display:inline-flex; align-items:center; gap:.45rem; color:#087465;
   background:#e6f5f0; border:1px solid #b9e2d5; padding:.35rem .7rem; border-radius:2rem;
   font-size:.8rem; font-weight:600; margin:.4rem 0 1rem;}
 .connection-dot {width:7px;height:7px;border-radius:50%;background:#11947e;}
 [data-testid="stMetric"] {background:#ffffff; border:1px solid #dce4ef;
   border-radius:12px; padding:.8rem 1rem;}
 [data-testid="stMetricValue"] {font-size:1.6rem;}
 [data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
 [data-baseweb="tab-list"] {gap:1.4rem; margin:.8rem 0 1rem;}
 button[data-baseweb="tab"] {font-size:.95rem; padding-bottom:.65rem;}
 button[kind="primary"] {border-radius:9px; min-height:2.8rem; font-weight:600;}
 .prediction {font-size:1.25rem;font-weight:700;line-height:1.4;margin:.45rem 0 1.2rem;overflow-wrap:anywhere;}
 .prediction-model {font-size:.76rem;font-weight:700;letter-spacing:.06em;color:#42628b;}
 @media (max-width:760px) {.block-container {padding:2rem 1rem;} h1 {font-size:1.8rem !important;}}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">6405 / NLP EXPERIMENTS</div>', unsafe_allow_html=True)
st.title("Text Classification Lab")
st.markdown('<p class="hero-description">Compare predictions, classification quality and compute cost across model architectures.</p>', unsafe_allow_html=True)

try:
    runs = get("/runs")
except (requests.RequestException, ValueError) as exc:
    connection_error("The inference service is unavailable. Start the backend, then retry.", exc)
    if st.button("Retry connection", key="retry_connection"):
        st.rerun()
    st.stop()
if not runs:
    st.info("No trained models are available yet. Complete a training run, then refresh this page.")
    if st.button("Refresh models", key="refresh_empty"):
        st.rerun()
    st.stop()

with st.sidebar:
    st.markdown('<div class="sidebar-brand">◈ Classification Lab</div>', unsafe_allow_html=True)
    st.caption("Your experiment workspace")
    st.markdown('<span class="connection"><span class="connection-dot"></span>Backend connected</span>', unsafe_allow_html=True)
    dataset = st.selectbox("Dataset", sorted({r["dataset"] for r in runs}),
                           format_func=lambda d: DATASETS.get(d, d), key="dataset")
    st.caption(TASKS.get(dataset, "Text classification"))
    dataset_runs = [r for r in runs if r["dataset"] == dataset]
    views = sorted({r["view"] for r in dataset_runs}, key=lambda v: (v != "shared512", v))
    view = st.selectbox("Input view", views, format_func=view_label, key=f"view_{dataset}",
                        help="This filter applies to all three tabs. A shared prefix gives models the same text content.")
    st.caption("SVM reads the full document. Neural models use their saved input limits." if view == "native"
               else "Models read the same text prefix. Actual token counts differ by tokenizer.")
    st.divider()
    st.markdown("**A simple workflow**")
    st.caption("1. Choose a dataset and input view.\n\n2. Select models and enter a document.\n\n3. Compare predictions, then inspect test results.")
    if st.button("Refresh experiments", key="refresh_runs", use_container_width=True):
        st.rerun()
    with st.expander("Service details"):
        st.code(API, language="text")
        st.caption("Predictions run sequentially. Switching models may take a moment.")

available = ordered_runs([r for r in dataset_runs if r["view"] == view])
by_name = {r["name"]: r for r in available}
names = list(by_name)
selection_labels = {r["name"]: model_name(r) if sum(other["model"] == r["model"] for other in available) == 1
                    else run_label(r) for r in available}
overview = st.columns(3)
overview[0].metric("Dataset", DATASETS.get(dataset, dataset))
overview[1].metric("Available architectures", len({r["model"] for r in available}))
overview[2].metric("Trained configurations", len(available))
live, charts, errors = st.tabs(["Classify", "Results & efficiency", "Error explorer"])

with live:
    left, right = st.columns([1.4, 1], gap="large")
    valid_input = True
    with left, st.container(border=True):
        st.subheader("1. Add a document")
        mode = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True,
                        key=f"input_mode_{dataset}", label_visibility="collapsed")
        if mode == "Paste text":
            if f"text_{dataset}" not in st.session_state:
                st.session_state[f"text_{dataset}"] = st.session_state.get(f"draft_{dataset}", "")
            text = st.text_area("English document", height=230, max_chars=MAX_CHARS,
                                placeholder="Paste an English document here…", key=f"text_{dataset}",
                                on_change=remember_text, args=(dataset,))
            st.button("Use example text", key=f"example_{dataset}", on_click=use_example, args=(dataset,))
        else:
            uploaded = st.file_uploader("Choose a UTF-8 .txt file", type=["txt"], key=f"upload_{dataset}")
            text = ""
            if uploaded is not None:
                try:
                    if uploaded.size > MAX_CHARS * 4 + 3:
                        raise ValueError("File is too large. Use a document with at most 400,000 characters.")
                    text = uploaded.getvalue().decode("utf-8-sig")
                    if len(text) > MAX_CHARS:
                        raise ValueError("File exceeds the 400,000-character limit.")
                    with st.expander("Preview uploaded text"):
                        st.text(text[:2000])
                except (UnicodeDecodeError, ValueError) as exc:
                    st.error("Use a UTF-8 encoded .txt file." if isinstance(exc, UnicodeDecodeError) else str(exc))
                    valid_input = False
        st.caption(f"{len(text):,} / {MAX_CHARS:,} characters · English text")
        st.caption("Example text is for trying the interface. Use exported test documents for your presentation.")
    with right, st.container(border=True):
        st.subheader("2. Choose models")
        selected = st.multiselect("Trained configurations", names, default=default_runs(available),
                                  format_func=lambda n: selection_labels[n], key=f"models_{dataset}_{view}")
        st.caption(f"{len(selected)} selected · Models run one at a time")
        submitted = st.button("Compare predictions", type="primary", key="compare", use_container_width=True,
                              disabled=not selected or not text.strip() or not valid_input)
        with st.expander("Configuration details"):
            details = pd.DataFrame([{"Model": model_name(r), "Study": study_name(r), "Token limit": r.get("max_length"),
                                     "Seed": r["seed"], "Training documents": r.get("n_train"), "Run ID": r["name"]}
                                    for r in available if r["name"] in selected])
            st.dataframe(details, hide_index=True, use_container_width=True)
        st.caption("The first prediction may take longer while weights load. Loading time is reported separately.")
        if dataset == "scotus":
            with st.expander("Legal topic reference"):
                labels = list(dict.fromkeys(str(label) for r in available for label in (r.get("label_names") or [])))
                st.dataframe(pd.DataFrame([{"Code": label, "Topic": SCOTUS_LABELS.get(label, "Unknown category")}
                                           for label in labels]), hide_index=True, use_container_width=True)
                st.markdown("Label codes follow the [SCDB issue-area reference](https://scdb.la.psu.edu/online-codebook/issue-area/).")
    current_signature = signature(dataset, selected, text)
    if submitted:
        output, failures = [], []
        progress = st.progress(0, text="Preparing models…")
        for i, name in enumerate(selected):
            progress.progress(i / len(selected), text=f"Running {model_name(by_name[name])} ({i + 1}/{len(selected)})…")
            started = time.perf_counter()
            try:
                response = requests.post(API + "/predict", json={"run": name, "text": text}, timeout=600)
                response.raise_for_status()
                result = response.json()
                output.append({"Model": model_name(by_name[name]), "Prediction": display_label(dataset, result["labels"][0]),
                               "Input tokens": result["input_tokens"][0], "Model time (ms)": result["model_ms"],
                               "Preprocessing (ms)": result["preprocess_ms"], "Local pipeline (ms)": result["local_pipeline_ms"],
                               "Load & warm-up (ms)": result["load_and_warmup_ms"],
                               "API round trip (ms)": (time.perf_counter() - started) * 1000, "Run ID": name})
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                failures.append(f"{run_label(by_name[name])}: prediction failed ({type(exc).__name__}). Check the backend log and retry.")
        progress.empty()
        st.session_state[f"prediction_{dataset}"] = {"signature": current_signature, "rows": output, "failures": failures}
    saved = st.session_state.get(f"prediction_{dataset}")
    st.subheader("3. Compare predictions")
    if saved:
        if saved["signature"] != current_signature:
            st.info("Input or model selection changed. Results below are from the previous request; compare again to refresh.")
        for failure in saved["failures"]:
            st.error(failure)
        output = saved["rows"]
        for start in range(0, len(output), 3):
            cards = st.columns(min(3, len(output) - start))
            for card, row in zip(cards, output[start:start + 3]):
                with card, st.container(border=True):
                    st.markdown(f'<div class="prediction-model">{html.escape(row["Model"])}</div>'
                                f'<div class="prediction">{html.escape(row["Prediction"])}</div>', unsafe_allow_html=True)
                    st.metric("Model time", f'{number(row["Model time (ms)"])} ms')
                    st.caption(f'Input tokens: {number(row["Input tokens"], 0)}')
                    st.caption(f'Load & warm-up: {number(row["Load & warm-up (ms)"])} ms')
                    matching = next((r for r in runs if r["name"] == row["Run ID"]), None)
                    st.caption(run_label(matching) if matching else row["Run ID"])
        if output:
            with st.expander("Timing breakdown & download"):
                frame = pd.DataFrame(output)
                st.dataframe(frame, hide_index=True, use_container_width=True)
                csv_download("Download predictions", frame, "predictions.csv", "download_predictions")
            st.caption("Live timings are single observations. API time includes server waiting and model loading, but excludes browser rendering. Use offline benchmarks for performance claims.")
    else:
        st.info("Add text and choose at least one model, then select Compare predictions.")

with charts:
    st.subheader("Classification quality")
    st.caption(f"{DATASETS.get(dataset, dataset)} · {view_label(view)} · Full test evaluations only")
    try:
        results = [r for r in get("/results") if r["name"] in by_name]
    except (requests.RequestException, ValueError) as exc:
        connection_error("Evaluation results could not be loaded. Live classification is still available.", exc)
        results = []
    quality, timing = [], []
    for result in results:
        metrics = result.get("test_metrics")
        if metrics and metrics.get("full_split", True):
            quality.append({"Configuration": run_label(result), "Model": model_name(result), "Seed": result["seed"],
                            "Macro-F1": metrics["macro_f1"], "Macro precision": metrics["macro_precision"],
                            "Macro recall": metrics["macro_recall"], "Accuracy": metrics["accuracy"],
                            "Test documents": metrics.get("n"), "Training documents": result.get("n_train"), "Run ID": result["name"]})
        for bench in result.get("benchmarks", []):
            timing.append({"Configuration": run_label(result), "Batch size": bench["batch_size"], "Precision": bench["precision"],
                           "Device": bench.get("device", "not recorded"), "Model time (ms/doc)": bench["model_ms_per_document"],
                           "Pipeline throughput (docs/s)": bench["pipeline_documents_per_second"],
                           "Peak GPU memory (MiB)": bench["peak_allocated_mb"], "Mean input tokens": bench["mean_actual_input_tokens"],
                           "Sample group": bench.get("sample_ids_signature", "not recorded"), "Run ID": result["name"]})
    if quality:
        frame = pd.DataFrame(quality)
        metric = st.selectbox("Compare metric", ["Macro-F1", "Macro precision", "Macro recall", "Accuracy"], key="quality_metric")
        plot = frame.assign(Score=frame[metric] * 100)
        chart = alt.Chart(plot).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Score:Q", title=f"{metric} (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Configuration:N", sort="-x", title=None, axis=alt.Axis(labelLimit=450)),
            color=alt.Color("Model:N", legend=None, scale=alt.Scale(range=["#2b62d9", "#12a594", "#7584c9", "#d68c2b"])),
            tooltip=["Configuration:N", alt.Tooltip("Score:Q", format=".2f"), "Run ID:N"]
        ).properties(height=max(160, len(plot) * 44))
        st.altair_chart(chart, use_container_width=True)
        display = frame.copy()
        for col in ("Macro-F1", "Macro precision", "Macro recall", "Accuracy"):
            display[col] = display[col].map(lambda v: f"{v:.2%}")
        st.dataframe(display, hide_index=True, use_container_width=True)
        csv_download("Download filtered test results", frame, f"test_results_{dataset}_{view}.csv", "download_results")
        st.caption("Each row is one trained configuration. Single-seed scores do not estimate variation across training seeds.")
    else:
        st.info("No full-test results for this selection yet. Run evaluation after training; pilot subset scores are excluded.")
    st.divider()
    st.subheader("Inference efficiency")
    if timing:
        bench_frame = pd.DataFrame(timing)
        col1, col2 = st.columns(2)
        batch = col1.selectbox("Batch size", sorted(bench_frame["Batch size"].unique().tolist()), key="benchmark_batch")
        precision = col2.selectbox("Precision", ["All recorded", *sorted(bench_frame["Precision"].unique())], key="benchmark_precision")
        filtered = bench_frame[bench_frame["Batch size"] == batch]
        if precision != "All recorded":
            filtered = filtered[filtered["Precision"] == precision]
        st.dataframe(filtered, hide_index=True, use_container_width=True)
        csv_download("Download filtered benchmarks", filtered, f"benchmarks_{dataset}_{view}.csv", "download_benchmarks")
        st.caption("Lower latency is faster; higher throughput is better. Compare matching sample groups, batch sizes and precision. CPU SVM has no GPU memory measurement. Local pipeline timing excludes model loading and network requests.")
    else:
        st.info("No benchmark measurements for this selection yet. Run the offline benchmark after evaluation.")

with errors:
    st.subheader("Explore misclassifications")
    st.caption("Inspect exported errors from the selected dataset and input view. The API returns at most 30 cases per run.")
    chosen = st.selectbox("Model configuration", names, format_func=lambda n: run_label(by_name[n]), key=f"error_run_{dataset}_{view}")
    try:
        mistakes = get(f"/errors/{chosen}")
    except (requests.RequestException, ValueError) as exc:
        connection_error("Error examples could not be loaded. Retry when the backend is available.", exc)
        mistakes = []
    if mistakes:
        case = st.selectbox("Document", list(range(len(mistakes))), key=f"error_case_{chosen}",
                            format_func=lambda i: str(mistakes[i].get("id", f"Case {i + 1}")))
        row = mistakes[case]
        expected, predicted = st.columns(2)
        with expected, st.container(border=True):
            st.caption("EXPECTED LABEL")
            st.write(display_label(dataset, row.get("true_label", "Unknown")))
        with predicted, st.container(border=True):
            st.caption("PREDICTED LABEL")
            st.write(display_label(dataset, row.get("predicted_label", "Unknown")))
        with st.container(border=True):
            st.markdown("**Document preview**")
            st.text(row.get("text_preview") or "No text preview was exported.")
            st.caption(f"Sample ID: {row.get('id', '—')} · Input characters: {row.get('input_chars', '—')}")
            st.caption("This preview contains up to the first 600 characters of the evaluated input, not the full original document.")
        with st.expander("All exported cases"):
            frame = pd.DataFrame(mistakes)
            for col in ("true_label", "predicted_label"):
                if col in frame:
                    frame[col] = frame[col].map(lambda v: display_label(dataset, v))
            frame = frame.rename(columns={"id": "Sample ID", "true_label": "Expected label", "predicted_label": "Predicted label",
                                          "true_id": "Expected class ID", "predicted_id": "Predicted class ID", "correct": "Correct",
                                          "input_tokens": "Input tokens", "input_chars": "Input characters", "coverage_chars": "Character coverage",
                                          "text_preview": "Text preview"})
            st.dataframe(frame, hide_index=True, use_container_width=True)
            csv_download("Download exported errors", frame, "error_examples.csv", "download_errors")
    else:
        st.info("No exported errors are available for this run. Evaluation may be pending, or the exported predictions contain no errors.")

st.divider()
st.caption("6405 · Text Classification Lab    |    Measured results only · TF-IDF + SVM / RoBERTa / Mamba / Longformer")
