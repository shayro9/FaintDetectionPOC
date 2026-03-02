"""
Smartwatch Data Visualizer
==========================
Usage:
    python visualize.py --csv smartwatch_dataset.csv
    python visualize.py --csv smartwatch_dataset.csv --session event_0001
    python visualize.py --csv smartwatch_dataset.csv --npz smartwatch_windows.npz
"""

import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


LABEL_COLORS = {
    "normal":      "#4CAF50",
    "pre_syncope": "#FF9800",
    "syncope":     "#F44336",
    "recovery":    "#2196F3",
}

SENSOR_COLS = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x",  "gyro_y",  "gyro_z",
    "heart_rate", "spo2", "skin_temp",
]


# ─────────────────────────────────────────────────────────────
# 1. FULL SESSION PLOT
# ─────────────────────────────────────────────────────────────
def plot_session(df: pd.DataFrame, session_id: str):
    """Plot all 9 sensors for one session with color-coded label bands."""
    sess = df[df["session_id"] == session_id].copy()
    if sess.empty:
        print(f"Session '{session_id}' not found.")
        return

    t = sess["timestamp"].values
    labels = sess["label"].values

    fig = make_subplots(
        rows=9, cols=1,
        shared_xaxes=True,
        subplot_titles=SENSOR_COLS,
        vertical_spacing=0.03,
    )

    for i, col in enumerate(SENSOR_COLS, start=1):
        fig.add_trace(
            go.Scatter(x=t, y=sess[col].values, mode="lines",
                       line=dict(width=1), name=col, showlegend=False),
            row=i, col=1,
        )

    # Add colored background bands for each phase
    labels = sess["label"].astype(str).to_numpy()
    phase_changes = np.where(labels[1:] != labels[:-1])[0]
    boundaries = np.concatenate([[0], phase_changes + 1, [len(t)]])
    added_labels = set()

    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        lbl = labels[start_idx]
        color = LABEL_COLORS.get(lbl, "#888888")
        show = lbl not in added_labels
        added_labels.add(lbl)

        fig.add_vrect(
            x0=t[start_idx], x1=t[min(end_idx, len(t)-1)],
            fillcolor=color, opacity=0.15, line_width=0,
            annotation_text=lbl if show else "",
            annotation_position="top left",
            annotation_font_size=10,
        )

    activity = sess["activity"].iloc[0]
    has_event = sess["has_event"].iloc[0]
    fig.update_layout(
        title=f"Session: {session_id}  |  Activity: {activity}  |  Event: {has_event}",
        height=1200,
        template="plotly_dark",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.show()


# ─────────────────────────────────────────────────────────────
# 2. LABEL DISTRIBUTION
# ─────────────────────────────────────────────────────────────
def plot_label_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(1)

    fig = px.bar(
        counts, x="label", y="count",
        color="label",
        color_discrete_map=LABEL_COLORS,
        text=counts["pct"].astype(str) + "%",
        title="Label Distribution (sample level)",
        template="plotly_dark",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=450)
    fig.show()


# ─────────────────────────────────────────────────────────────
# 3. SENSOR STATS BY LABEL
# ─────────────────────────────────────────────────────────────
def plot_sensor_by_label(df: pd.DataFrame, sensor: str = "heart_rate"):
    sample = df.groupby("label").apply(lambda x: x.sample(min(2000, len(x)), random_state=42)).reset_index()

    fig = px.box(
        sample, x="label", y=sensor,
        color="label",
        color_discrete_map=LABEL_COLORS,
        title=f"{sensor} distribution by label",
        template="plotly_dark",
        points=False,
    )
    fig.update_layout(showlegend=False, height=450)
    fig.show()


# ─────────────────────────────────────────────────────────────
# 4. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame):
    sample = df[SENSOR_COLS].dropna().sample(min(10000, len(df)), random_state=42)
    corr = sample.corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title="Sensor Correlation Matrix",
        template="plotly_dark",
        height=600, width=700,
    )
    fig.show()


# ─────────────────────────────────────────────────────────────
# 5. WINDOW CLASS DISTRIBUTION (from .npz)
# ─────────────────────────────────────────────────────────────
def plot_window_distribution(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    y = data["y"]

    label_map = {0: "normal", 1: "pre_syncope", 2: "syncope", 3: "recovery"}
    counts = {label_map[i]: int((y == i).sum()) for i in range(4)}
    df_w = pd.DataFrame(list(counts.items()), columns=["label", "count"])
    df_w["pct"] = (df_w["count"] / len(y) * 100).round(1)

    fig = px.bar(
        df_w, x="label", y="count",
        color="label",
        color_discrete_map=LABEL_COLORS,
        text=df_w["pct"].astype(str) + "%",
        title=f"Window Label Distribution  (total windows: {len(y):,})",
        template="plotly_dark",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=450)
    fig.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Smartwatch data visualizer")
    parser.add_argument("--csv",     required=True,  help="Path to smartwatch_dataset.csv")
    parser.add_argument("--npz",     default=None,   help="Path to smartwatch_windows.npz (optional)")
    parser.add_argument("--session", default=None,   help="Session ID to plot (e.g. event_0001). If omitted, picks a random event session.")
    parser.add_argument("--sensor",  default="heart_rate", help="Sensor for box plot (default: heart_rate)")
    args = parser.parse_args()

    print("Loading CSV...")
    df = pd.read_csv(args.csv)
    print(f"  Loaded {len(df):,} rows, {df['session_id'].nunique()} sessions")

    # Pick session to plot
    session_id = args.session
    if session_id is None:
        event_sessions = df[df["has_event"] == True]["session_id"].unique()
        session_id = str(np.random.choice(event_sessions))
        print(f"  No session specified, picking random event session: {session_id}")

    print("\n[1/5] Plotting session signals...")
    plot_session(df, session_id)

    print("[2/5] Plotting label distribution...")
    plot_label_distribution(df)

    print("Columns right before plot_sensor_by_label:", list(df.columns))
    print("Has label?", "label" in df.columns)
    print("[3/5] Plotting sensor by label...")
    plot_sensor_by_label(df, sensor=args.sensor)

    print("[4/5] Plotting correlation heatmap...")
    plot_correlation(df)

    if args.npz:
        print("[5/5] Plotting window distribution...")
        plot_window_distribution(args.npz)
    else:
        print("[5/5] Skipping window distribution (no --npz provided)")

    print("\nDone. All plots opened in browser.")


if __name__ == "__main__":
    main()
