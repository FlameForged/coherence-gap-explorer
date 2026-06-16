import gradio as gr
import pandas as pd
import os
from datetime import date
from huggingface_hub import HfApi

CSV_PATH = "observations.csv"
REPO_ID = "FlameForged/coherence-gap-explorer"
HF_TOKEN = os.environ.get("HF_TOKEN")

def load_data():
    return pd.read_csv(CSV_PATH, on_bad_lines='skip', engine='python')

def get_choices(df, column):
    if column not in df.columns:
        return ["All"]
    vals = sorted(df[column].dropna().unique().tolist())
    return ["All"] + [str(v) for v in vals]

def filter_data(platform, behavior_category, confidence):
    df = load_data()
    if platform != "All":
        df = df[df["platform"].astype(str) == platform]
    if behavior_category != "All":
        df = df[df["behavior_category"].astype(str) == behavior_category]
    if confidence != "All":
        df = df[df["interpretive_confidence"].astype(str) == confidence]
    return df

def submit_observation(
    f_date, platform, model_version, memory_enabled,
    window_type, prompt_class, behavior_category,
    description, evidence, screenshot_ids,
    raw_prompt, raw_response, confidence,
    alternative_explanations, reproducibility_status, coder_notes
):
    new_row = {
        "date": f_date, "platform": platform,
        "model_version": model_version, "memory_enabled": memory_enabled,
        "window_type": window_type, "prompt_class": prompt_class,
        "behavior_category": behavior_category, "description": description,
        "evidence": evidence, "screenshot_ids": screenshot_ids,
        "raw_prompt_available": raw_prompt, "raw_response_available": raw_response,
        "interpretive_confidence": confidence,
        "alternative_explanations": alternative_explanations,
        "reproducibility_status": reproducibility_status,
        "coder_notes": coder_notes
    }

    df = load_data()
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(CSV_PATH, index=False)

    if HF_TOKEN:
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=CSV_PATH,
            path_in_repo="observations.csv",
            repo_id=REPO_ID,
            repo_type="space"
        )
        return "✅ Observation submitted and saved permanently!"
    else:
        return "⚠️ Saved locally but HF_TOKEN not set — won't persist after restart."

initial_df = load_data()

with gr.Blocks(title="Coherence Gap Explorer") as demo:
    gr.Markdown("# Coherence Gap Explorer")
    gr.Markdown(
        "Interactive browser for the observational dataset from the Coherence Gap paper."
    )

    with gr.Tab("Browse & Filter"):
        with gr.Row():
            platform_dd = gr.Dropdown(
                choices=get_choices(initial_df, "platform"),
                value="All", label="Platform"
            )
            category_dd = gr.Dropdown(
                choices=get_choices(initial_df, "behavior_category"),
                value="All", label="Behavior Category"
            )
            confidence_dd = gr.Dropdown(
                choices=get_choices(initial_df, "interpretive_confidence"),
                value="All", label="Interpretive Confidence"
            )

        table = gr.Dataframe(
            value=initial_df, label="Observations",
            interactive=False, wrap=True
        )

        for dd in [platform_dd, category_dd, confidence_dd]:
            dd.change(
                fn=filter_data,
                inputs=[platform_dd, category_dd, confidence_dd],
                outputs=table
            )

    with gr.Tab("Submit Observation"):
        gr.Markdown("### Add a New Observation")

        with gr.Row():
            f_date = gr.Textbox(label="Date", value=str(date.today()), placeholder="YYYY-MM-DD")
            f_platform = gr.Textbox(label="Platform")
            f_model_version = gr.Textbox(label="Model Version")

        with gr.Row():
            f_memory = gr.Dropdown(choices=["True", "False", "Unknown"], label="Memory Enabled")
            f_window = gr.Textbox(label="Window Type")
            f_prompt_class = gr.Textbox(label="Prompt Class")
            f_category = gr.Textbox(label="Behavior Category")

        with gr.Row():
            f_raw_prompt = gr.Dropdown(choices=["True", "False", "Unknown"], label="Raw Prompt Available")
            f_raw_response = gr.Dropdown(choices=["True", "False", "Unknown"], label="Raw Response Available")
            f_confidence = gr.Dropdown(choices=["Low", "Medium", "High"], label="Interpretive Confidence")
            f_repro = gr.Textbox(label="Reproducibility Status")

        f_description = gr.Textbox(label="Description", lines=3)
        f_evidence = gr.Textbox(label="Evidence", lines=3)
        f_alternative = gr.Textbox(label="Alternative Explanations", lines=2)
        f_screenshots = gr.Textbox(label="Screenshot IDs")
        f_notes = gr.Textbox(label="Coder Notes", lines=2)

        submit_btn = gr.Button("Submit Observation", variant="primary")
        status = gr.Markdown("")

        submit_btn.click(
            fn=submit_observation,
            inputs=[
                f_date, f_platform, f_model_version, f_memory,
                f_window, f_prompt_class, f_category,
                f_description, f_evidence, f_screenshots,
                f_raw_prompt, f_raw_response, f_confidence,
                f_alternative, f_repro, f_notes
            ],
            outputs=status
        )

demo.launch()
