import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

text_summary = pipeline(
    "summarization", model="Falconsai/text_summarization", dtype=torch.bfloat16
)


def summarize(input_text):
    summary_text = text_summary(input_text)
    return summary_text[0]["summary_text"]


app = gr.Interface(
    fn=summarize,
    inputs=[gr.Textbox(label="Input text to summarize")],
    outputs=[gr.Textbox(label="Summarized Text")],
    title="Text Summarizer",
    description="GenAI Text Summarizer",
)

app.launch()
