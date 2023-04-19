import os
from revChatGPT.V3 import Chatbot
from paddleocr import PaddleOCR
import gradio as gr
from PIL import Image
import numpy as np

# 连接open ai
chatGPT = Chatbot(api_key=os.environ["openai_api_key"])

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def add_file(history, file):
    result = ocr.ocr(np.array(Image.open(file.name).convert('RGB')), cls=True)
    paragraph = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            paragraph.append(line[-1][0])
    question = ""
    if paragraph:
        question = "\n".join(paragraph)
    history = history + [((file.name, question), None)]
    return history


def bot(history):
    if isinstance(history[-1][0], (str,)):
        question = history[-1][0]
    else:
        question = history[-1][0][1]
    history[-1][1] = ""
    for data in chatGPT.ask_stream(question):
        history[-1][1] += data
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)
    txt = gr.Textbox(
        show_label=False,
        placeholder="在这里输入文字点击发送或者直接上传图片",
    ).style(container=False)
    with gr.Row():
        with gr.Column(scale=0.5, min_width=0):
            send_btn = gr.Button("send", variant="primary")
        with gr.Column(scale=0.5, min_width=0):
            upload_btn = gr.UploadButton("📁", file_types=["image"])
    with gr.Row():
        gr.Markdown("- OpenAI ChatGPT3.5模型，github开源PaddleOCR、acheong08/ChatGPT；\n- 注：上传的图片需带有文字描述的问题，类似小猿搜题。")

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(bot, chatbot, chatbot)
    upload_btn.upload(add_file, [chatbot, upload_btn], [chatbot]).then(bot, chatbot, chatbot)
    send_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(bot, chatbot, chatbot)

demo.queue(concurrency_count=10).launch(share=True, debug=True)