import os
from revChatGPT.V3 import Chatbot
from paddleocr import PaddleOCR
import gradio as gr
from PIL import Image
import numpy as np
from paddlespeech.cli.tts.infer import TTSExecutor

# 连接open ai
chatGPT = Chatbot(api_key=os.environ["openai_api_key"])

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

MAX_FILE_SIZE = 3 * 1024 * 1024  # 单位字节，此处为2M

# 加载paddlespeech模型
audio_number = 0  # 语音文件编号
tts = TTSExecutor()
tts(
    text="initailize model.",
    output=f'output{audio_number}.wav',
    am='fastspeech2_mix',
    am_config=None,
    am_ckpt=None,
    am_stat=None,
    spk_id=0,
    phones_dict=None,
    tones_dict=None,
    speaker_dict=None,
    voc='hifigan_ljspeech',
    voc_config=None,
    voc_ckpt=None,
    voc_stat=None,
    lang='mix',
)  # 参数配置参见：https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/text_to_speech/README_cn.md
# 发现目前每个TTSExecutor()配置参数后，不能再更改，需要重新创建一个对象
# 另外cpu下极慢，需prompt说回答不要超过多少字
audio_number += 1


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def add_file(history, file):
    if os.path.getsize(file.name) > MAX_FILE_SIZE:
        history = history + [("暂不支持超过3M的图片", None)]
        return history
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


def bot(history, use_speech=False):
    if isinstance(history[-1][0], (str,)):
        question = history[-1][0]
        if question.startswith("暂不支持"):
            question = None
    elif isinstance(history[-1][0], (list, tuple)):
        question = history[-1][0][1]
    elif isinstance(history[-1][0], (dict, )):
        question = history[-1][0]["alt_text"]
    else:
        question = None
    print(f"question: {question}")
    if not question:
        history[-1][1] = "没有读取到问题"
        yield history
    else:
        history[-1][1] = ""
        for data in chatGPT.ask_stream(question):
            history[-1][1] += data
            yield history
        # 音频暂不支持stream，因此等输出完之后再生成
        if use_speech and len(history[-1][1].split(" ")) < 100:
            print("audio generating...")
            global audio_number
            tts(
              text=history[-1][1],
              output=f'output{audio_number}.wav',
              am='fastspeech2_mix',
              am_config=None,
              am_ckpt=None,
              am_stat=None,
              spk_id=0,
              phones_dict=None,
              tones_dict=None,
              speaker_dict=None,
              voc='hifigan_ljspeech',
              voc_config=None,
              voc_ckpt=None,
              voc_stat=None,
              lang='mix',
            )  
            history += [(None, (f"output{audio_number}.wav", None))]
            audio_number += 1
            yield history


prompts_dict = {
    "None": "Please forget what I said before.",
    "Teacher": "I want you to act as a spoken English teacher and improver. I will speak to you in English and you will reply to me in English to practice my spoken English. I want you to keep your reply neat, limiting the reply to 100 words. I want you to strictly correct my grammar mistakes, typos, and factual errors. I want you to ask me a question in your reply. Now let's start practicing, you could ask me a question first. Remember, I want you to strictly correct my grammar mistakes, typos, and factual errors.",
}

def change_prompt(history, label_name):
    if label_name == "Spoken English Teacher":
        history += [(prompts_dict["None"], None)]
    else:
        history += [(prompts_dict["Teacher"], None)]
    return history


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
        with gr.Column(scale=0.5, min_width=0):
            use_speech = gr.Checkbox(label="Speech(slow on cpu)", info="Whether to use PaddleSpeech?")
        with gr.Column(scale=0.5, min_width=0):
            prompts = gr.Radio(["Spoken English Teacher", "None"], value="None", label="Prompt")
    with gr.Row():
        gr.Markdown("- OpenAI ChatGPT3.5模型，github开源PaddleOCR、acheong08/ChatGPT、PaddleSpeech；\n- 注：上传的图片需带有文字描述的问题，类似小猿搜题；\n- 语音输入请用手机上自带的语音识别，另外为节省时间，只会输出100个字以内的语音。")


    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(bot, [chatbot, use_speech], chatbot)
    upload_btn.upload(add_file, [chatbot, upload_btn], [chatbot]).then(bot, [chatbot, use_speech], chatbot)
    send_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(bot, chatbot, chatbot)
    prompts.select(fn=change_prompt, inputs=[chatbot, prompts], outputs=[chatbot]).then(bot, [chatbot, use_speech], chatbot)

demo.queue(concurrency_count=5).launch()
