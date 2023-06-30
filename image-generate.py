from typing import TextIO
import keras_cv
from translate import Translator
import matplotlib.pyplot as plt


model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
translator = Translator(from_lang="ko", to_lang="en")


def generate_img(text:str, cnt:int) -> list:
    #입력 받은 텍스트를 cnt개의 이미지로 생성하여 return

    text = translator.translate(text)
    images = model.text_to_image(text, batch_size=cnt)
    return images

def plot_image(images) -> None:
    #입력 받은 이미지들을 이미지로 저장
    plt.figure(figsize=(20, 20))

    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.tight_layout()


    plt.savefig(f"text.png")

import gradio as gr


def inference(text:str):
    image = generate_img(text, 1).squeeze()
    return image

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:

  with gr.Row():
    with gr.Column():

      text=gr.Textbox(lines=5,placeholder='Prompt here',label="prompt")
      create_btn = gr.Button(value="Create")
    with gr.Column():
      output=gr.Image(shape=(200,200))
  create_btn.click(fn=inference,inputs=text,outputs=output)

#demo = gr.Interface(fn=inference, inputs=gr.Textbox(lines=2,placeholder='prompt here'), outputs="image", title="Image Generater")
demo.launch(share=True)