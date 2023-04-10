import gradio as gr
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline
from huggingface_hub import _login
from huggingface_hub.hf_api import HfApi, HfFolder
import subprocess
import sys
import pathlib
import importlib.util
import numpy as np
import random
import datetime
from PIL import Image


python = sys.executable

onnx_dir = pathlib.Path().absolute()/'onnx_models'
output_dir = pathlib.Path().absolute()/'output'

if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


def txt2img(prompt, negative_prompt, steps, height, width, scale, denoise_strength=0, seed=None, scheduler=None, num_image=None):
    try:
        seed = int(seed)
        if seed < 0:
            seed = random.randint(0,4294967295)
    except:
        seed = random.randint(0, 4294967295)
        
    generator = np.random.RandomState(seed)
        
    image = pipe(prompt,
                negative_prompt = negative_prompt,
                num_inference_steps=steps,
                height = height,
                width = width,
                guidance_scale=scale,
                generator = generator,
                num_images_per_prompt = num_image
                
                ).images[0]

    img_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + ".png"
    image.save(output_dir/img_name)
    return image

def img2img(prompt, negative_prompt, image_input, steps, height, width, scale, denoise_strength, seed=None, scheduler=None, num_image=None):
    
    if seed == '':

        seed = random.randint(0,4294967295)
    elif seed != '':
        seed = int(seed)
        if seed < 0:
            seed = random.randint(0,4294967295)
    print(f'Ключ генерации: {seed}')
        
    generator = np.random.RandomState(seed)
    print(image_input)
    image = pipe_img2img(prompt=prompt,
                image = Image.fromarray(image_input),
                strength=denoise_strength,
                num_inference_steps=steps,
                guidance_scale=scale,
                negative_prompt = negative_prompt,
                num_images_per_prompt = num_image,
                generator = generator,
                ).images[0]
                
    img_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + ".png"
    image.save(output_dir/img_name)
    
    return image


def huggingface_login(token):
    try:
        output = _login._login(token = token, add_to_git_credential = True)
        return "Успешно."
    except Exception as e:
        return str(e)
    


def pip_install(lib):
    subprocess.run(f'echo Устанавливаю {lib}...', shell=True)
    if 'ort_nightly_directml' in lib:
        subprocess.run(f'echo 1', shell=True)
        subprocess.run(f'echo "{python}" -m pip install {lib}', shell=True)
        subprocess.run(f'"{python}" -m pip install {lib} --force-reinstall', shell=True)
    else:
        subprocess.run(f'echo 2', shell=True)
        subprocess.run(f'echo "{python}" -m pip install {lib}', shell=True, capture_output=True)
        subprocess.run(f'"{python}" -m pip install {lib}', shell=True, capture_output=True)

def pip_uninstall(lib):
    subprocess.run(f'echo Удаляю {lib}...', shell=True)
    subprocess.run(f'"{python}" -m pip uninstall -y {lib}', shell=True, capture_output=True)

def is_installed(lib):
    library =  importlib.util.find_spec(lib)
    return (library is not None)

def download_sd_model(model_path):
    pip_install('onnx')
    from src.diffusers.scripts import convert_stable_diffusion_checkpoint_to_onnx
    onnx_opset = 14
    onnx_fp16 = False
    try:
        model_name = model_path.split('/')[1]
    except:
        model_name = model_path
    onnx_model_dir = onnx_dir/model_name 
    if not onnx_dir.exists():
        onnx_dir.mkdir(parents=True, exist_ok=True)
        print(model_name)
    convert_stable_diffusion_checkpoint_to_onnx.convert_models(model_path, str(onnx_model_dir), onnx_opset, onnx_fp16)
    pip_uninstall('onnx')


def display_onnx_models():
    if not onnx_dir.exists():
        onnx_dir.mkdir(parents=True, exist_ok=True)
    return [m.name for m in onnx_dir.iterdir() if m.is_dir()]


def load_onnx_model(model):
    global pipe
    pipe = OnnxStableDiffusionPipeline.from_pretrained(str(onnx_dir/model),
                                                       safety_checker = None,
                                                       provider="DmlExecutionProvider")
    
    return 'Модель готова'


def load_onnx_model_i2i(model):
    global pipe_img2img
    pipe_img2img = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(str(onnx_dir/model),
                                                              safety_checker = None,
                                                              provider="DmlExecutionProvider")
    return 'Модель готова'


def start_app():
    with gr.Blocks() as app:
        gr.Markdown('StableDiffusion WebUI для видеокарт AMD на операционной системе Windows')
        with gr.Tab('txt2img'):
            txt2img_prompt_input = gr.Textbox(label='Запрос')
            txt2img_negative_prompt_input = gr.Textbox(label='Отрицательный запрос')
            with gr.Row():
                with gr.Column(scale = 1):

                    with gr.Row():
                        txt2img_model_input = gr.Dropdown(label='Выберите модель:', choices = display_onnx_models())
                        test_output = gr.Textbox(label='Статус модели')
                    inference_step_input = gr.Slider(label='Шаги', value = 30, minimum = 5, maximum=100, step = 1)
                    with gr.Row():
                        image_height = gr.Slider(label='Высота', value = 512, minimum = 0, maximum=2048, step = 32)
                        image_width = gr.Slider(label='Ширина', value = 512, minimum = 0, maximum=2048, step = 32)
                    with gr.Row():
                        scale = gr.Slider(label='Скейл', value = 7.5, minimum = 0, maximum=15, step = 0.1)
                        denoise_strength = gr.Slider(label='Сила Denoise', value = 1, minimum = 0, maximum=1, step = 0.1)
                    with gr.Row():
                        seed_input = gr.Textbox(label='Ключ генерации')
                        scheduler_input = gr.Dropdown(['Опция 1', 'Опция 2'])
                        
                    num_image = gr.Slider(label='Количество изображений', value = 1, minimum = 1, maximum=10, step = 1)
                        
                    with gr.Row():
                        txt2img_button = gr.Button('Сгенерировать')
                    
                txt2img_output = gr.Image(label='Итоговое изображение')
        with gr.Tab('img2img'):
            img2img_prompt_input = gr.Textbox(label='')
            img2img_negative_prompt_input = gr.Textbox(label='Отрицательный запрос')
            with gr.Row():
                img2img_image_input = gr.Image()
                
                img2img_image_output = gr.Image(label='Вывод img2img')
            with gr.Row():
                    img2img_model_input = gr.Dropdown(label='Выберите модель:', choices = display_onnx_models())
                    img2img_test_output = gr.Textbox(label='Статус модели')
                    img2img_inference_step_input = gr.Slider(label='Steps', value = 30, minimum = 5, maximum=100, step = 1)
            with gr.Row():
                img2img_image_height = gr.Slider(label='Высота', value = 512, minimum = 0, maximum=2048, step = 64)
                img2img_image_width = gr.Slider(label='Ширина', value = 512, minimum = 0, maximum=2048, step = 64)
            with gr.Row():
                img2img_scale = gr.Slider(label='Скейл', value = 7.5, minimum = 0, maximum=15, step = 0.1)
                img2img_denoise_strength = gr.Slider(label='Сила Denoise', value = 1, minimum = 0, maximum=1, step = 0.1)
            with gr.Row():
                img2img_seed_input = gr.Textbox(label='Ключ генерации')
                img2img_num_image = gr.Slider(label='Количество изображений', value = 1, minimum = 1, maximum=10, step = 1)
                
                
            img2img_button = gr.Button('Сгенерировать')
        with gr.Tab('Менеджер моделей'):
            gr.Markdown("Некоторые модели требуют, чтобы вы вошли в Huggingface и согласились с их условиями. Обязательно сделайте это перед загрузкой моделей!")
            model_download_input = gr.Textbox()
            model_download_button = gr.Button('Установить модель')
            model_dir_refresh = gr.Button('Обновить')
            
        with gr.Tab('Настройки'):
            gr.HTML("Нажмите на эту ссылку чтобы получить свой API ключ: <a href='https://huggingface.co/settings/tokens' style='color:blue'>HuggingFace Access Token</a>")
            hugginface_token_input = gr.Textbox()
            huggingface_login_message = gr.Textbox()
            huggingface_login_button = gr.Button('Войти в HuggingFace')
            
        txt2img_button.click(txt2img, inputs=[txt2img_prompt_input, txt2img_negative_prompt_input, inference_step_input,
                                                                                  image_height, image_width, scale, denoise_strength,
                                                                                  seed_input, scheduler_input,
                                                                                  num_image], outputs = txt2img_output)
        
        img2img_button.click(img2img, inputs=[img2img_prompt_input, img2img_negative_prompt_input, img2img_image_input, img2img_inference_step_input,
                                                                                  img2img_image_height, img2img_image_width, img2img_scale, img2img_denoise_strength,
                                                                                  img2img_seed_input, img2img_num_image
                                                                                  ], outputs = img2img_image_output, show_progress=True)


        huggingface_login_button.click(huggingface_login,
                                       inputs = hugginface_token_input,
                                       outputs = huggingface_login_message)
        
        model_download_button.click(download_sd_model, inputs = model_download_input)
        txt2img_model_input.change(load_onnx_model, inputs=txt2img_model_input, show_progress=True, outputs = test_output)
        img2img_model_input.change(load_onnx_model_i2i, inputs=img2img_model_input, show_progress=True, outputs = img2img_test_output)

    app.launch(inbrowser = True)


if __name__ == "__main__":
    start_app()
