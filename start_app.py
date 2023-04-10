import importlib.util
import platform
import subprocess
import sys
import pathlib
import time


python = sys.executable
if sys.version[:4] == "3.10":
    onnx_nightly = 'ort_nightly_directml-1.13.0.dev20220908001-cp310-cp310-win_amd64.whl'
elif sys.version[:3] == "3.9":
    onnx_nightly = 'ort_nightly_directml-1.13.0.dev20220908001-cp39-cp39-win_amd64.whl'
elif sys.version[:3] == "3.8":
    onnx_nightly = 'ort_nightly_directml-1.13.0.dev20220908001-cp38-cp38-win_amd64.whl'
elif sys.version[:3] == "3.7":
    onnx_nightly = 'ort_nightly_directml-1.13.0.dev20220908001-cp37-cp37-win_amd64.whl'
else:
    print('Найдена версия которая не поддерживает DirectML, пожалуйста установите 3.7, 3.8, 3.9, или 3.10!')
print(f'Вы используете Python {sys.version}')



required_lib = ['torch', 'onnxruntime', 'transformers', 'scipy', 'ftfy', 'gradio']
standard_onnx = 'onnx'
repositories = pathlib.Path().absolute() / 'repositories'
git_repos = ['https://github.com/huggingface/diffusers']
requirements = pathlib.Path().absolute()  /'requirements.txt'


def pip_install(lib):
    subprocess.run(f'echo Устанавливаю {lib}...', shell=True)
    subprocess.run(f'echo "{python}" -m pip install {lib}', shell=True)
    subprocess.run(f'"{python}" -m pip install {lib}', shell=True, capture_output=True)
    subprocess.run(f'"{python}" -m pip install {lib}', shell=True, capture_output=True)

def pip_install_requirements():
    subprocess.run(f'echo Устанавливаю requirements.txt', shell=True)
    subprocess.run(f'"{python}" -m pip install -r requirements.txt', shell=True, capture_output=True)

def is_installed(lib):
    library =  importlib.util.find_spec(lib)
    return (library is not None)

def git_clone(repo_url, repo_name):
    repo_dir = repositories/repo_name
    if not repo_dir.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(f'echo cloning {repo_dir}', shell=True)
        subprocess.run(f'git clone {repo_url} "{repo_dir}"', shell=True)
    else:
        subprocess.run(f'echo {repo_name} уже существует!', shell=True)

pip_install_requirements()
subprocess.run(f'"{python}" -m pip install repositories/{onnx_nightly}', shell=True)
subprocess.run('echo Успешная установка', shell=True)



import amd_webui
amd_webui.start_app()
