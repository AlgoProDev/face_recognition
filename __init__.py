import os
import gdown
import subprocess


def download_folder_from_google_drive(folder_path):
    file_id = "1JGESpeciguUaKD902syAb15-qlFGLI5e"
    url = f"https://drive.google.com/uc?id={file_id}"
    zip_file_path = os.path.join(folder_path + ".rar")

    gdown.download(url, zip_file_path, quiet=False)
    os.mkdir(folder_path)
    if os.path.exists(zip_file_path):
        try:
            rar_executable_path = "C:\\Program Files\\WinRAR\\WinRAR.exe"
            subprocess.run([rar_executable_path, "x", zip_file_path, folder_path])
        except Exception as e:
            print(f"Error extracting {zip_file_path}: {e}")
        else:
            os.remove(zip_file_path)
    else:
        print(f"Error: {zip_file_path} does not exist!")


script_dir = os.path.dirname(os.path.abspath(__file__))
base_models = os.path.join(script_dir, "basemodels")
folder_path = os.path.join(base_models, "models")

if not os.path.exists(folder_path):
    print(f"models does not exist. Downloading from Google Drive...")
    download_folder_from_google_drive(folder_path)
