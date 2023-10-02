import os
import zipfile
import gdown


def download_folder_from_google_drive(subfolder_path, folder_path):
    # This would be the link to the zip file on Google Drive.
    file_id = "1JGESpeciguUaKD902syAb15-qlFGLI5e"
    url = f"https://drive.google.com/uc?id={file_id}"
    zip_file_path = folder_path + ".zip"
    gdown.download(url, zip_file_path, quiet=False)

    # Extract the zip file content to the folder_path
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(subfolder_path)

    # Optionally, remove the zip file after extracting its content
    os.remove(zip_file_path)


# Specify the path where you expect the folder to be
folder_path = os.path.join("basemodels", "models")

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"models does not exist. Downloading from Google Drive...")

    # Here, you need to implement or call the function/method to download
    # the folder from Google Drive and save it to the specified path.
    download_folder_from_google_drive("basemodels", folder_path)
