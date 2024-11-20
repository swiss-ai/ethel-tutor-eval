import io
import os
import shutil
import zipfile
import tarfile
import requests
import yaml

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_gsm8k(dataset_dir: str):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    logger.info("Downloading GSM8k dataset")
    dataset_url = 'https://github.com/openai/grade-school-math/archive/refs/heads/master.zip'

    response = requests.get(dataset_url)
    response.raise_for_status()

    # Extracting the zip file content
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        for member in zip_ref.namelist():
            # We need to extract only files from the data directory
            if member.startswith(os.path.join('grade-school-math-master', 'grade_school_math', 'data') + os.sep):
                if '.' not in member.split(os.sep)[-1]:
                    continue  # ignore /data/ directory
                # This will extract the file into the desired directory structure
                zip_ref.extract(member, dataset_dir)
                # Rename the extracted files to remove the prefix directories
                extracted_path = os.path.join(dataset_dir, member)
                target_path = os.path.join(dataset_dir, os.path.basename(member))
                shutil.move(extracted_path, target_path)

    extracted_main_dir = os.path.join(dataset_dir, 'grade-school-math-master')
    if os.path.exists(extracted_main_dir):
        shutil.rmtree(extracted_main_dir)

    logger.info("Downloaded GSM8k dataset!")


def download_math(dataset_dir: str):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    logger.info("Downloading MATH dataset")
    dataset_url = 'https://people.eecs.berkeley.edu/~hendrycks/MATH.tar'

    response = requests.get(dataset_url)
    response.raise_for_status()

    # Extracting the tar file content
    with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:') as tar_ref:
        tar_ref.extractall(path=dataset_dir)

    logger.info("Downloaded and extracted MATH dataset!")


if __name__ == '__main__':
    project_root_path = os.path.join(os.path.dirname(__file__), '..')

    config_path = os.path.join(project_root_path, 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # download_gsm8k(str(os.path.join(project_root_path, config['datasets']['gsm8k'])))
    download_math(str(os.path.join(project_root_path, config['datasets']['math'])))