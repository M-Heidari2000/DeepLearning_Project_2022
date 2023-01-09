from torch.utils.data import Dataset
from torchvision import datasets
import gdown
import pathlib
import os
import zipfile
import git
import logging
import shutil
import glob

logging.basicConfig(level=logging.INFO)

class MSCTD(Dataset):

    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    git_url = 'git@github.com:XL2248/MSCTD.git'
    images_urls = {
        # 'train': 'https://drive.google.com/uc?id=1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj',
        # 'test' : 'https://drive.google.com/uc?id=1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W',
        # 'dev' : 'https://drive.google.com/uc?id=12HM8uVNjFg-HRZ15ADue4oLGFAYQwvTA',
        'train' : 'https://drive.google.com/uc?id=1fFyJ0O9ke3SAfkswUNVPm_BnG51F4lw4'
    }

    def __init__(self, root='data', mode='train', download=True, transform=None, target_transform=None):
        cls = self.__class__
        self.root = pathlib.Path(cls.BASE_DIR / root / cls.__name__)
        self.root.mkdir(exist_ok=True, parents=True)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download_and_extract_data()

    def download_and_extract_data(self):
        cls = self.__class__
        subfolder = 'images'

        # clean the root directory if already exists
        if os.path.exists(self.root / self.mode):
            shutil.rmtree(self.root / self.mode)

        # download and extract images from Google drive
        try:
            output = self.root / f'myzip.zip'
            gdown.download(url=cls.images_urls.get(self.mode), output=str(output))
            with zipfile.ZipFile(output, 'r') as zip_file:
                zip_file.extractall(self.root / self.mode / subfolder)
        finally:
            os.remove(output)

        # clone the git repository containing sentiments, sentences and image indexes
        try:
            subfolder = 'texts'
            logging.info('cloning the git repository ...')
            repo_dir =  self.root / self.mode / 'repo'
            if os.path.exists(repo_dir):
                shutil.rmtree(repo_dir)
            git.Repo.clone_from(cls.git_url, repo_dir)
            logging.info('cloning finished!')
            
            # copy related text files
            file_pattern = str(repo_dir / 'MSCTD_data' / 'ende' / f'*_{self.mode}.txt')
            file_names = glob.glob(file_pattern)
            folder_path = self.root / self.mode / subfolder
            os.makedirs(folder_path, exist_ok=True)
            for file_name in file_names:
                shutil.copy2(file_name, folder_path / file_name.split('/')[-1])
        
        finally:
            shutil.rmtree(repo_dir)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError


if __name__ == '__main__':
    dataset = MSCTD(
        root='data',
        mode='train',
        download=True
    )