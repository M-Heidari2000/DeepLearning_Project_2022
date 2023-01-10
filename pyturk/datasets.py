from torch.utils.data import Dataset
from torchvision.io import read_image
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

    BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
    git_url = 'git@github.com:XL2248/MSCTD.git'
    images_urls = {
        'test' : 'https://drive.google.com/uc?id=1sMoQvrFP85bv9Dhv7mlLVBDj-1Efi9p-',
        'dev' : 'https://drive.google.com/uc?id=1VWZi-vmgagAfNsO052QhIOWQIWg174-Z',
        'train' : 'https://drive.google.com/uc?id=1fFyJ0O9ke3SAfkswUNVPm_BnG51F4lw4'
    }
    path_dict = {
        'train': 'train_ende',
        'test': 'test',
        'dev': 'dev'
    }

    def __init__(self, root='data', mode='train', download=True, image_transform=None, text_transform=None, target_transform=None):
        cls = self.__class__
        self.root = pathlib.Path(cls.BASE_DIR / root / cls.__name__)
        self.root.mkdir(exist_ok=True, parents=True)
        self.mode = mode
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.target_transform = target_transform

        if download:
            self.download_and_extract_data()
        self.data = self.read_conversation()

    def read_conversation(self):  
        text_dir_path = str(self.root / self.mode / 'texts')
        mode = self.mode

        index_path = str(pathlib.Path(text_dir_path) / f'image_index_{mode}.txt')
        texts_path = str(pathlib.Path(text_dir_path) / f'english_{mode}.txt')
        sentiment_path = str(pathlib.Path(text_dir_path) / f'sentiment_{mode}.txt')
        data = []
        logging.basicConfig(level=logging.INFO)

        logging.info('opening and reading files...')
        with open(index_path) as index_file, open(texts_path) as texts_file, open(sentiment_path) as sentiment_file:
            all_texts = tuple(texts_file)
            all_sentiment = tuple(sentiment_file)

            for indices in index_file:
                indices = eval(indices.strip())
                texts = [all_texts[idx].strip() for idx in indices]
                sentiment = [all_sentiment[idx].strip() for idx in indices]

                texts = tuple(texts)
                sentiment = tuple(sentiment)
                images = tuple(indices)
                data.append(({'images': images, 'texts': texts}, sentiment))
        logging.info('finished reading files...')
        logging.info('done...')
        return data

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
                zip_file.extractall(p)
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
        return len(self.data)

    def __getitem__(self, index):
        image_dir_path = self.root / self.mode / 'images' / self.path_dict[self.mode]
        image_text, labels = self.data[index]
        indices = image_text['images']
        texts = image_text['texts']
        images = []

        for index in indices:
            path = image_dir_path / f'{index}.jpg'
            images.append(read_image(str(path)))
        
        if self.image_transform is not None:
            images = tuple(self.image_transform(image) for image in images)
        if self.text_transform is not None:
            texts = tuple(self.text_transform(text) for text in texts)
        if self.target_transform is not None:
            labels = tuple(self.target_transform(label) for label in labels)
        
        return ({'images': images, 'texts': texts}, labels)
