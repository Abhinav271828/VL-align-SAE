from torch.utils.data import Dataset
from datasets import load_dataset
from utils import convert_raw_to_embeddings
import os

#/scratch/bani/.cache/datasets/downloads/extracted/70111e6614c7cec8efd278347b6207e7de5dd3b8babd21af307ade43cbc27d02/val2014/filename
#ds['train']['filename'][i]
#ds['train']['sentences'][i]['raw']

class MSCOCO(Dataset):
    def __init__(self, cache_dir  : str = '/scratch/bani/.cache/datasets', split : str = 'train'):
        self.cache_dir = cache_dir
        self.split = split
        self.image_cache = f'coco_images_{split}.h5py'
        self.text_cache = f'coco_text_{split}.h5py'
        if not (os.path.exists(self.image_cache) and os.path.exists(self.text_cache)):
            self.gen_embeddings()
        
        self.retrieve_embeddings()

    def gen_embeddings(self):
        ds = load_dataset("HuggingFaceM4/COCO", cache_dir='/scratch/bani/.cache/datasets', trust_remote_code=True)
        convert_raw_to_embeddings(self.split,
                                  'meta-llama/Llama-3.2-1B', 'facebook/dinov2-base',
                                  (self.image_cache, self.text_cache))

    def retrieve_embeddings(self):
        text_hf = h5py.File(text_embed_path, "r")
        image_hf = h5py.File(image_embed_path, "r")

        indices_text = list(text_hf.keys())
        indices_text = [int(i) for i in indices_text]
        # sort the indices
        indices_text.sort()

        text_embeddings = []
        for i in indices_text:
            text_embeddings.append(
                text_hf.get(str(i))[:]
            )  # sequence length, dimensions

        indices_image = list(image_hf.keys())
        indices_image = [int(i) for i in indices_image]
        # sort the indices
        indices_image.sort()

        image_embeddings = []
        for i in indices_image:
            image_embeddings.append(image_hf.get(str(i))[:])  # dimensions

        text_hf.close()
        image_hf.close()

        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
    
    def __len__(self):
        return len(self.text_embeddings)
    
    def __getitem__(self, idx):
        return self.text_embeddings[idx], self.image_embeddings[idx]
