from torch.utils.data import Dataset
from datasets import load_dataset

#/scratch/bani/.cache/datasets/downloads/extracted/70111e6614c7cec8efd278347b6207e7de5dd3b8babd21af307ade43cbc27d02/val2014/filename
#ds['train']['filename'][i]
#ds['train']['sentences'][i]['raw']

class MSCOCO(Dataset):
    def __init__(self, cache_dir  : str = '/scratch/bani/.cache/datasets', split : str = 'train'):
        self.cache_dir = cache_dir
        self.image_cache = f'coco_images_{split}.h5py'
        self.text_cache = f'coco_text_{split}.h5py'
    
    def gen_image_embeddings(self):
        ds = load_dataset("HuggingFaceM4/COCO", cache_dir='/scratch/bani/.cache/datasets', trust_remote_code=True)
        pass

    def gen_text_embeddings(self):
        ds = load_dataset("HuggingFaceM4/COCO", cache_dir='/scratch/bani/.cache/datasets', trust_remote_code=True)
        pass
