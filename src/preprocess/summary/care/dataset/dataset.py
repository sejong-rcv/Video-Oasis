import torch
import json
from PIL import Image
from utils.video import read_frames_decord
from torchvision.transforms import PILToTensor
import os

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [b[key] for b in batch]
    return collated_batch

class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            anno_path: str, 
            data_root=None, 
            decode=True,
            apply_paragraph_retrieval=False,
            trim30=False,
            num_frames=32,
            **kwargs,
        ):
        with open(anno_path) as f:
            self.data = json.load(f)
        self.data_root = data_root
        self.apply_paragraph_retrieval = apply_paragraph_retrieval
        self.trim30 = trim30
        self.num_frames = num_frames
        self.decode = decode
        self.texts = []
        self.texts_vision_index = []
        self.vision_texts_index = []

        self._proprocess()
        self.return_text = False

    def _proprocess(self):
        if self.apply_paragraph_retrieval:
            self._cast('caption', lambda x: ' '.join(x))
        self._cast('video', lambda x: os.path.join(self.data_root, x))
        self._indexing()
        
    def _indexing(self):
        # add idx to each data
        for idx in range(len(self.data)):
            self.data[idx]['idx'] = idx
        # generate texts and vision_texts_index
        for idx, ann in enumerate(self.data):
            self.vision_texts_index.append([])
            if isinstance(ann["caption"], list):
                _captions = ann["caption"]
            else:
                _captions = [ann["caption"]]
            for i, caption in enumerate(_captions):
                self.texts.append(caption)
                self.texts_vision_index.append(idx)
                self.vision_texts_index[idx].append(len(self.texts_vision_index) - 1)
    
    def _cast(self, key, func: callable):
        for idx in range(len(self.data)):
            self.data[idx][key] = func(self.data[idx][key])

    def __len__(self):
        if self.return_text:
            return len(self.texts)
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_text:
            return {'caption': self.texts[idx]}
        d = self.data[idx]
        if self.decode:
            d['video'] = read_frames_decord(d['video'], num_frames=self.num_frames, trimmed30=self.trim30)
        return {'idx': d['idx'], 'video': d['video'], 'caption': d['caption']}
