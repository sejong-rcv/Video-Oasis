import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import torch
from tqdm import tqdm
from accelerate import Accelerator
from PIL import Image
import torch.nn.functional as F
from models.modeling_encoders import AutoEncoder
from utils.video import read_frames_decord
import decord
from torchvision.transforms.v2 import PILToTensor
from dataset.dataset import VideoTextDataset


decord.bridge.set_bridge("torch")
accelerator = Accelerator(device_placement=False)

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

def emb_data(encoder, dataset, device,
             emb_type='text'):
    # convert batch from to a dictionary
    def custom_collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = [b[key] for b in batch]
        return collated_batch
    if emb_type == 'text':
        dataset.return_text = True
    else:
        dataset.return_text = False

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=1,
        collate_fn=custom_collate_fn
    )
    dataloader = accelerator.prepare(dataloader)
    embs = []
    bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        emb = []
        if emb_type == 'text':
            emb = encoder.encode_text(batch['caption'])
        elif emb_type == 'video':
            emb = encoder.encode_vision(batch['video'])
        elif emb_type == 'image':
            emb = encoder.encode_vision(batch['image'])

        emb = F.normalize(emb, dim=-1)
        emb = accelerator.gather_for_metrics(emb)
        embs.append(emb.cpu().float())
        bar.update(1)

    embs = torch.cat(embs)
    bar.close()
    return embs

def ir(encoder, data_config, device, num_frames=32):

    dataset = VideoTextDataset(**data_config, num_frames=num_frames)
    text_embs = emb_data(encoder, dataset, device, emb_type='text')
    texts_image_index = dataset.texts_vision_index
    vision_texts_index = dataset.vision_texts_index
    assert len(texts_image_index) == len(text_embs), f'length of text embs({len(text_embs)}) and texts_image_index({len(texts_image_index)}) should be the same'
    video_embs = emb_data(encoder, dataset, device, emb_type='video')

    assert text_embs.isnan().sum().item() == 0, 'nan in retrieve emb'
    assert video_embs.isnan().sum().item() == 0, 'nan in images emb'

    # get the score for each text and image pair
    scores  = text_embs @ video_embs.t()

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    recall_k_list = [1, 5, 10]
    batch_size = 64
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item() * 100
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item() * 100
    
    return metrics

def main(
        # model: str = None,
        model_path: str = None,
        data: str = "msrvtt",
        num_frames: int = 32,
):

    device=accelerator.device

    # encoder = init_encoder(model, model_path, tokenizer_path, bf16)
    encoder = AutoEncoder.from_pretrained(model_path)

    from datasets import disable_caching
    disable_caching()

    assert os.path.exists("data.config"), "data.config not found"
    with open("data.config") as f:
        data_configs = json.load(f)

    metrics = ir(
        encoder=encoder, 
        device=device, 
        data_config=data_configs[data],
        num_frames=num_frames,
    )

    if accelerator.is_main_process:
        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}")


if __name__ == '__main__':
    from fire import Fire
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    Fire(main)
