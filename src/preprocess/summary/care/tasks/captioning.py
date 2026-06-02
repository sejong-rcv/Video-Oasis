import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging
import functools
from tqdm import tqdm
from models.modeling_captioners import AutoCaptioner
from typing import List
import json
import os
from torch.utils.data import DataLoader
from accelerate import Accelerator
from dataset.dataset import VideoTextDataset


accelerator = Accelerator(device_placement=False)

def wrap_main_process(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        if accelerator.is_main_process:
            return function(*args, **kwargs)
        return lambda *args, **kwargs: None
    return run

def wraped_getLogger(name: str | None = None) -> logging.Logger:
    logger = logging._getLogger(name)
    logger.log       = wrap_main_process(logger.log)
    logger.info      = wrap_main_process(logger.info)
    logger.error     = wrap_main_process(logger.error)
    logger.warning   = wrap_main_process(logger.warning)
    logger.debug     = wrap_main_process(logger.debug)
    return logger


logging._getLogger = logging.getLogger
logging.getLogger = wraped_getLogger

logger = logging.getLogger(__name__)


def get_dataloader(config_path: str, dataset_name: str, num_frames: int) -> DataLoader:
    # load data.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    anno_path = config[dataset_name]['anno_path']
    data_root = config[dataset_name]['data_root']
    media_type = config[dataset_name]['media_type']
    assert media_type == 'video', 'media_type must be video'
    dataset = VideoTextDataset(
        anno_path=anno_path,
        data_root=data_root,
        num_frames=num_frames
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    return dataloader

def convert_list_to_dict(data: List[dict], index_key: str='video') -> dict:
    """
    Converts a list of dictionaries into a dictionary of dictionaries, using a specified key as the index.

    Args:
        data (List[dict]): A list of dictionaries to be converted.
        index_key (str): The key to be used as the index in the resulting dictionary. Defaults to 'video'.

    Returns:
        dict: A dictionary where each key is the value of the specified index_key from the input dictionaries,
              and each value is a dictionary containing the remaining key-value pairs from the input dictionaries.

    Raises:
        ValueError: If the specified index_key is not present in the keys of the input dictionaries.
    """
    keys = data[0].keys()
    if index_key not in keys:
        raise ValueError(f'Index key `{index_key}` not in keys')
    return {d[index_key]: {k: d[k] for k in keys if k != index_key} for d in data}

def gen_description(
    config_path: str,
    dataset_name: str,
    model_path: str,
    save_path: str = None,
    num_frames: int = 64,
) -> str:
    
    if os.path.exists(save_path):
        logger.info(f'{save_path} already exists. Skipping...')
        with open(save_path, 'r') as f:
            data = json.load(f)
        return data

    if model_path is None:
        raise ValueError('model_path must be provided if description.json does not exist')
    
    logger.info('Generating descriptions...')
    captioner = AutoCaptioner.from_pretrained(model_path, is_llm=False)
    dataloader = get_dataloader(config_path, dataset_name, num_frames)
    dataloader = accelerator.prepare(dataloader)
    data = []
    for batch in tqdm(dataloader):
        d = []
        preds = captioner.describe(batch['video'])
        for idx, gt, pred in zip(batch['idx'], batch['caption'], preds):
            d.append({'idx': idx.item(), 'pred': pred, 'gt': gt})
        d = accelerator.gather_for_metrics(d)
        data += d

    # NOTE: `data` only contains 'idx', 'pred' and 'gt'
    # since __getitem__ in VideoTextDataset doesn't return events
    #
    # We need to get the original data from VideoTextDataset
    # and try to merge events to `data`
    #
    # We can't get events from __getitem__ directly since events may be None
    # and can't be collected in batch.
    
    data = convert_list_to_dict(data, index_key='idx')
    raw_data = convert_list_to_dict(dataloader.dataset.data, index_key='idx')

    events_none = 0
    objects_none = 0
    for k in data.keys():
        data[k]['events'] = raw_data[k].get('events', None)
        data[k]['objects'] = raw_data[k].get('objects', None)
        if data[k]['events'] is None:
            events_none += 1
        if data[k]['objects'] is None:
            objects_none += 1
    
    if events_none > 0:
        logger.info(f'No events found for {events_none} entries. Events will be extracted while evaluating.')
    else:
        logger.info('Events found for all entries. Events will not be extracted again.')
    
    if objects_none > 0:
        logger.info(f'No objects found for {objects_none} entries. Objects will be extracted while evaluating.')
    else:
        logger.info('Objects found for all entries. Objects will not be extracted again.')

    if save_path is not None:
        logger.info(f'Saving results to {save_path}')
        with open(save_path, 'w') as f:
            json.dump(data, f)
    return data
    
def evaluate_gpt(data, result_dir, api_endpoint, api_key, api_model, api_num_worker):
    logger.info('Evaluating GPT...')

    os.environ['AZURE_ENDPOINT'] = api_endpoint
    os.environ['OPENAI_API_KEY'] = api_key

    from utils.dream_gpt import DREAMGPTMetric

    metric = DREAMGPTMetric("TEST")
    metric.num_worker = api_num_worker
    metric.model = api_model

    dataset = []
    events_none = 0
    for idx, anno in data.items():
        data = {}
        data['idx'] = idx
        data['dataset'] = "overall"
        data['response'] = anno['gt']
        data['prediction'] = anno['pred']
        data['events'] = anno['events']
        data['objects'] = anno['objects']
        
        dataset.append(data)
    
    if events_none > 0:
        logger.warning(f'No events found for {events_none} entries. Events will be extracted while evaluating.')
        
    metric.process(dataset[:])
    metric._summarize_metric_by_subtask()

    os.makedirs(result_dir, exist_ok=True)
    metric.save_results(result_dir)
    metric.save_eval_infos(result_dir)

def set_logger(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    # add file handler to root logger
    logging.getLogger().addHandler(file_handler)


def main(
    config_path: str,
    dataset_name: str,
    model_path: str,
    save_dir: str,
    num_frames: int,
    evaluate: bool = True, # if not evaluate, just generate descriptions
    api_endpoint: str = None,
    api_key: str = None,
    api_model: str = None,
    api_num_worker: int = 10,
):
    
    os.makedirs(save_dir, exist_ok=True)
    DESCRIPTION_JSON_PATH = os.path.join(save_dir, 'description.json')
    LOGGING_PATH = os.path.join(save_dir, 'run.log')
    set_logger(LOGGING_PATH)

    logger.info('********** Start Video Captioning Task **********')
    logger.info(f'config_path: {config_path}')
    logger.info(f'dataset_name: {dataset_name}')
    logger.info(f'model_path: {model_path}')
    logger.info(f'save_dir: {save_dir}')
    logger.info(f'num_frames: {num_frames}')
    logger.info(f'api_model: {api_model}')
    logger.info(f'api_num_worker: {api_num_worker}')
    logger.info(f'api_endpoint: {api_endpoint}')
    logger.info(f'api_key: {api_key[:7] + "*" * (len(api_key) - 8) + api_key[-4:]}')
    
    if evaluate and (api_endpoint is None or api_key is None):
        logger.error('api_endpoint and api_key must be provided')
        return
    
    data = gen_description(config_path, dataset_name, model_path, DESCRIPTION_JSON_PATH, num_frames)
    if evaluate and accelerator.is_main_process:
        evaluate_gpt(data, save_dir, api_endpoint, api_key, api_model, api_num_worker)

if __name__ == '__main__':
    from fire import Fire
    Fire(main)

