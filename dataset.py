import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import json
from utils import *


class MMQADataset(Dataset):
    def __init__(self, split: str,
                 file_path: str = '../dataset/multimodalqa/dataset/'):
        self.file_path = file_path
        self.file_name = file_path + f'MMQA_{split}.jsonl.gz'
        self.total_data = self.load_data()

    def load_data(self):
        imageq = 0
        textq = 0
        total_data_list = []
        raw_data = read_jsonl_gz(self.file_name)
        image_data = read_jsonl_gz(self.file_path + 'MMQA_images.jsonl.gz')
        text_data = read_jsonl_gz(self.file_path + 'MMQA_texts.jsonl.gz')
        for data in raw_data:
            data_dict = {}
            if data['metadata']['type'].lower() not in ['imageq', 'textq']:
                continue
            if data['metadata']['type'].lower() == 'imageq':
                imageq += 1
            if data['metadata']['type'].lower() == 'textq':
                textq += 1
            support_data_list = []
            for support_data in data['supporting_context']:
                if support_data['doc_part'] == 'image':
                    for image in image_data:
                        if image['id'] == support_data['doc_id']:
                            image.update({'modality': support_data['doc_part']})
                            support_data_list.append(image)
                if support_data['doc_part'] == 'text':
                    for text in text_data:
                        if text['id'] == support_data['doc_id']:
                            text.update({'modality': support_data['doc_part']})
                            support_data_list.append(text)
            # for answer in data['answers']:
            #     if answer['modality'] not in self.excluded_modality:
            #         cleaned_answer_list.append(answer)
            data_dict['answers'] = data['answers']
            data_dict['question'] = data['question']
            data_dict['context'] = support_data_list
            data_dict['modality'] = data['metadata']['type']
            data_dict['Qcate'] = ''
            total_data_list.append(data_dict)
        print(f'ImageQ: {imageq}. TextQ: {textq}')
        return total_data_list

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        item = {"query": self.total_data[idx]['question'],
                'context': self.total_data[idx]['context'],
                'answers': self.total_data[idx]['answers'],
                'modality': self.total_data[idx]['modality'],
                'Qcate': self.total_data[idx]['Qcate']}
        return item


class WebQADataset(Dataset):
    def __init__(self, split: str = 'val',
                 file_path: str = '../dataset/WebQA/webqa_data/'):
        self.file_path = file_path
        self.file_name = file_path + 'webqa_val.json'
        self.split = split
        self.total_data = self.load_data()

    def load_data(self):
        total_data_list = []
        with open(self.file_name, 'r') as f:
            raw_data = json.load(f)
        for idx, data in raw_data.items():
            data_dict = {}
            # if data['split'] != self.split:
            #     continue
            data_dict['answers'] = []
            for answer in data['A']:
                # answer = data['A'][0]
                if data['Qcate'].lower() == 'yesno':
                    if 'yes' in answer.lower():
                        answer = 'yes'
                    else:
                        answer = 'no'
                else:
                    answer = remove_quotes(answer)
                data_dict['answers'].append({'answer': answer})
            data_dict['question'] = remove_quotes(data['Q'])
            data_dict['context'] = []
            data_dict['Qcate'] = data['Qcate'].lower()
            data_dict['modality'] = ''
            total_data_list.append(data_dict)
        return total_data_list

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        item = {"query": self.total_data[idx]['question'],
                'context': self.total_data[idx]['context'],
                'answers': self.total_data[idx]['answers'],
                'modality': self.total_data[idx]['modality'],
                'Qcate': self.total_data[idx]['Qcate']}
        return item


class MRAMGDataset(Dataset):
    def __init__(self, split: str = 'arxiv',
                 file_path: str = '../dataset/MRAMG-Bench'):
        self.file_path = file_path
        # self.image_path = f'{file_path}/IMAGE/images/{split.upper()}/'
        self.qa_file = read_jsonl(f'{file_path}/{split}_mqa.jsonl')
        self.text_documents = read_jsonl(f'{file_path}/doc_{split}.jsonl')
        self.total_data = self.load_data()

    def load_data(self):
        total_data = []
        convert_dict = {}
        for idx, doc, img_idx in self.text_documents:
            convert_dict[idx] = doc
        for qa_data in self.qa_file:
            q_id = qa_data['id']
            question = qa_data['question'].replace('<PIC>', '').strip()
            answer_str = qa_data['ground_truth'].replace('<PIC>', '').strip()
            answer = [{'answer': answer_str}]
            context_id_list = qa_data['provenance']
            context_data = [convert_dict[i] for i in context_id_list]
            data_dict = {
                'id': q_id,
                'query': question,
                'answers': answer,
                'context': context_data,
                'Qcate': '',
            }
            total_data.append(data_dict)
        return total_data

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        item = {"query": self.total_data[idx]['query'],
                'context': self.total_data[idx]['context'],
                'answers': self.total_data[idx]['answers'],
                'id': self.total_data[idx]['id'],
                'Qcate': self.total_data[idx]['Qcate']}
        return item


def load_support_img_caption_mramg(model, split='arxiv'):
    file_path = f'../dataset/MRAMG-Bench/IMAGE/images_info/{split}_imgs_collection.json'
    with open(file_path, 'r') as f:
        image_data = json.load(f)
    parent_dir = f'../dataset/MRAMG-Bench/IMAGE/images/{split.upper()}/'
    image_caption_dict = {}
    for idx, image in image_data.items():
        image_path = parent_dir + image['image_path']
        if not os.path.exists(image_path):
            print(f'Image {image_path} does not exist.')
            print(f'Download from {image["image_url"]}')
            try:
                download_image_and_save(image['image_url'], image_path)
            except:
                print(f'Failed to open image, skip.')
                continue
        else:
            # load image to check if it's corrupted
            try:
                img = Image.open(image_path)
                img = img.convert('RGB')
                img.close()
            except:
                print(f'Failed to open image, re-download.')
                try:
                    download_image_and_save(image['image_url'], image_path)
                except:
                    print(f'Failed to download image, skip.')
                    continue
        image_caption_dict[idx] = {'caption': image['image_caption'],
                                   'path': image_path,
                                   'url': image['image_url']
                                   }
    return image_caption_dict


def load_support_img_caption_webqa(model, file_path='../dataset/WebQA/webqa_data/webqa_image.json', batch_size=512):
    image_path = '../dataset/WebQA/webqa_image/image_file/'
    with open(file_path, 'r') as f:
        image_data = json.load(f)
    image_caption_dict = {}
    for image in image_data.values():
        image_caption_dict[image['image_id']] = {'caption': image['caption'],
                                                 'path': image_path + str(image['image_id']) + '.jpg',
                                                 'url': image['url']
                                                 }
    return image_caption_dict


def load_support_img_caption_mmqa(model, file_path='../dataset/multimodalqa/dataset/MMQA_images.jsonl.gz',
                                  batch_size=512):
    image_data = read_jsonl_gz(file_path)
    image_path = '../dataset/multimodalqa/dataset/final_dataset_images/'
    image_caption_dict = {}

    # Batch process images for better performance
    for i in trange(0, len(image_data), batch_size, desc='Loading image captions'):
        batch_images = image_data[i:i + batch_size]
        image_files = []
        image_identifier = []

        # Collect valid image files for the batch
        for image in batch_images:
            image_files.append(image_path + image['path'])
            image_identifier.append(image['id'])

        try:
            # Process the batch of images
            batch_captions = model(image_files)
            # Extract captions from the batch results
            for idx, caption_result in enumerate(batch_captions):
                if isinstance(caption_result, list):
                    # Some models return list of results per image
                    image_caption_dict[image_identifier[idx]] = {'caption': caption_result[0]['generated_text'],
                                                                 'path': image_files[idx]
                                                                 }
                else:
                    # Others return single result
                    image_caption_dict[image_identifier[idx]] = {'caption': caption_result['generated_text'],
                                                                 'path': image_files[idx]
                                                                 }
        except Exception as e:
            # Fallback to individual processing if batch processing fails
            print(f'Batch processing failed with error: {e}. Falling back to individual processing.')
            for image in batch_images:
                try:
                    caption = model(image_path + image['path'])[0]['generated_text']
                    image_caption_dict[image['id']] = {'caption': caption,
                                                       'path': image_path + image['path']}
                except:
                    print(f'Failed to load image {image_path + image["path"]}')
                    continue

    return image_caption_dict


def load_support_text(dataset: str):
    support_text_list = []
    if dataset == 'mmqa':
        file_path = '../dataset/multimodalqa/dataset/MMQA_texts.jsonl.gz'
        text_data = read_jsonl_gz(file_path)
        for data in text_data:
            support_text_list.append(f'{data["title"]}\n{data["text"]}')
    elif dataset == 'webqa':
        file_path = '../dataset/WebQA/webqa_data/webqa_text.json'
        with open(file_path, 'r') as f:
            text_data = json.load(f)
        for data in text_data:
            support_text_list.append(f'{data["title"]}\n{data["text"]}')
    elif dataset in ['arxiv', 'manual', 'web', 'wiki', 'wit', 'recipe']:
        file_path = f'../dataset/MRAMG-Bench/doc_{dataset}.jsonl'
        text_data = read_jsonl(file_path)
        for idx, data, img_idx_list in text_data:
            support_text_list.append(data)
    return support_text_list
