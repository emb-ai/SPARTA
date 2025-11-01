import json
import os

def read_json_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                file_texts = data.get('text', [])
                texts.extend(file_texts)
    return texts


directory = '../../dataset/reason_seg/ReasonSeg/test'
texts = read_json_files(directory)