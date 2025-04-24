# Fardin Rastakhiz @ 2023 - Razieh Askarizade @ 2025

import json
from os import path
import stanza
from stanza.pipeline.core import DownloadMethod
import fasttext


class Config:

    def __init__(self, project_root_path: str, config_local_path: str = ''):
        self.root = project_root_path
        self.token_lemma = stanza.Pipeline(
            "fa", download_method=DownloadMethod.REUSE_RESOURCES, processors=["tokenize", "lemma"])

        if config_local_path == '':
            config_local_path = 'Scripts/Configs/Config.json'
        config_path = path.join(self.root, config_local_path)

        with open(config_path, 'rt') as cf:
            config_data = json.load(cf)

        self.device = config_data['device'] if 'device' in config_data else 'cpu'

        # farsi
        if 'fa' in config_data:
            self.fa: FaConfig = FaConfig(config_data['fa'])

        if 'datasets' in config_data:
            self.datasets = Datasets(config_data['datasets'])

        self.nlp_pipeline = self.fa.pipeline
        self.nlp = fasttext.load_model(self.nlp_pipeline)

# farsi


class FaConfig:
    def __init__(self, json_data: dict):
        if 'pipeline' in json_data:
            self.pipeline: str = json_data['pipeline']


class Datasets:
    def __init__(self, json_data: dict):
        if 'digikala' in json_data:
            self.dagikala = Dataset(json_data=['digikala'])


class Dataset:
    def __init__(self, json_data: dict):
        if 'train' in json_data:
            self.train = json_data['train']
        if 'validation' in json_data:
            self.validation = json_data['validation']
        if 'test' in json_data:
            self.test = json_data['test']
