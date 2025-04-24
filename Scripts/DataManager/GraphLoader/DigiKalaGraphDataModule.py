# Fardin Rastakhiz @ 2023 - Razieh Askarizade @ 2025

from copy import copy
import numpy as np
from os import path
from typing import Dict

import pandas as pd
from torch_geometric.loader import DataLoader

from Scripts.Configs.ConfigClass import Config
from Scripts.DataManager.GraphConstructor.TagDepTokenGraphConstructor import TagDepTokenGraphConstructor
from Scripts.DataManager.GraphConstructor.SentimentGraphConstructor import SentimentGraphConstructor
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor, TextGraphType
from Scripts.DataManager.GraphLoader.GraphDataModule import GraphDataModule
from Scripts.Preprocess.digi_oversampling import oversampling
from torch.utils.data.dataset import Subset
import torch
from Scripts.DataManager.Datasets.GraphConstructorDataset import GraphConstructorDatasetRanged

from stanza.pipeline.core import DownloadMethod
import stanza
import time


class DigiKalaGraphDataModule(GraphDataModule):

    def __init__(self, config: Config, test_size=0.2, val_size=0.2, num_workers=2, drop_last=True, dataset_path='', graphs_path='', batch_size=32, device='cpu', shuffle=False, start_data_load=0, end_data_load=-1, graph_type: TextGraphType = TextGraphType.FULL, load_preprocessed_data=True, reweights={}, removals=[], max_length=1024, *args, **kwargs):
        super(DigiKalaGraphDataModule, self)\
            .__init__(config, device, test_size, val_size, *args, **kwargs)

        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.graph_type = graph_type
        self.reweights = reweights
        self.removals = removals

        self.graphs_path = graphs_path if graphs_path != '' else 'data/GraphData/DigiKala'
        self.dataset_path = 'data/DigiKala/data.csv' if dataset_path == '' else dataset_path

        self.labels = None
        self.dataset = None
        self.shuffle = shuffle
        self.start_data_load = start_data_load
        self.end_data_load = end_data_load
        self.df: pd.DataFrame = pd.DataFrame()
        self.__train_dataset, self.__val_dataset, self.__test_dataset = None, None, None
        self.load_preprocessed_data = load_preprocessed_data

    def load_labels(self):
        print("load labels")
        self.train_df, self.test_df = oversampling(self.config.root, self.dataset_path)
        self.train_df.columns = ['Opinion', 'rate_class']
        self.test_df.columns = ['Opinion', 'rate_class']
        

        self.df = pd.concat([self.train_df, self.test_df])
        # dropna (12 rows)
        self.df = self.df.dropna(subset=['Opinion'])
        

        self.df['Opinion'] = self.df['Opinion'].apply(lambda t: t[:self.max_length])
        texts2 = []
        nlp = stanza.Pipeline("fa", download_method=DownloadMethod.REUSE_RESOURCES, processors="tokenize")
        for row in self.df.Opinion.values:
            doc = nlp(row)
            tokens = [t.text for sent in doc.sentences for t in sent.tokens]
            while len(tokens) < 2:
                tokens.append("#")
            texts2.append(' '.join(tokens))
        self.df['Text'] = texts2

        self.end_data_load = self.end_data_load if self.end_data_load > 0 else self.df.shape[
            0]
        self.end_data_load = self.end_data_load if self.end_data_load < self.df.shape[
            0] else self.df.shape[0]
        self.df = self.df.iloc[self.start_data_load:self.end_data_load]
        self.df.index = np.arange(0, self.end_data_load - self.start_data_load)
        # activate one line below
        labels = self.df['rate_class'][:self.end_data_load -
                                       self.start_data_load]
        labels = labels.to_numpy()
        labels = torch.from_numpy(labels)
        self.num_classes = len(torch.unique(labels))
        self.labels = torch.nn.functional.one_hot(
            (labels-1).to(torch.int64)).to(torch.float32).to(self.device)

        self.num_data = self.df.shape[0]
        self.train_range = range(
            int((1-self.val_size-self.test_size)*self.num_data))
        self.val_range = range(
            self.train_range[-1]+1, int((1-self.test_size)*self.num_data))
        self.test_range = range(self.val_range[-1]+1, self.num_data)

    def load_graphs(self):
        print("load graphs")
        st = time.time()
        self.graph_constructors = self.__set_graph_constructors(self.graph_type)
        print(f"init Completed: {(time.time() - st)/1000:.4f} ms")
        st = time.time()
        self.dataset, self.num_node_features = {}, {}
        self.__train_dataset, self.__val_dataset, self.__test_dataset = {}, {}, {}
        self.__train_dataloader, self.__test_dataloader, self.__val_dataloader = {}, {}, {}
        for key in self.graph_constructors:
            self.graph_constructors[key].setup(self.load_preprocessed_data)
            print(f"setup Completed: {(time.time() - st)/1000:.4f} ms")
            st = time.time()
            # reweighting
            if key in self.reweights:
                for r in self.reweights[key]:
                    self.graph_constructors[key].reweight_all(r[0], r[1])
            # removals
            if isinstance(self.graph_constructors[key], SentimentGraphConstructor):
                for node_type in self.removals:
                    self.graph_constructors[key].remove_node_type_from_graphs(
                        node_type)
            self.dataset[key] = GraphConstructorDatasetRanged(
                self.graph_constructors[key], self.labels, self.start_data_load, self.end_data_load)

            self.__train_dataset[key] = Subset(
                self.dataset[key], self.train_range)
            self.__val_dataset[key] = Subset(self.dataset[key], self.val_range)
            self.__test_dataset[key] = Subset(
                self.dataset[key], self.test_range)

            self.__train_dataloader[key] = DataLoader(self.__train_dataset[key], batch_size=self.batch_size,
                                                      drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] = DataLoader(
                self.__test_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] = DataLoader(
                self.__val_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)

        self.set_active_graph(key)

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

        for key in self.graph_constructors:
            self.__train_dataloader[key] = DataLoader(self.__train_dataset[key], batch_size=self.batch_size,
                                                      drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] = DataLoader(
                self.__test_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] = DataLoader(
                self.__val_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)

        self.set_active_graph(key)

    def get_data(self, datamodule):
        self.labels = datamodule.labels
        self.num_classes = datamodule.num_classes
        self.graph_constructors = datamodule.graph_constructors
        self.dataset, self.num_node_features = datamodule.dataset, datamodule.num_node_features
        self.__train_dataset, self.__val_dataset, self.__test_dataset = datamodule.__train_dataset, datamodule.__val_dataset, datamodule.__test_dataset
        self.__train_dataloader, self.__test_dataloader, self.__val_dataloader = datamodule.__train_dataloader, datamodule.__test_dataloader, datamodule.__val_dataloader
        self.set_active_graph(datamodule.active_key)

    def set_active_graph(self, graph_type: TextGraphType = TextGraphType.FULL):
        assert graph_type in self.dataset, 'The provided key is not valid'
        self.active_key = graph_type
        sample_graph = self.graph_constructors[self.active_key].get_first()
        self.num_node_features = sample_graph.num_features

    def create_sub_data_loader(self, begin: int, end: int):
        for key in self.graph_constructors:
            dataset = GraphConstructorDatasetRanged(
                self.graph_constructors[key], self.labels, begin, end)

            train_dataset = Subset(dataset, self.train_range)
            val_dataset = Subset(dataset,  self.val_range)
            test_dataset = Subset(dataset,  self.test_range)

            self.__train_dataloader[key] = DataLoader(
                train_dataset, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] = DataLoader(
                test_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] = DataLoader(
                val_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)

        self.set_active_graph(key)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        pass

    def teardown(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return self.__train_dataloader[self.active_key]

    def test_dataloader(self):
        return self.__test_dataloader[self.active_key]

    def val_dataloader(self):
        return self.__val_dataloader[self.active_key]

    def __set_graph_constructors(self, graph_type: TextGraphType):
        graph_type = copy(graph_type)
        graph_constructors: Dict[TextGraphType, GraphConstructor] = {}

        tag_dep_seq_sent = TextGraphType.DEPENDENCY | TextGraphType.TAGS | TextGraphType.SEQUENTIAL | TextGraphType.SENTENCE
        if tag_dep_seq_sent in graph_type:
            graph_constructors[tag_dep_seq_sent] = self.__get_full_graph()
            graph_type = graph_type - tag_dep_seq_sent


        if TextGraphType.FULL_SENTIMENT in graph_type:
            graph_constructors[TextGraphType.FULL_SENTIMENT] = self.__get_sentiment_graph()
            graph_type = graph_type - TextGraphType.FULL_SENTIMENT

        return graph_constructors
    
    def __get_full_graph(self):
        return TagDepTokenGraphConstructor(self.df['Text'][:self.end_data_load], path.join(self.graphs_path, 'full'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load, use_sentence_nodes=True, use_general_node=True)

    def __get_sentiment_graph(self):
        return SentimentGraphConstructor(self.df['Text'][:self.end_data_load], path.join(self.graphs_path, 'sentiment'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load, use_sentence_nodes=True, use_general_node=True)

    def zero_rule_baseline(self):
        return f'zero_rule baseline: {(len(self.labels[self.labels>0.5])* 100.0 / len(self.labels))  : .2f}%'
