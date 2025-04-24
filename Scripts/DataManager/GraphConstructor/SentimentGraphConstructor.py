# Fardin Rastakhiz @ 2023 - Razieh Askarizade @ 2025

from typing import List
import pandas as pd

from Scripts.DataManager.GraphConstructor.TagDepTokenGraphConstructor import TagDepTokenGraphConstructor
from Scripts.Configs.ConfigClass import Config
import torch
import numpy as np


class SentimentGraphConstructor(TagDepTokenGraphConstructor):

    class _Variables(TagDepTokenGraphConstructor._Variables):
        def __init__(self):
            
            super(SentimentGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 load_preprocessed_data=False, naming_prepend='', use_compression=True, use_sentence_nodes=False, use_general_node=True, start_data_load=0, end_data_load=-1, num_general_nodes=1):

        super(SentimentGraphConstructor, self)\
            .__init__(texts, save_path, config, load_preprocessed_data,
                      naming_prepend, use_compression, use_sentence_nodes, use_general_node, start_data_load, end_data_load, num_general_nodes)
            
        self.inital_sentiment_nodes = self._build_initial_sentiment_vector()
        self.polarity_df, self.words_ids, self.ids_words = self.persion_polarity()

    def persion_polarity(self, dataset_path=r"data\PerSent.xlsx"):
        xlsx = pd.ExcelFile(dataset_path)
        df = xlsx.parse('Dataset')
        words_ids = {df.iloc[i]['Words']:i for i in df.index}
        ids_words = {v:k for k,v in words_ids.items()}
        return df, words_ids, ids_words

    def to_graph(self, text: str):
        # farsi
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        token_list = self.token_lemma(str(text))
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for word in sentence.words:
                doc.append((idx, word.text, word.lemma,
                            word.upos, word.head, word.deprel))
        return self.__create_sentiment_graph(doc)

    def _build_initial_sentiment_vector(self):
        return torch.zeros((2, self.nlp.get_dimension()), dtype=torch.float32)

    def __create_sentiment_graph(self, doc, token_list, for_compression=False):
        if for_compression:
            data = super().to_graph_indexed_token_list(token_list)
        else:
            data = super().to_graph_token_list(token_list)
        # adding sentiment nodes
        if for_compression:
            data['sentiment'].x = torch.full((2,), -1, dtype=torch.float32)
        else:
            data['sentiment'].x = self.inital_sentiment_nodes #self._build_initial_sentiment_vector()
        word_sentiment_edge_index = []
        word_sentiment_edge_attr = []

        df = self.polarity_df
        for i, token in enumerate(doc[1:]):
            if token[2] in self.words_ids:
                polarity = df.iloc[self.words_ids[token[2]]].Polarity
                if abs(polarity) > 0:
                    word_sentiment_edge_attr.append(abs(polarity))
                    word_sentiment_edge_index.append([i, 1 if polarity > 0 else 0])
                
        data['word', 'word_sentiment', 'sentiment'].edge_index = torch.transpose(torch.from_numpy( np.array(
            word_sentiment_edge_index, dtype=np.int32)), 0, 1) if len(word_sentiment_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['sentiment', 'sentiment_word', 'word'].edge_index = data['word', 'word_sentiment', 'sentiment'].edge_index[[1,0]]
        
        data['word', 'word_sentiment', 'sentiment'].edge_attr = torch.from_numpy( np.array(word_sentiment_edge_attr, dtype=np.float32))
        data['sentiment', 'sentiment_word', 'word'].edge_attr = data['word', 'word_sentiment', 'sentiment'].edge_attr
        return data

    def to_graph_indexed(self, text: str):
        # farsi
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        token_list = self.token_lemma(text)
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for word in sentence.words:
                doc.append((idx, word.text, word.lemma,
                            word.upos, word.head, word.deprel))
        return self.__create_sentiment_graph(doc, token_list, for_compression=True)

    def prepare_loaded_data(self, graph):
        graph = super(SentimentGraphConstructor, self).prepare_loaded_data(graph)
        graph['sentiment'].x = self._build_initial_sentiment_vector()
        for t in graph.edge_types:
            if graph[t].edge_index.shape[1] == 0:
                graph[t].edge_index = torch.empty(2, 0, dtype=torch.int32)
        return graph

    def remove_node_type_from_graphs(self, node_name: str):
        for i in range(len(self._graphs)):
            if self._graphs[i] is not None:
                if node_name in self._graphs[i].node_types:
                    del self._graphs[i][node_name]
                for edge_type in self._graphs[i].edge_types:
                    if edge_type[0] == node_name or edge_type[1] == node_name:
                        del self._graphs[i][edge_type]
