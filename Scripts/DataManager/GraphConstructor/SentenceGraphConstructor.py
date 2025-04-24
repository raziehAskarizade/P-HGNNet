# Fardin Rastakhiz @ 2023 - Razieh Askarizade @ 2025


from typing import List
from Scripts.DataManager.GraphConstructor.SequentialGraphConstructor import SequentialGraphConstructor
from torch_geometric.data import HeteroData
from Scripts.Configs.ConfigClass import Config

import torch
import numpy as np

# farsi
import copy


class SentenceGraphConstructor(SequentialGraphConstructor):

    class _Variables(SequentialGraphConstructor._Variables):
        def __init__(self):
            super(SentenceGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 load_preprocessed_data=False, naming_prepend='', use_general_node=False, use_compression=True, start_data_load=0, end_data_load=-1, num_general_nodes=1):

        super(SentenceGraphConstructor, self)\
            .__init__(texts, save_path, config, load_preprocessed_data,
                      naming_prepend, False, use_compression, start_data_load, end_data_load, num_general_nodes)
        self.settings = {"token_sentence_weight": 1,
                         "token_token_weight": 2, "general_sentence_weight": 2}
        self.use_general_node = use_general_node
        self.var.graph_num = len(self.raw_data)

        # farsi
        self.nlp = config.nlp

        self.token_lemma = config.token_lemma

        self.word_ids = self.get_word_by_id()

    def to_graph(self, text: str):

        # farsi
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        token_list = self.token_lemma(text)
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for token in sentence.words:
                doc.append((token.text, idx, token.lemma))

        if len(doc[1:]) < 2:
            return
        return self.__create_sentence_graph(doc)

    def __create_sentence_graph(self, doc, for_compression=False):
        sequential_data = super()._create_graph(doc, for_compression)  # homogeneous
        data = HeteroData()
        sentence_embeddings = np.array(
            [self.nlp.get_sentence_vector(sent[0]) for sent in doc[0]])
        data['word'].x = sequential_data.x
        if for_compression:
            if self.use_general_node:
                data['general'].x = torch.full((1,), -1, dtype=torch.float32)
        else:
            if self.use_general_node:
                data['general'].x = self._build_initial_general_vector()
        data['sentence'].x = torch.tensor(
            sentence_embeddings, dtype=torch.float32)
        sentence_general_edge_index = []
        general_sentence_edge_index = []
        sentence_word_edge_index = []
        word_sentence_edge_index = []
        sentence_general_edge_attr = []
        general_sentence_edge_attr = []
        sentence_word_edge_attr = []
        word_sentence_edge_attr = []
        if self.use_general_node:
            for i, _x in enumerate(doc[0]):
                # connecting sentences to general node
                sentence_general_edge_index.append([i, 0])
                general_sentence_edge_index.append([0, i])
                sentence_general_edge_attr.append(
                    self.settings['general_sentence_weight'])
                # different weight for directed edges can be set in the future
                general_sentence_edge_attr.append(
                    self.settings['general_sentence_weight'])
        sent_index = -1
        doc_copy = copy.deepcopy(doc)
        for i, token in enumerate(doc_copy[1:]):
            # connecting words to sentences
            for j, sent_start in enumerate(doc_copy[0]):
                if token[0] == sent_start[1] and token[1] == sent_start[2]:
                    sent_index += 1
                    doc_copy[0].pop(j)
            word_sentence_edge_index.append([i, sent_index])
            sentence_word_edge_index.append([sent_index, i])
            word_sentence_edge_attr.append(
                self.settings['token_sentence_weight'])
            sentence_word_edge_attr.append(
                self.settings['token_sentence_weight'])
        if self.use_general_node:
            data['general', 'general_sentence', 'sentence'].edge_index = torch.transpose(torch.tensor(
                general_sentence_edge_index, dtype=torch.int32), 0, 1) if len(general_sentence_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
            data['sentence', 'sentence_general', 'general'].edge_index = torch.transpose(torch.tensor(
                sentence_general_edge_index, dtype=torch.int32), 0, 1) if len(sentence_general_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
            data['general', 'general_sentence', 'sentence'].edge_attr = torch.tensor(
                general_sentence_edge_attr, dtype=torch.float32)
            data['sentence', 'sentence_general', 'general'].edge_attr = torch.tensor(
                sentence_general_edge_attr, dtype=torch.float32)
        data['word', 'word_sentence', 'sentence'].edge_index = torch.transpose(torch.tensor(
            word_sentence_edge_index, dtype=torch.int32), 0, 1) if len(word_sentence_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['sentence', 'sentence_word', 'word'].edge_index = torch.transpose(torch.tensor(
            sentence_word_edge_index, dtype=torch.int32), 0, 1) if len(sentence_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'seq', 'word'].edge_index = sequential_data.edge_index
        data['word', 'word_sentence', 'sentence'].edge_attr = torch.tensor(
            word_sentence_edge_attr, dtype=torch.float32)
        data['sentence', 'sentence_word', 'word'].edge_attr = torch.tensor(
            sentence_word_edge_attr, dtype=torch.float32)
        data['word', 'seq', 'word'].edge_attr = sequential_data.edge_attr
        return data

    def to_graph_indexed(self, text: str):
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        token_list = self.token_lemma(text)
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for token in sentence.words:
                doc.append((token.text, idx, token.lemma))

        if len(doc[1:]) < 2:
            return
        return self.__create_sentence_graph(doc, for_compression=True)

    def get_word_by_id(self):
        words_id = {}
        for word in self.nlp.get_words():
            words_id[self.nlp.get_word_id(word)] = word
        return words_id

    def prepare_loaded_data(self, graph):
        words = torch.zeros(
            (len(graph['word'].x), self.nlp.get_dimension()), dtype=torch.float32)

        for i in range(len(graph['word'].x)):
            if self.word_ids.get(int(graph['word'].x[i])) is not None:
                words[i] = torch.tensor(
                    self.nlp.get_word_vector(self.word_ids[int(graph['word'].x[i])]))
        graph['word'].x = words
        if self.use_general_node:
            graph = self._add_multiple_general_nodes(
                graph, True, self.num_general_nodes)
        for t in graph.edge_types:
            if len(graph[t].edge_index) == 0:
                graph[i].edge_index = torch.empty(2, 0, dtype=torch.int32)
        return graph
