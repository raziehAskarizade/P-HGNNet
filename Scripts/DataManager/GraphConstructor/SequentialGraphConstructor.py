# Omid Davar @ 2023

from typing import List

import networkx as nx
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data, HeteroData
from Scripts.Configs.ConfigClass import Config
import torch


class SequentialGraphConstructor(GraphConstructor):

    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(SequentialGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 load_preprocessed_data=False, naming_prepend='', use_general_node=False, use_compression=True, start_data_load=0, end_data_load=-1, num_general_nodes=1):

        super(SequentialGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, load_preprocessed_data,
                      naming_prepend, use_compression, start_data_load, end_data_load)
        self.settings = {"token_token_weight": 2, "general_token_weight": 2}
        self.use_general_node = use_general_node
        self.var.graph_num = len(self.raw_data)

        # farsi
        self.nlp = config.nlp
        self.token_lemma = config.token_lemma
        self.num_general_nodes = num_general_nodes
        
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
        if self.use_general_node:
            return self._create_graph_with_general_node(doc)
        else:
            return self._create_graph(doc)

    def _create_graph(self, doc, for_compression=False):
        docs_length = len(doc[1:])
        node_attr = torch.zeros(
            (len(doc[1:]), self.nlp.get_dimension()), dtype=torch.float32)
        if for_compression:
            node_attr = [-1 for i in range(len(doc[1:]))]
        edge_index = []
        edge_attr = []
        for i, token in enumerate(doc[1:]):
            token_id = self.nlp.get_word_id(token[2])
            if token_id != -1:
                if for_compression:
                    node_attr[i] = token_id
                else:
                    node_attr[i] = torch.tensor(
                        self.nlp.get_word_vector(token[2]))
            if i != len(doc[1:]) - 1:
                # using zero vectors for edge features
                edge_index.append([i, i + 1])
                edge_index.append([i + 1, i])
                edge_attr.append(self.settings["token_token_weight"])
                edge_attr.append(self.settings["token_token_weight"])
        edge_index = torch.transpose(torch.tensor(
            edge_index, dtype=torch.int32), 0, 1)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

    def _build_initial_general_vector(self, num: int = 1):
        return torch.zeros((num, self.nlp.get_dimension()), dtype=torch.float32)

    def _create_graph_with_general_node(self, doc, for_compression=False):
        data = HeteroData()
        if for_compression:
            data['general'].x = torch.full((1,), 0, dtype=torch.float32)
            data['word'].x = [-1 for i in range(len(doc[1:]))]
        else:
            data['general'].x = self._build_initial_general_vector()
            data['word'].x = torch.zeros(
                (len(doc[1:]), self.nlp.get_dimension()), dtype=torch.float32)
        word_general_edge_index = []
        general_word_edge_index = []
        word_word_edge_index = []
        word_general_edge_attr = []
        general_word_edge_attr = []
        word_word_edge_attr = []
        for i, token in enumerate(doc[1:]):
            token_id = self.nlp.get_word_id(token[2])
            if token_id != -1:
                if for_compression:
                    data['word'].x[i] = token_id
                else:
                    data['word'].x[i] = torch.tensor(
                        self.nlp.get_word_vector(token[2]))
            word_general_edge_index.append([i, 0])
            word_general_edge_attr.append(
                self.settings["general_token_weight"])
            general_word_edge_index.append([0, i])
            general_word_edge_attr.append(
                self.settings["general_token_weight"])
            # adding sequential edges between tokens - uncomment the codes for vectorized edges
            if i != len(doc[1:]) - 1:
                word_word_edge_index.append([i, i + 1])
                word_word_edge_attr.append(self.settings["token_token_weight"])
                word_word_edge_index.append([i + 1, i])
                word_word_edge_attr.append(self.settings["token_token_weight"])
        data['general', 'general_word', 'word'].edge_index = torch.transpose(torch.tensor(
            general_word_edge_index, dtype=torch.int32), 0, 1) if len(general_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'word_general', 'general'].edge_index = torch.transpose(torch.tensor(
            word_general_edge_index, dtype=torch.int32), 0, 1) if len(word_general_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'seq', 'word'].edge_index = torch.transpose(torch.tensor(word_word_edge_index, dtype=torch.int32), 0, 1) if len(
            word_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['general', 'general_word', 'word'].edge_attr = torch.tensor(
            general_word_edge_attr, dtype=torch.float32)
        data['word', 'word_general', 'general'].edge_attr = torch.tensor(
            word_general_edge_attr, dtype=torch.float32)
        data['word', 'seq', 'word'].edge_attr = torch.tensor(
            word_word_edge_attr, dtype=torch.float32)
        return data

    def draw_graph(self, idx: int):
        node_tokens = []
        if self.use_general_node:
            node_tokens.append("gen_node")

        # farsi
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        token_list = self.token_lemma(self.raw_data[idx])
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for token in sentence.words:
                doc.append((token.text, idx, token.lemma))

        for i, t in enumerate(doc[1:]):
            node_tokens.append(t[2])
        graph_data = self.get_graph(idx)
        g = to_networkx(graph_data)
        layout = nx.spring_layout(g)
        nx.draw(g, pos=layout)
        words_dict = {i: node_tokens[i] for i in range(len(node_tokens))}
        nx.draw_networkx_labels(g, pos=layout, labels=words_dict)

    def to_graph_indexed(self, text: str):
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
        if self.use_general_node:
            return self._create_graph_with_general_node(doc, for_compression=True)
        else:
            return self._create_graph(doc, for_compression=True)

    def get_word_by_id(self):
        words_id = {}
        for word in self.nlp.get_words():
            words_id[self.nlp.get_word_id(word)] = word
        return words_id

    def prepare_loaded_data(self, graph):
        if self.use_general_node:
            words = torch.zeros(
                (len(graph['word'].x), self.nlp.get_dimension()), dtype=torch.float32)

            for i in range(len(graph['word'].x)):
                if self.word_ids.get(int(graph['word'].x[i])) is not None:
                    words[i] = torch.tensor(
                        self.nlp.get_word_vector(self.word_ids[int(graph['word'].x[i])]))
            graph['word'].x = words
            graph = self._add_multiple_general_nodes(
                graph, False, self.num_general_nodes)
        else:
            words = torch.zeros(
                (len(graph.x), self.nlp.get_dimension()), dtype=torch.float32)
            for i in range(len(graph.x)):
                if graph.x[i] in self.nlp.get_words():
                    words[i] = torch.tensor(
                        self.nlp.get_word_vector(graph.x[i]))
            graph.x = words
        return graph

    def _add_multiple_general_nodes(self, graph, use_sentence_nodes, num_general_nodes):
        if not use_sentence_nodes:
            graph['general'].x = self._build_initial_general_vector(
                num=self.num_general_nodes)
            if self.num_general_nodes > 1:
                # connecting other general nodes
                general_word_edge_index = torch.transpose(torch.tensor(
                    graph['general', 'general_word', 'word'].edge_index, dtype=torch.int32), 0, 1).tolist()
                word_general_edge_index = torch.transpose(torch.tensor(
                    graph['word', 'word_general', 'general'].edge_index, dtype=torch.int32), 0, 1).tolist()
                general_word_edge_attr = graph['general',
                                               'general_word', 'word'].edge_attr.tolist()
                word_general_edge_attr = graph['word',
                                               'word_general', 'general'].edge_attr.tolist()
                for j in range(1, num_general_nodes):
                    for i in range(len(graph['word'].x)):
                        word_general_edge_index.append([i, j])
                        general_word_edge_index.append([j, i])
                        word_general_edge_attr.append(
                            self.settings["general_token_weight"])
                        general_word_edge_attr.append(
                            self.settings["general_token_weight"])

                # what is data here?
                # data['general' , 'general_word' , 'word'].edge_index = torch.transpose(torch.tensor(general_word_edge_index, dtype=torch.int32) , 0 , 1)
                # data['word' , 'word_general' , 'general'].edge_index = torch.transpose(torch.tensor(word_general_edge_index, dtype=torch.int32) , 0 , 1)
                graph['general', 'general_word', 'word'].edge_attr = torch.tensor(
                    general_word_edge_attr, dtype=torch.float32)
                graph['word', 'word_general', 'general'].edge_attr = torch.tensor(
                    word_general_edge_attr, dtype=torch.float32)
        else:
            graph['general'].x = self._build_initial_general_vector(
                num=self.num_general_nodes)
            if self.num_general_nodes > 1:
                # connecting other general nodes
                general_sentence_edge_index = torch.transpose(torch.tensor(
                    graph['general', 'general_sentence', 'sentence'].edge_index, dtype=torch.int32), 0, 1).tolist()
                sentence_general_edge_index = torch.transpose(torch.tensor(
                    graph['sentence', 'sentence_general', 'general'].edge_index, dtype=torch.int32), 0, 1).tolist()
                general_sentence_edge_attr = graph['general',
                                                   'general_sentence', 'sentence'].edge_attr.tolist()
                sentence_general_edge_attr = graph['sentence',
                                                   'sentence_general', 'general'].edge_attr.tolist()
                for j in range(1, num_general_nodes):
                    for i in range(len(graph['sentence'].x)):
                        sentence_general_edge_index.append([i, j])
                        general_sentence_edge_index.append([j, i])
                        sentence_general_edge_attr.append(
                            self.settings["general_sentence_weight"])
                        general_sentence_edge_attr.append(
                            self.settings["general_sentence_weight"])
                graph['general', 'general_sentence', 'sentence'].edge_index = torch.transpose(
                    torch.tensor(general_sentence_edge_index, dtype=torch.int32), 0, 1)
                graph['sentence', 'sentence_general', 'general'].edge_index = torch.transpose(
                    torch.tensor(sentence_general_edge_index, dtype=torch.int32), 0, 1)
                graph['general', 'general_sentence', 'sentence'].edge_attr = torch.tensor(
                    general_sentence_edge_attr, dtype=torch.float32)
                graph['sentence', 'sentence_general', 'general'].edge_attr = torch.tensor(
                    sentence_general_edge_attr, dtype=torch.float32)
        return graph
