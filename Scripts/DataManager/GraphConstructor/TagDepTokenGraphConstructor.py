# Fardin Rastakhiz @ 2023 - Razieh Askarizade @ 2025

import numpy as np
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import HeteroData
from Scripts.Configs.ConfigClass import Config
import torch
from typing import List

import stanza
import copy
from stanza.pipeline.core import DownloadMethod


class TagDepTokenGraphConstructor(GraphConstructor):

    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(TagDepTokenGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 load_preprocessed_data=False, naming_prepend='', use_compression=True, use_sentence_nodes=False, use_general_node=True, start_data_load=0, end_data_load=-1, num_general_nodes=1):

        super(TagDepTokenGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, load_preprocessed_data, naming_prepend, use_compression, start_data_load, end_data_load)
        self.settings = {"dep_token_weight": 1, "token_token_weight": 2, "tag_token_weight": 1,
                         "general_token_weight": 1, "general_sentence_weight": 1, "token_sentence_weight": 1}
        self.use_sentence_nodes = use_sentence_nodes
        self.use_general_node = use_general_node

        self.var.graph_num = len(self.raw_data)

        # farsi
        self.nlp = config.nlp

        self.token_lemma = stanza.Pipeline(
            "fa", download_method=DownloadMethod.REUSE_RESOURCES, processors=["tokenize", "lemma", "pos", "depparse"])#, "depparse"

        # self.dependencies = ['acl', 'acl:relcl', 'advcl', 'advcl:relcl', 'advmod', 'advmod:emph', 'advmod:lmod', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'cc', 'cc:preconj', 'ccomp', 'clf', 'compound', 'compound:lvc', 'compound:prt', 'compound:redup', 'compound:svc', 'conj', 'cop', 'csubj', 'csubj:outer', 'csubj:pass', 'dep', 'det', 'det:numgov', 'det:nummod', 'det:poss', 'discourse', 'dislocated', 'expl', 'expl:impers', 'expl:pass', 'expl:pv', 'fixed', 'flat', 'flat:foreign', 'flat:name', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nmod:poss', 'nmod:tmod', 'nsubj', 'nsubj:outer', 'nsubj:pass', 'nummod', 'nummod:gov', 'obj', 'obl', 'obl:agent', 'obl:arg', 'obl:lmod', 'obl:tmod', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']
        self.dependencies = ['nmod', 'case', 'conj', 'advmod', 'obl', 'amod', 'nsubj', 'cc', 'obj']#, 'compound:lvc', 'dep']

        self.dep_idx = {d:idx for idx, d in enumerate(self.dependencies)}
        self.idx_dep = {idx:d for idx, d in enumerate(self.dependencies)}
        self.dep_range = torch.arange(0, len(self.dependencies))
        
        # self.tags = ['NOUN', 'DET', 'PROPN', 'NUM', 'VERB', 'PART', 'PRON', 'SCONJ', 'ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ', 'X']
        self.tags= ['NOUN', 'ADJ', 'VERB', 'ADP', 'ADV', 'CCONJ', 'PRON', 'SCONJ', 'PROPN', 'DET', 'AUX']#, 'NUM']
        self.tags_idx = {t:idx for idx, t in enumerate(self.tags)}
        self.idx_tags = {idx:t for idx, t in enumerate(self.tags)}
        self.tag_range = torch.arange(0, len(self.tags))

        self.dep_length = len(self.dependencies)
        self.tag_length = len(self.tags)
        
        self.num_general_nodes = num_general_nodes
        self.initial_general_nodes = self.__build_initial_general_vector(self.num_general_nodes)

        self.word_ids = self.get_word_by_id()

    def to_graph(self, text: str):
        # farsi
        token_list = self.token_lemma(text)
        self.to_graph_token_list(token_list)
        
    def to_graph_token_list(self, token_list):
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for word in sentence.words:
                doc.append((idx, word.text, word.lemma,
                            word.upos, word.head, word.deprel))

        # if len(doc) < 2:
        #     return
        if self.use_sentence_nodes:
            return self.__create_graph_with_sentences(doc)
        else:
            return self.__create_graph(doc, use_general_node=self.use_general_node)

    def __find_dep_index(self, dependency: str): 
        return self.dep_idx[dependency] if dependency in self.dep_idx else -1 


    def __find_tag_index(self, tag: str):
        return self.tags_idx[tag] if tag in self.tags_idx else -1

    def __build_initial_general_vector(self, num: int = 1):
        return torch.zeros((num, self.nlp.get_dimension()), dtype=torch.float32)

    def __create_graph_with_sentences(self, doc, for_compression=False):
        data = self.__create_graph(doc, for_compression, False)
        sentence_embeddings = np.array(
            [self.nlp.get_sentence_vector(sent[0]) for sent in doc[0]])
        data['sentence'].x = torch.tensor(sentence_embeddings, dtype=torch.float32)
        if self.use_general_node:
            if for_compression:
                data['general'].x = torch.full((1,), 0, dtype=torch.float32)
            else:
                data['general'].x = self.initial_general_nodes #self.__build_initial_general_vector()
            
        word_sentence_edge_index = []
            
        sent_index = -1
        doc_copy = copy.deepcopy(doc)
        for i, token in enumerate(doc[1:]):
            # connecting words to sentences
            for j, sent_start in enumerate(doc_copy[0]):
                if token[1] == sent_start[1] and token[0] == sent_start[2]:
                    sent_index += 1
                    doc_copy[0].pop(j)
            if sent_index == -1: sent_index = 0
            word_sentence_edge_index.append([i, sent_index])
        if self.use_general_node:
            data['general', 'general_sentence', 'sentence'].edge_index = torch.concat(
                [torch.arange(0, len(doc[0]), dtype=torch.int).unsqueeze(0),
                 torch.zeros((1, len(doc[0])), dtype=torch.int)])
            data['sentence', 'sentence_general', 'general'].edge_index = data['general', 'general_sentence', 'sentence'].edge_index[[1, 0]]
            data['general', 'general_sentence', 'sentence'].edge_attr = torch.full((data['general', 'general_sentence', 'sentence'].edge_index.shape[1],), self.settings["general_sentence_weight"],dtype=torch.float32)
            data['sentence', 'sentence_general', 'general'].edge_attr = data['general', 'general_sentence', 'sentence'].edge_attr
        data['word', 'word_sentence', 'sentence'].edge_index = torch.transpose(
            torch.from_numpy(np.array(word_sentence_edge_index, dtype=np.int32)), 0, 1)
        data['sentence', 'sentence_word', 'word'].edge_index = data['word', 'word_sentence', 'sentence'].edge_index[[1, 0]]
        data['word', 'word_sentence', 'sentence'].edge_attr = torch.full((data['word', 'word_sentence', 'sentence'].edge_index.shape[1], ), self.settings['token_sentence_weight'], dtype=torch.float32)
        data['sentence', 'sentence_word', 'word'].edge_attr = data['word', 'word_sentence', 'sentence'].edge_attr
        return data

    def __create_graph(self, doc, for_compression=False, use_general_node=True):
        # nodes size is dependencies + tokens
        data = HeteroData()
        data['dep'].length = self.dep_length
        data['tag'].length = self.tag_length
        if for_compression:
            data['dep'].x = torch.full((self.dep_length,), -1, dtype=torch.float32)
            data['word'].x = torch.full((len(doc),), -1, dtype=torch.float32)
            data['tag'].x = torch.full((self.tag_length,), -1, dtype=torch.float32)
            if use_general_node:
                data['general'].x = torch.full((1,), -1, dtype=torch.float32)
        else:
            data['dep'].x = torch.arange(0, self.dep_length)
            data['word'].x = torch.zeros((len(doc), self.nlp.get_dimension()), dtype=torch.float32)
            data['tag'].x = torch.arange(0, self.tags_length)
            if use_general_node:
                data['general'].x = self.initial_general_nodes # torch.zeros((1, self.nlp.get_dimension()), dtype=torch.float32)
                
        dep_word_edge_index = []
        tag_word_edge_index = []
        word_word_edge_index = []
        general_word_edge_index = []
        
        for i, token in enumerate(doc[1:]):
            token_id = self.nlp.get_word_id(token[2])
            if token_id != -1:
                if for_compression:
                    data['word'].x[i] = token_id
                else:
                    data['word'].x[i] = torch.from_numpy(np.array(
                        self.nlp.get_word_vector(token[2])))
            # adding dependency edges
            if token[5] != 'root':
                dep_idx = self.__find_dep_index(token[5])
                if dep_idx != -1:
                    dep_word_edge_index.append([dep_idx, i])
                    
            # adding tag edges
            tag_idx = self.__find_tag_index(token[3])
            if tag_idx != -1:
                tag_word_edge_index.append([tag_idx, i])
            # adding sequence edges
            if i != len(doc) - 1:
                word_word_edge_index.append([i, i + 1])
                word_word_edge_index.append([i + 1, i])
            # adding general node edges
            if use_general_node:
                general_word_edge_index.append([0, i])
                
        data['dep', 'dep_word', 'word'].edge_index = torch.transpose(torch.from_numpy(np.array(
            dep_word_edge_index, dtype=np.int32)), 0, 1) if len(dep_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'word_dep', 'dep'].edge_index = data['dep', 'dep_word', 'word'].edge_index[[1, 0]]
        
        data['tag', 'tag_word', 'word'].edge_index = torch.transpose(torch.from_numpy(np.array(
            tag_word_edge_index, dtype=np.int32)), 0, 1) if len(tag_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'word_tag', 'tag'].edge_index = data['tag', 'tag_word', 'word'].edge_index[[1, 0]]

        data['word', 'seq', 'word'].edge_index = torch.transpose(torch.from_numpy(np.array(word_word_edge_index, dtype=np.int32)), 0, 1) if len(word_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        
        data['dep', 'dep_word', 'word'].edge_attr = torch.full((data['dep', 'dep_word', 'word'].edge_index.shape[1],), self.settings["dep_token_weight"], dtype=torch.float32)
        data['word', 'word_dep', 'dep'].edge_attr = data['dep', 'dep_word', 'word'].edge_attr
        
        data['tag', 'tag_word', 'word'].edge_attr = torch.full((data['tag', 'tag_word', 'word'].edge_index.shape[1],), self.settings["tag_token_weight"], dtype=torch.float32)
        data['word', 'word_tag', 'tag'].edge_attr = data['tag', 'tag_word', 'word'].edge_attr
        
        data['word', 'seq', 'word'].edge_attr = torch.full((data['word', 'seq', 'word'].edge_index.shape[1],), self.settings["token_token_weight"], dtype=torch.float32)
        
        if use_general_node:
            data['general', 'general_word', 'word'].edge_index = torch.transpose(torch.from_numpy(np.array(
                general_word_edge_index, dtype=np.int32)), 0, 1) if len(general_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
            data['word', 'word_general', 'general'].edge_index = data['general', 'general_word', 'word'].edge_index [[1, 0]]
            
            data['general', 'general_word', 'word'].edge_attr = torch.full((data['general', 'general_word', 'word'].edge_index.shape[1],), self.settings["general_token_weight"], dtype=torch.float32)
            data['word', 'word_general', 'general'].edge_attr = data['general', 'general_word', 'word'].edge_attr
        return data

    def draw_graph(self, idx: int):
        # TODO : do this part if needed
        pass

    def to_graph_indexed(self, text: str):
        token_list = self.token_lemma(str(text))
        self.to_graph_indexed_token_list(token_list)
    
    def to_graph_indexed_token_list(self, token_list):
        
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for word in sentence.words:
                doc.append((idx, word.text, word.lemma,
                            word.upos, word.head, word.deprel))
        if self.use_sentence_nodes:
            return self.__create_graph_with_sentences(doc, for_compression=True)
        else:
            return self.__create_graph(doc, for_compression=True, use_general_node=self.use_general_node)

    def get_word_by_id(self):
        words_id = {}
        for word in self.nlp.get_words():
            words_id[self.nlp.get_word_id(word)] = word
        return words_id

    def prepare_loaded_data(self, graph):
        words = torch.zeros(
            (len(graph['word'].x), self.nlp.get_dimension()), dtype=torch.float32)

        for i in range(len(graph['word'].x)):
            ids = int(graph['word'].x[i])
            if ids in self.word_ids:
                words[i] = torch.from_numpy(np.array(self.nlp.get_word_vector(self.word_ids[ids])))
        graph['word'].x = words
        graph['dep'].x = self.dep_range
        graph['tag'].x = self.tag_range
        if self.use_general_node:
            graph = self._add_multiple_general_nodes(graph, self.use_sentence_nodes, self.num_general_nodes)
        for t in graph.edge_types:
            if len(graph[t].edge_index) == 0:
                graph[i].edge_index = torch.empty(2, 0, dtype=torch.int32)
        return graph

    def _add_multiple_general_nodes(self, graph, use_sentence_nodes, num_general_nodes):
        if not use_sentence_nodes:
            graph['general'].x = self.initial_general_nodes # self.__build_initial_general_vector(num=self.num_general_nodes)
            if self.num_general_nodes > 1:
                # connecting other general nodes
                general_word_edge_index = []
                for j in range(1, num_general_nodes):
                    for i in range(len(graph['word'].x)):
                        general_word_edge_index.append([j, i])
                        
                graph['general', 'general_word', 'word'].edge_index = torch.transpose(
                    torch.tensor(general_word_edge_index, dtype=torch.int32), 0, 1)
                graph['word', 'word_general', 'general'].edge_index = graph['general', 'general_word', 'word'].edge_index[[1, 0]]
                graph['general', 'general_word', 'word'].edge_attr = torch.full((graph['general', 'general_word', 'word'].edge_index.shape[1],), self.settings["general_token_weight"], dtype=torch.float32)
                graph['word', 'word_general', 'general'].edge_attr = graph['general', 'general_word', 'word'].edge_attr
        else:
            graph['general'].x = self.initial_general_nodes #self.__build_initial_general_vector(num=self.num_general_nodes)
            if self.num_general_nodes > 1:
                # connecting other general nodes
                general_sentence_edge_index = torch.transpose(torch.tensor(graph['general', 'general_sentence', 'sentence'].edge_index, dtype=torch.int32), 0, 1).tolist()
                for j in range(1, num_general_nodes):
                    for i in range(len(graph['sentence'].x)):
                        general_sentence_edge_index.append([j, i])
                graph['general', 'general_sentence', 'sentence'].edge_index = torch.transpose(
                    torch.tensor(general_sentence_edge_index, dtype=torch.int32), 0, 1)
                graph['sentence', 'sentence_general', 'general'].edge_index = graph['general', 'general_sentence', 'sentence'].edge_index[[1, 0]]
                graph['general', 'general_sentence', 'sentence'].edge_attr = torch.full((graph['general', 'general_word', 'word'].edge_index.shape[1],), self.settings["general_token_weight"], dtype=torch.float32)
                graph['sentence', 'sentence_general', 'general'].edge_attr = graph['general', 'general_sentence', 'sentence'].edge_attr
        return graph
