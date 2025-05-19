import json
import pandas as pd
import pymysql
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Optional
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytrec_eval
import glob
import shutil

MYSQL_CONFIG =  {
    "host": "...",
    "port": 3306,
    "user": "...",
    "password": "...",
    "database": "..."
}

BERT_MODEL = 'bert-base-uncased'
STELLA_MODEL = 'dunzhang/stella_en_1.5B_v5'
BATCH_SIZE = 512
KEYPHRASE_NUM = 15


class EmbeddingStore:
    def __init__(self, emb_file: str = "embeddings.npy",
                 map_file: str = "id_to_index.pkl"):
        """加载预计算的嵌入"""
        self.embeddings = np.load(emb_file)
        with open(map_file, "rb") as f:
            self.id_to_index = pickle.load(f)

    def __getitem__(self, id_: str) -> np.ndarray:
        """通过ID获取嵌入"""
        return self.embeddings[self.id_to_index[id_]]

    def __contains__(self, id_: str) -> bool:
        """检查ID是否存在"""
        return id_ in self.id_to_index


class Retriever:
    def __init__(self, queries: Dict[str, str], corpus: Dict[str, str]):
        if not queries or not corpus:
            raise ValueError("queries and corpus cannot be empty")
        self.queries = queries
        self.corpus = corpus
        self._bm25 = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self.tokenizer = TfidfVectorizer().build_tokenizer()  # 复用 TF-IDF 的分词器

    def _get_top_results(self, scores: np.ndarray, top_k: int) -> Dict[str, float]:
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return {list(self.corpus.keys())[i]: scores[i] for i in top_indices}

    def retrieval_bm25(self, top_k: int = 20) -> Dict[str, Dict[str, float]]:
        """使用BM25检索"""
        if self._bm25 is None:
            corpus = list(self.corpus.values())
            tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
            self._bm25 = BM25Okapi(tokenized_corpus)

        results = {}
        for query_id, query_text in self.queries.items():
            tokenized_query = self.tokenizer(query_text)
            scores = self._bm25.get_scores(tokenized_query)
            results[query_id] = self._get_top_results(scores, top_k)

        return results

    def retrieval_tfidf(self, top_k: int = 20) -> Dict[str, Dict[str, float]]:
        """使用TF-IDF检索"""
        if self._tfidf_vectorizer is None:
            corpus = list(self.corpus.values())
            self._tfidf_vectorizer = TfidfVectorizer()
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

        results = {}
        query_ids, query_texts = list(self.queries.keys()), list(self.queries.values())
        query_matrix = self._tfidf_vectorizer.transform(query_texts)
        scores = cosine_similarity(query_matrix, self._tfidf_matrix)

        for index, query_id in enumerate(query_ids):
            results[query_id] = self._get_top_results(scores[index], top_k)

        return results

    def retrieval(self, retriever: str, top_k: int = 20) -> Dict[str, Dict[str, float]]:
        if retriever == 'bm25':
            return self.retrieval_bm25(top_k)
        elif retriever == 'tfidf':
            return self.retrieval_tfidf(top_k)
        else:
            raise ValueError(f"Unsupported retriever: {retriever}. Choose 'bm25' or 'tfidf'")

    @staticmethod
    def eval_results(
            qrels_dict: Dict,
            run_dict: Dict,
            metrics: List[str],
            save_path: Optional[str] = None
    ) -> Dict[str, float]:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, metrics)
        eval_results = evaluator.evaluate(run_dict)
        results = {}
        for metric in metrics:
            results[metric] = sum([x[metric] for x in eval_results.values()]) / len(eval_results)

        # Print results
        print("\t".join(f"{metric}: {results[metric]:.4f}" for metric in metrics))

        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)

        return results


def get_bert_embedding(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Get BERT embeddings for a list of texts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = BertModel.from_pretrained(BERT_MODEL).to(device)

    model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize并移至设备
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512  # 避免过长文本
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 使用attention_mask计算有效token的均值
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_stella_embedding(texts: List[str], batch_size: int = 32, normalize: bool = False) -> np.ndarray:
    """
    Get STELLA embeddings for a list of texts using the dunzhang/stella_en_1.5B_v5 model.

    Args:
        texts: List of input texts
        batch_size: Batch size for processing
        normalize: Whether to normalize the embeddings

    Returns:
        numpy.ndarray: Array of shape (num_texts, embedding_dim) containing the embeddings
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(STELLA_MODEL, trust_remote_code=True).to(device)

    model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        with torch.no_grad():
            # STELLA模型内置了批处理和normalize功能
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                device=device
            )

        all_embeddings.append(batch_embeddings)

    return np.concatenate(all_embeddings, axis=0)


def get_node_text(source, node_type):
    connection = pymysql.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    id_text_dict = {}
    if source == 'ntcir':
        if node_type == 'dataset':
            fields = ['title', 'description']
            sql = f"SELECT dataset_id, {', '.join(fields)} FROM ntcir_metadata"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
        elif node_type == 'datafile':
            fields = ['detect_format', 'data_organization', 'data_url']
            sql = f"SELECT file_id, {', '.join(fields)} FROM ntcir_datafile"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
    elif source == 'ntcir_metadata':
        if node_type == 'dataset':
            fields = ['title', 'description']
            sql = f"SELECT dataset_id, {', '.join(fields)} FROM ntcir_metadata"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
        elif node_type == 'datafile':
            fields = ['detect_format', 'data_organization', 'data_url']
            sql = f"SELECT file_id, {', '.join(fields)} FROM ntcir_datafile"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
        elif node_type == 'publisher':
            fields = ['label', 'wd_label', 'wd_description', 'url']
            # exclude null
            sql = f"SELECT id, {', '.join(fields)} FROM ntcir_publisher WHERE LENGTH(CONCAT(label,sameAs,url)) > 0"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item)) if item[i]])
        elif node_type == 'theme':
            fields = ['label', 'definition']
            sql = f"SELECT code, {', '.join(fields)} FROM subject_theme"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
    elif source == 'ntcir_metadata_content':
        if node_type == 'dataset':
            fields = ['title', 'description']
            sql = f"SELECT dataset_id, {', '.join(fields)} FROM ntcir_metadata"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
        elif node_type == 'datafile':
            sql = "SELECT file_id, keyphrase FROM ntcir_keyphrase"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[int(item[0])] = '\t'.join(json.loads(item[1])[:KEYPHRASE_NUM]) + '\n'
            fields = ['detect_format', 'data_organization', 'data_url']
            sql = f"SELECT file_id, {', '.join(fields)} FROM ntcir_datafile"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = id_text_dict.get(item[0], '') + '\n'.join(
                    [str(item[i]) for i in range(1, len(item))])
        elif node_type == 'publisher':
            fields = ['label', 'wd_label', 'wd_description', 'url']
            # exclude null
            sql = f"SELECT id, {', '.join(fields)} FROM ntcir_publisher WHERE LENGTH(CONCAT(label,sameAs,url)) > 0"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item)) if item[i]])
        elif node_type == 'theme':
            fields = ['label', 'definition']
            sql = f"SELECT code, {', '.join(fields)} FROM subject_theme"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
    elif source == 'acordar':
        if node_type == 'dataset':
            fields = ['title', 'description', 'author', 'tags']
            sql = f"SELECT dataset_id, {', '.join(fields)} FROM acordar_datasets"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item) - 1)])
                id_text_dict[item[0]] += item[-1].replace(';', '\t')
        elif node_type == 'datafile':
            sql = f"SELECT dataset_id, download FROM acordar_datasets"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join(eval(item[1]) + ['rdf'])
    elif source == 'acordar_metadata':
        if node_type == 'dataset':
            fields = ['title', 'description', 'author', 'tags']
            sql = f"SELECT dataset_id, {', '.join(fields)} FROM acordar_datasets"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item) - 1)])
                id_text_dict[item[0]] += item[-1].replace(';', '\t')
        elif node_type == 'datafile':
            sql = f"SELECT dataset_id, download FROM acordar_datasets"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join(eval(item[1]) + ['rdf'])
        elif node_type == 'publisher':
            fields = ['label', 'wd_label', 'wd_description', 'url']
            # exclude null
            sql = f"SELECT id, {', '.join(fields)} FROM acordar_publisher WHERE LENGTH(CONCAT(label,sameAs,url)) > 0"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item)) if item[i]])
        elif node_type == 'theme':
            fields = ['label', 'definition']
            sql = f"SELECT code, {', '.join(fields)} FROM subject_theme"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
    elif source == 'acordar_metadata_content':
        if node_type == 'dataset':
            fields = ['title', 'description', 'author', 'tags']
            sql = f"SELECT dataset_id, {', '.join(fields)} FROM acordar_datasets"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item) - 1)])
                id_text_dict[item[0]] += item[-1].replace(';', '\t')
        elif node_type == 'datafile':
            sql = "SELECT file_id, keyphrase FROM acordar_keyphrase"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\t'.join(json.loads(item[1])[:KEYPHRASE_NUM]) + '\n'
            sql = f"SELECT dataset_id, download FROM acordar_datasets"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = id_text_dict.get(item[0], '') + '\n'.join(eval(item[1]) + ['rdf'])
        elif node_type == 'publisher':
            fields = ['label', 'wd_label', 'wd_description', 'url']
            # exclude null
            sql = f"SELECT id, {', '.join(fields)} FROM acordar_publisher WHERE LENGTH(CONCAT(label,sameAs,url)) > 0"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item)) if item[i]])
        elif node_type == 'theme':
            fields = ['label', 'definition']
            sql = f"SELECT code, {', '.join(fields)} FROM subject_theme"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
    return id_text_dict


def generate_node_embeddings(source, node_type, output_dir='output/'):
    # embedding_fun = get_bert_embedding
    embedding_fun = get_stella_embedding

    id_text_dict = get_node_text(source, node_type)
    node_texts = list(id_text_dict.values())
    node_ids = list(id_text_dict.keys())
    node_embeddings = embedding_fun(node_texts, BATCH_SIZE)
    id_to_index = {id_: idx for idx, id_ in enumerate(node_ids)}
    np.save(os.path.join(output_dir, f'{source}_{node_type}_bert.npy'), node_embeddings)
    with open(os.path.join(output_dir, f'{source}_{node_type}_id_to_index.pkl'), 'wb') as f:
        pickle.dump(id_to_index, f)
    return node_ids


def generate_node_data(source, node_types, state_pkl, output_dir='output/'):
    # 初始化state（如果文件存在则加载，否则创建新的）
    state = {
        'node_base_id': 0,
        'node_type_base_id': 0,
        'node_type_map': {},
        'node_id_map': {}
    }

    if os.path.exists(os.path.join(output_dir, state_pkl)):
        with open(os.path.join(output_dir, state_pkl), 'rb') as f:
            saved_state = pickle.load(f)
            # 只恢复计数器的值，不覆盖映射关系（因为每次node_types可能不同）
            state['node_base_id'] = saved_state['node_base_id']
            state['node_type_base_id'] = saved_state['node_type_base_id']

    row_data = []
    current_node_id = state['node_base_id']  # 当前批次起始ID
    current_type_id = state['node_type_base_id']

    for node_type in node_types:
        # 为当前node_type分配类型ID（如果是新类型则递增）
        if node_type not in state['node_type_map']:
            state['node_type_map'][node_type] = current_type_id
            current_type_id += 1

        # 处理节点数据
        node_ids = generate_node_embeddings(source, node_type, output_dir)
        embeddings = EmbeddingStore(
            emb_file=os.path.join(output_dir, f'{source}_{node_type}_bert.npy'),
            map_file=os.path.join(output_dir, f'{source}_{node_type}_id_to_index.pkl')
        )

        state['node_id_map'][node_type] = {}
        for node_idx, node_id in enumerate(node_ids):
            state['node_id_map'][node_type][node_id] = current_node_id + node_idx
            node_embedding = embeddings[node_id]
            row_data.append((
                current_node_id + node_idx,
                node_id,
                state['node_type_map'][node_type],
                ",".join([f"{x:.6f}" for x in node_embedding.squeeze()])
            ))

        current_node_id += len(node_ids)

    # 更新全局计数器
    state['node_base_id'] = current_node_id
    state['node_type_base_id'] = current_type_id

    print(f'node num: {len(row_data)}')
    os.makedirs(os.path.join(output_dir, source), exist_ok=True)
    with open(os.path.join(output_dir, source, 'node.dat'), 'w') as f:
        for row in row_data:
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")

    with open(os.path.join(output_dir, state_pkl), 'wb') as f:
        pickle.dump(state, f)


def generate_sim_pairs(csv_file, state, type_id, threshold=1.0, weight=1.0):
    """
    从CSV文件生成相似对列表

    参数:
        csv_file: CSV文件路径
        state: 包含节点ID映射的字典 {原始ID: 映射ID}
        type_id: 预定义的类型ID
        threshold: 相似度阈值(默认0.8)

    返回:
        list: 格式为 [(state[id1], state[id2], type_id, 1.0), ...] 的列表
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file, dtype={'id1': str, 'id2': str, 'j_sim': float})

    # 筛选相似度大于threshold的行
    high_sim_df = df[df['j_sim'] >= threshold]

    # 生成结果列表
    result = []
    for _, row in high_sim_df.iterrows():
        id1 = state['node_id_map']['datafile'][row['id1']]
        id2 = state['node_id_map']['datafile'][row['id2']]

        if id1 == id2:
            continue

        # 添加双向关系
        result.append((id1, id2, type_id, weight))
        result.append((id2, id1, type_id, weight))

    return result


def generate_link_data(source, link_types, state_pkl, output_dir='output/'):
    if os.path.exists(os.path.join(output_dir, state_pkl)):
        with open(os.path.join(output_dir, state_pkl), 'rb') as f:
            state = pickle.load(f)
    else:
        print('error! state file not found')
        return

    link_type_base_id = state.get('link_type_base_id', 0)
    state['link_type_map'] = state.get('link_type_map', {})
    type_data_dict = {}

    connection = pymysql.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()

    if source == 'ntcir':
        for link_type in link_types:
            if link_type == 'metadata':  # google, dataset -> dataset
                sql = 'SELECT dataset_id1, dataset_id2, relationship FROM ntcir_relationship'
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id1 = state['node_id_map']['dataset'][item[0]]
                    dataset_id2 = state['node_id_map']['dataset'][item[1]]
                    if item[2] not in state['link_type_map']:
                        state['link_type_map'][item[2]] = link_type_base_id
                        link_type_base_id += 1
                    type_id = state['link_type_map'][item[2]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id1, dataset_id2, type_id, 1.0))
                    type_data_dict.setdefault(type_id, []).append((dataset_id2, dataset_id1, type_id, 1.0))
            elif link_type == 'dump':  # dataset -> datafile
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, file_id FROM ntcir_datafile"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    file_id = state['node_id_map']['datafile'][item[1]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id, file_id, type_id, 1.0))
        state['link_type_base_id'] = link_type_base_id
    elif source == 'ntcir_metadata':
        for link_type in link_types:
            if link_type == 'metadata':  # google, dataset -> dataset
                sql = 'SELECT dataset_id1, dataset_id2, relationship FROM ntcir_relationship'
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id1 = state['node_id_map']['dataset'][item[0]]
                    dataset_id2 = state['node_id_map']['dataset'][item[1]]
                    if item[2] not in state['link_type_map']:
                        state['link_type_map'][item[2]] = link_type_base_id
                        link_type_base_id += 1
                    type_id = state['link_type_map'][item[2]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id1, dataset_id2, type_id, 1.0))
                    type_data_dict.setdefault(type_id, []).append((dataset_id2, dataset_id1, type_id, 1.0))
            elif link_type == 'dump':  # dataset -> datafile
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, file_id FROM ntcir_datafile"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    file_id = state['node_id_map']['datafile'][item[1]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id, file_id, type_id, 1.0))
            elif link_type == 'publish':
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, publisher_id FROM ntcir_metadata_aug"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    if item[1] in state['node_id_map']['publisher']:
                        publisher_id = state['node_id_map']['publisher'][item[1]]
                        type_data_dict.setdefault(type_id, []).append((dataset_id, publisher_id, type_id, 1.0))
            elif link_type == 'theme':
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, `subject` FROM ntcir_metadata_aug"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    themes = json.loads(item[1])
                    for theme in themes:
                        theme_code = state['node_id_map']['theme'][theme]
                        type_data_dict.setdefault(type_id, []).append((dataset_id, theme_code, type_id, 1.0))
        state['link_type_base_id'] = link_type_base_id
    elif source == 'ntcir_metadata_content':
        for link_type in link_types:
            if link_type == 'metadata':  # google, dataset -> dataset
                sql = 'SELECT dataset_id1, dataset_id2, relationship FROM ntcir_relationship'
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id1 = state['node_id_map']['dataset'][item[0]]
                    dataset_id2 = state['node_id_map']['dataset'][item[1]]
                    if item[2] not in state['link_type_map']:
                        state['link_type_map'][item[2]] = link_type_base_id
                        link_type_base_id += 1
                    type_id = state['link_type_map'][item[2]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id1, dataset_id2, type_id, 1.0))
                    type_data_dict.setdefault(type_id, []).append((dataset_id2, dataset_id1, type_id, 1.0))
            elif link_type == 'dump':  # dataset -> datafile
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, file_id FROM ntcir_datafile"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    file_id = state['node_id_map']['datafile'][item[1]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id, file_id, type_id, 1.0))
            elif link_type == 'publish':
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, publisher_id FROM ntcir_metadata_aug"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    if item[1] in state['node_id_map']['publisher']:
                        publisher_id = state['node_id_map']['publisher'][item[1]]
                        type_data_dict.setdefault(type_id, []).append((dataset_id, publisher_id, type_id, 1.0))
            elif link_type == 'theme':
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, `subject` FROM ntcir_metadata_aug"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    themes = json.loads(item[1])
                    for theme in themes:
                        theme_code = state['node_id_map']['theme'][theme]
                        type_data_dict.setdefault(type_id, []).append((dataset_id, theme_code, type_id, 1.0))
            elif link_type == 'overlap':
                lt = 'content_sim'  # data(content) overlap
                if lt not in state['link_type_map']:
                    state['link_type_map'][lt] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][lt]
                for fmt in ['text', 'table', 'json_xml']:
                    csv_file = f'output/similarity/{fmt}_similarity_results.csv'
                    link_res = generate_sim_pairs(csv_file, state, type_id)
                    type_data_dict.setdefault(type_id, []).extend(link_res)
                lt = 'pattern_sim'  # patttern overlap
                if lt not in state['link_type_map']:
                    state['link_type_map'][lt] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][lt]
                for fmt in ['table', 'json_xml']:
                    csv_file = f'output/similarity/{fmt}_pattern_similarity_results.csv'
                    link_res = generate_sim_pairs(csv_file, state, type_id)
                    type_data_dict.setdefault(type_id, []).extend(link_res)
                lt = 'keyphrase_sim'  # keyword(keyphrase) overlap
                if lt not in state['link_type_map']:
                    state['link_type_map'][lt] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][lt]
                csv_file = 'output/similarity/ntcir_keyphrase_15_similarity_results.csv'
                link_res = generate_sim_pairs(csv_file, state, type_id)
                type_data_dict.setdefault(type_id, []).extend(link_res)
        state['link_type_base_id'] = link_type_base_id
    elif source == 'acordar':
        for link_type in link_types:
            if link_type == 'metadata':  # google, dataset -> dataset
                sql = 'SELECT dataset_id1, dataset_id2, relationship FROM acordar_relationship'
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id1 = state['node_id_map']['dataset'][item[0]]
                    dataset_id2 = state['node_id_map']['dataset'][item[1]]
                    if item[2] not in state['link_type_map']:
                        state['link_type_map'][item[2]] = link_type_base_id
                        link_type_base_id += 1
                    type_id = state['link_type_map'][item[2]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id1, dataset_id2, type_id, 1.0))
                    type_data_dict.setdefault(type_id, []).append((dataset_id2, dataset_id1, type_id, 1.0))
            elif link_type == 'dump':  # dataset -> datafile
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, dataset_id FROM acordar_datasets"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    file_id = state['node_id_map']['datafile'][item[1]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id, file_id, type_id, 1.0))
        state['link_type_base_id'] = link_type_base_id
    elif source == 'acordar_metadata':
        for link_type in link_types:
            if link_type == 'metadata':  # google, dataset -> dataset
                sql = 'SELECT dataset_id1, dataset_id2, relationship FROM acordar_relationship'
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id1 = state['node_id_map']['dataset'][item[0]]
                    dataset_id2 = state['node_id_map']['dataset'][item[1]]
                    if item[2] not in state['link_type_map']:
                        state['link_type_map'][item[2]] = link_type_base_id
                        link_type_base_id += 1
                    type_id = state['link_type_map'][item[2]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id1, dataset_id2, type_id, 1.0))
                    type_data_dict.setdefault(type_id, []).append((dataset_id2, dataset_id1, type_id, 1.0))
            elif link_type == 'dump':  # dataset -> datafile
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, dataset_id FROM acordar_datasets"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    file_id = state['node_id_map']['datafile'][item[1]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id, file_id, type_id, 1.0))
            elif link_type == 'publish':
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, publisher_id FROM acordar_metadata_aug"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    if item[1] in state['node_id_map']['publisher']:
                        publisher_id = state['node_id_map']['publisher'][item[1]]
                        type_data_dict.setdefault(type_id, []).append((dataset_id, publisher_id, type_id, 1.0))
            elif link_type == 'theme':
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, `subject` FROM acordar_metadata_aug"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    themes = json.loads(item[1])
                    for theme in themes:
                        theme_code = state['node_id_map']['theme'][theme]
                        type_data_dict.setdefault(type_id, []).append((dataset_id, theme_code, type_id, 1.0))
        state['link_type_base_id'] = link_type_base_id
    elif source == 'acordar_metadata_content':
        for link_type in link_types:
            if link_type == 'metadata':  # google, dataset -> dataset
                sql = 'SELECT dataset_id1, dataset_id2, relationship FROM acordar_relationship'
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id1 = state['node_id_map']['dataset'][item[0]]
                    dataset_id2 = state['node_id_map']['dataset'][item[1]]
                    if item[2] not in state['link_type_map']:
                        state['link_type_map'][item[2]] = link_type_base_id
                        link_type_base_id += 1
                    type_id = state['link_type_map'][item[2]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id1, dataset_id2, type_id, 1.0))
                    type_data_dict.setdefault(type_id, []).append((dataset_id2, dataset_id1, type_id, 1.0))
            elif link_type == 'dump':  # dataset -> datafile
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, dataset_id FROM acordar_datasets"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    file_id = state['node_id_map']['datafile'][item[1]]
                    type_data_dict.setdefault(type_id, []).append((dataset_id, file_id, type_id, 1.0))
            elif link_type == 'publish':
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, publisher_id FROM acordar_metadata_aug"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    if item[1] in state['node_id_map']['publisher']:
                        publisher_id = state['node_id_map']['publisher'][item[1]]
                        type_data_dict.setdefault(type_id, []).append((dataset_id, publisher_id, type_id, 1.0))
            elif link_type == 'theme':
                if link_type not in state['link_type_map']:
                    state['link_type_map'][link_type] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][link_type]
                sql = "SELECT dataset_id, `subject` FROM acordar_metadata_aug"
                cursor.execute(sql)
                for item in cursor.fetchall():
                    dataset_id = state['node_id_map']['dataset'][item[0]]
                    themes = json.loads(item[1])
                    for theme in themes:
                        theme_code = state['node_id_map']['theme'][theme]
                        type_data_dict.setdefault(type_id, []).append((dataset_id, theme_code, type_id, 1.0))
            elif link_type == 'overlap':
                lt = 'content_sim'  # data(content) overlap
                if lt not in state['link_type_map']:
                    state['link_type_map'][lt] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][lt]
                for fmt in ['rdf']:
                    csv_file = f'output/similarity/{fmt}_similarity_results.csv'
                    link_res = generate_sim_pairs(csv_file, state, type_id)
                    type_data_dict.setdefault(type_id, []).extend(link_res)
                lt = 'pattern_sim'  # patttern overlap
                if lt not in state['link_type_map']:
                    state['link_type_map'][lt] = link_type_base_id
                    link_type_base_id += 1
                type_id = state['link_type_map'][lt]
                for fmt in ['rdf']:
                    csv_file = f'output/similarity/{fmt}_pattern_similarity_results.csv'
                    link_res = generate_sim_pairs(csv_file, state, type_id)
                    type_data_dict.setdefault(type_id, []).extend(link_res)
                # lt = 'keyphrase_sim'  # keyword(keyphrase) overlap
                # if lt not in state['link_type_map']:
                #     state['link_type_map'][lt] = link_type_base_id
                #     link_type_base_id += 1
                # type_id = state['link_type_map'][lt]
                # csv_file = 'output/similarity/acordar_keyphrase_15_similarity_results.csv'
                # link_res = generate_sim_pairs(csv_file, state, type_id)
                # type_data_dict.setdefault(type_id, []).extend(link_res)
        state['link_type_base_id'] = link_type_base_id

    with open(os.path.join(output_dir, state_pkl), 'wb') as f:
        pickle.dump(state, f)

    cnt = 0
    with open(os.path.join(output_dir, source, 'link.dat'), 'w') as f:
        for type_id in sorted(type_data_dict.keys()):
            for row in type_data_dict[type_id]:
                f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")
                cnt += 1
    print(f'link num: {cnt}')


def generate_corpus(source, state_pkl, output_dir='output/'):
    if os.path.exists(os.path.join(output_dir, state_pkl)):
        with open(os.path.join(output_dir, state_pkl), 'rb') as f:
            state = pickle.load(f)
    else:
        print('error! state file not found')
        return

    corpus_dict = {}
    if source in ['ntcir', 'acordar', 'ntcir_metadata', 'ntcir_metadata_content', 'acordar_metadata',
                  'acordar_metadata_content']:
        for node_type in state['node_type_map'].keys():
            id_text_dict = get_node_text(source, node_type)
            corpus_dict.update({str(state['node_id_map'][node_type][k]): v for k, v in id_text_dict.items()})

    with open(os.path.join(output_dir, source, 'corpus.json'), 'w') as f:
        json.dump(corpus_dict, f, indent=2)


def split_ntcir_train_test():  # ntcir-15
    ntcir_qrel_dir = '../data/ntcir'
    save_dir = 'data/ntcir'

    all_train_queries = []
    with open(os.path.join(ntcir_qrel_dir, 'data_search_e_train_topics.tsv'), 'r') as f:
        for line in f:
            all_train_queries.append(line.strip().split('\t'))
    train_queries, val_queries = train_test_split(all_train_queries, test_size=0.2, random_state=42)

    test_queries = []
    with open(os.path.join(ntcir_qrel_dir, 'data_search_e_test_topics.tsv'), 'r') as f:
        for line in f:
            test_queries.append(line.strip().split('\t'))

    queries = {
        'train': {item[0]: item[1] for item in train_queries},
        'val': {item[0]: item[1] for item in val_queries},
        'test': {item[0]: item[1] for item in test_queries}
    }
    with open(os.path.join(save_dir, 'queries.json'), 'w') as f:
        json.dump(queries, f, indent=2)


def generate_queries_and_qrels(source, state_pkl, output_dir='output/'):
    if os.path.exists(os.path.join(output_dir, state_pkl)):
        with open(os.path.join(output_dir, state_pkl), 'rb') as f:
            state = pickle.load(f)
    else:
        print('error! state file not found')
        return

    ntcir_qrel_dir = '../data/ntcir'
    qrels = {'train': {}, 'val': {}, 'test': {}}
    if source in ['ntcir', 'ntcir_metadata', 'ntcir_metadata_content']:
        with open('data/ntcir/queries.json', 'r') as f:
            queries_json = json.load(f)
        queries = [(k, v) for q_dict in queries_json.values() for k, v in q_dict.items()]
        with open(os.path.join(output_dir, source, 'queries.tsv'), 'w') as f:
            for item in queries:
                f.write(f"{item[0]}\t{item[1]}\n")

        with open(os.path.join(ntcir_qrel_dir, 'data_search_e_train_qrels.txt'), 'r') as f:
            for line in f:
                items = line.strip().split(' ')
                query_id = items[0]
                node_id = state['node_id_map']['dataset'][items[1]]
                qrel = int(items[2][-1])
                if query_id in queries_json['train']:
                    qrels['train'][query_id] = qrels['train'].get(query_id, {})
                    qrels['train'][query_id][node_id] = qrel
                elif query_id in queries_json['val']:
                    qrels['val'][query_id] = qrels['val'].get(query_id, {})
                    qrels['val'][query_id][node_id] = qrel
        with open(os.path.join(ntcir_qrel_dir, 'data_search_e_test_qrels.txt'), 'r') as f:
            for line in f:
                items = line.strip().split(' ')
                query_id = items[0]
                node_id = state['node_id_map']['dataset'][items[1]]
                qrel = int(items[2][-1])
                qrels['test'][query_id] = qrels['test'].get(query_id, {})
                qrels['test'][query_id][node_id] = qrel
    elif source in ['acordar', 'acordar_metadata', 'acordar_metadata_content']:
        shutil.copyfile('../data/acordar/queries.txt', os.path.join(output_dir, source, 'queries.tsv'))
        acordar_qrel_dir = '../data/acordar/Splits for Cross Validation'
        for sub_dir in os.listdir(acordar_qrel_dir):
            os.makedirs(os.path.join(output_dir, source, sub_dir), exist_ok=True)
            for qrel_file in os.listdir(os.path.join(acordar_qrel_dir, sub_dir)):
                qrel_dict = {}
                with open(os.path.join(acordar_qrel_dir, sub_dir, qrel_file), 'r') as f:
                    for line in f:
                        items = line.strip().split('\t')
                        query_id = items[0]
                        node_id = state['node_id_map']['dataset'][items[2]]
                        qrel = int(items[3])
                        qrel_dict[query_id] = qrel_dict.get(query_id, {})
                        qrel_dict[query_id][node_id] = qrel
                filename = 'val.json' if qrel_file == 'valid.txt' else os.path.splitext(qrel_file)[0] + '.json'
                with open(os.path.join(output_dir, source, sub_dir, filename), 'w') as f:
                    json.dump(qrel_dict, f, indent=2)
        return

    with open(os.path.join(output_dir, source, 'train.json'), 'w') as f:
        json.dump(qrels['train'], f, indent=2)
    with open(os.path.join(output_dir, source, 'val.json'), 'w') as f:
        json.dump(qrels['val'], f, indent=2)
    with open(os.path.join(output_dir, source, 'test.json'), 'w') as f:
        json.dump(qrels['test'], f, indent=2)


def generate_retrieval_results(source, retrieval_methods, state_pkl, output_dir='output/', top_k=20):
    if os.path.exists(os.path.join(output_dir, state_pkl)):
        with open(os.path.join(output_dir, state_pkl), 'rb') as f:
            state = pickle.load(f)
    else:
        print('error! state file not found')
        return

    eval_metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    queries = {}
    with open(os.path.join(output_dir, source, 'queries.tsv'), 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            queries[items[0]] = items[1]
    with open(os.path.join(output_dir, source, 'corpus.json'), 'r') as f:
        corpus = json.load(f)
    corpus = {k: v for k, v in corpus.items() if int(k) in state['node_id_map']['dataset'].values()}
    if source in ['ntcir', 'ntcir_metadata', 'ntcir_metadata_content']:
        with open(os.path.join(output_dir, source, 'test.json'), 'r') as f:
            test_qrels = json.load(f)

        retriever = Retriever(queries, corpus)
        for method in retrieval_methods:
            retrieval_results = retriever.retrieval(method, top_k)
            with open(os.path.join(output_dir, source, f'{method}.json'), 'w') as f:
                json.dump(retrieval_results, f, indent=2)
            run_dict = {k: v for k, v in retrieval_results.items() if k in test_qrels.keys()}
            print(f'[{method}]')
            retriever.eval_results(test_qrels, run_dict, eval_metrics)
    elif source in ['acordar', 'acordar_metadata', 'acordar_metadata_content']:
        all_qrels, all_run_dict = {}, {method: {} for method in retrieval_methods}
        retriever = Retriever({'-1': 'temp'}, corpus)
        for sub_dir in os.listdir(os.path.join(output_dir, source)):
            if os.path.isdir(os.path.join(output_dir, source, sub_dir)):
                print(f'[{sub_dir}]')
                with open(os.path.join(output_dir, source, sub_dir, 'test.json'), 'r') as f:
                    test_qrels = json.load(f)
                all_qrels.update(test_qrels)
                sub_queries = {k: v for k, v in queries.items() if k in test_qrels}
                retriever.queries = sub_queries
                for method in retrieval_methods:
                    retrieval_results = retriever.retrieval(method, top_k)
                    with open(os.path.join(output_dir, source, sub_dir, f'{method}.json'), 'w') as f:
                        json.dump(retrieval_results, f, indent=2)
                    run_dict = {k: v for k, v in retrieval_results.items() if k in test_qrels.keys()}
                    all_run_dict[method].update(run_dict)
                    print(f'[{method}]')
                    retriever.eval_results(test_qrels, run_dict, eval_metrics)
        for method in retrieval_methods:
            print(f'[{method} (all)]')
            retriever.eval_results(all_qrels, all_run_dict[method], eval_metrics)


if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    # generate_node_data(source='ntcir', node_types=['dataset', 'datafile'], state_pkl='ntcir.state.pkl', output_dir='output/')
    # generate_link_data(source='ntcir', link_types=['dump', 'metadata'], state_pkl='ntcir.state.pkl', output_dir='output/')
    # generate_corpus(source='ntcir', state_pkl='ntcir.state.pkl', output_dir='output/')
    # split_ntcir_train_test()
    # generate_queries_and_qrels(source='ntcir', state_pkl='ntcir.state.pkl', output_dir='output/')
    # generate_retrieval_results(source='ntcir',
    #                            retrieval_methods=['bm25', 'tfidf'],
    #                            state_pkl='ntcir.state.pkl', output_dir='output/', top_k=20)

    # generate_node_data(source='acordar', node_types=['dataset', 'datafile'], state_pkl='acordar.state.pkl', output_dir='output/')
    # generate_link_data(source='acordar', link_types=['dump', 'metadata'], state_pkl='acordar.state.pkl', output_dir='output/')
    # generate_corpus(source='acordar', state_pkl='acordar.state.pkl', output_dir='output/')
    # generate_queries_and_qrels(source='acordar', state_pkl='acordar.state.pkl', output_dir='output/')
    # generate_retrieval_results(source='acordar',
    #                            retrieval_methods=['bm25', 'tfidf'],
    #                            state_pkl='acordar.state.pkl', output_dir='output/', top_k=20)

    # generate_node_data(source='ntcir_metadata', node_types=['dataset', 'datafile', 'publisher', 'theme'], state_pkl='ntcir_metadata.state.pkl', output_dir='output/')
    # generate_link_data(source='ntcir_metadata', link_types=['dump', 'metadata', 'publish', 'theme'], state_pkl='ntcir_metadata.state.pkl', output_dir='output/')
    # generate_corpus(source='ntcir_metadata', state_pkl='ntcir_metadata.state.pkl', output_dir='output/')
    # generate_queries_and_qrels(source='ntcir_metadata', state_pkl='ntcir_metadata.state.pkl', output_dir='output/')
    # generate_retrieval_results(source='ntcir_metadata',
    #                            retrieval_methods=['bm25', 'tfidf'],
    #                            state_pkl='ntcir_metadata.state.pkl', output_dir='output/', top_k=20)

    generate_node_data(source='ntcir_metadata_content', node_types=['dataset', 'datafile', 'publisher', 'theme'], state_pkl='ntcir_metadata_content.state.pkl', output_dir='output/')
    generate_link_data(source='acordar_metadata_content', link_types=['dump', 'metadata', 'publish', 'theme', 'overlap'], state_pkl='acordar_metadata_content.state.pkl', output_dir='output/')
    generate_corpus(source='ntcir_metadata_content', state_pkl='ntcir_metadata_content.state.pkl', output_dir='output/')
    generate_queries_and_qrels(source='ntcir_metadata_content', state_pkl='ntcir_metadata_content.state.pkl', output_dir='output/')
    generate_retrieval_results(source='ntcir_metadata_content',
                               retrieval_methods=['bm25', 'tfidf'],
                               state_pkl='ntcir_metadata_content.state.pkl', output_dir='output/', top_k=20)
    pass
