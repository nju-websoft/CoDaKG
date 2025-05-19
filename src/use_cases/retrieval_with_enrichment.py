import json
import pandas as pd
import pymysql
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytrec_eval
from typing import List, Dict, Optional
import numpy as np
import os

MYSQL_CONFIG =  {
    "host": "...",
    "port": 3306,
    "user": "...",
    "password": "...",
    "database": "..."
}

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
    
    def retrieval(self, retriever: str, top_k: int = 20) -> Dict[str, Dict[str, float]]:
        if retriever == 'bm25':
            return self.retrieval_bm25(top_k)
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

def generate_queries_and_qrels_acordar():
    acordar_qrel_dir = 'data/acordar'
    qrel_dict = {}
    with open(os.path.join(acordar_qrel_dir, 'qrels.txt'), 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            query_id = items[0]
            node_id = items[2]
            qrel = int(items[3])
            qrel_dict[query_id] = qrel_dict.get(query_id, {})
            qrel_dict[query_id][node_id] = qrel
    queries = {}
    with open(os.path.join(acordar_qrel_dir, 'all_queries.txt'), 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            queries[items[0]] = items[1]
    print(f'queries: {len(queries)}, qrels: {len(qrel_dict)}')
    return queries, qrel_dict


def retrieval_bm25_acordar():
    connection = pymysql.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    eval_metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']

    queries, qrels = generate_queries_and_qrels_acordar()

    id_text_dict = {}
    fields = ['title', 'description', 'author', 'tags']
    sql = f"SELECT dataset_id, {', '.join(fields)} FROM acordar_datasets"
    cursor.execute(sql)
    for item in cursor.fetchall():
        id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item)-1)])
        id_text_dict[item[0]] += item[-1].replace(';', '\t')
    # sql = "SELECT file_id, keyphrase FROM acordar_keyphrase"
    # cursor.execute(sql)
    # for item in cursor.fetchall():
    #     id_text_dict[item[0]] += '\t'.join(json.loads(item[1])[:KEYPHRASE_NUM]) + '\n'

    rtv = Retriever(queries=queries, corpus=id_text_dict)
    run_dict = rtv.retrieval('bm25')
    res = rtv.eval_results(qrels, run_dict, eval_metrics)


def retrieval_bm25_with_enrichment_acordar():
    connection = pymysql.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    eval_metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']

    queries, qrels = generate_queries_and_qrels_acordar()

    id_text_dict = {}

    fields = ['title', 'description', 'tags'] # , 'author'
    sql = f"SELECT dataset_id, {', '.join(fields)} FROM acordar_datasets"
    cursor.execute(sql)
    for item in cursor.fetchall():
        id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item)-1)])
        id_text_dict[item[0]] += item[-1].replace(';', '\t')

    fields = ['label']
    sql = f"SELECT code, {', '.join(fields)} FROM subject_theme"
    cursor.execute(sql)
    theme_dict = {}
    for item in cursor.fetchall():
        theme_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])

    fields = ['label', 'wd_label', 'url']
    # exclude null
    sql = f"SELECT id, {', '.join(fields)} FROM acordar_publisher WHERE LENGTH(CONCAT(label,sameAs,url)) > 0"
    cursor.execute(sql)
    publisher_dict = {}
    for item in cursor.fetchall():
        if item[1] and len(item[1]) > 0:
            publisher_dict[item[0]] = item[1]
        elif item[2] and len(item[2]) > 0:
            publisher_dict = item[2]

    sql = "SELECT dataset_id, publisher_id, subject FROM acordar_metadata_aug"
    cursor.execute(sql)
    for item in cursor.fetchall():
        id_text_dict[item[0]] += '\n' + publisher_dict.get(item[1], ' ') + '\n'
        for theme in json.loads(item[2]):
            id_text_dict[item[0]] += theme_dict[theme]

    rtv = Retriever(queries=queries, corpus=id_text_dict)
    run_dict = rtv.retrieval('bm25')
    res = rtv.eval_results(qrels, run_dict, eval_metrics)


def generate_queries_and_qrels_ntcir():
    with open('code/data/ntcir/queries.json', 'r') as f:
        queries_json = json.load(f)
    queries = queries_json['test']
    qrels = {}
    with open('data/ntcir/data_search_e_test_qrels.txt', 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            query_id = items[0]
            node_id = items[1]
            qrel = int(items[2][-1])
            qrels[query_id] = qrels.get(query_id, {})
            qrels[query_id][node_id] = qrel
    return queries, qrels


def retrieval_bm25_ntcir():
    connection = pymysql.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    eval_metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']

    queries, qrels = generate_queries_and_qrels_ntcir()

    id_text_dict = {}
    fields = ['title', 'description']
    sql = f"SELECT dataset_id, {', '.join(fields)} FROM ntcir_metadata"
    cursor.execute(sql)
    for item in cursor.fetchall():
        id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
    rtv = Retriever(queries=queries, corpus=id_text_dict)
    run_dict = rtv.retrieval('bm25')
    res = rtv.eval_results(qrels, run_dict, eval_metrics)

def retrieval_bm25_with_enrichment_ntcir():
    connection = pymysql.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    eval_metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']

    queries, qrels = generate_queries_and_qrels_ntcir()

    id_text_dict = {}
    fields = ['title', 'description']
    sql = f"SELECT dataset_id, {', '.join(fields)} FROM ntcir_metadata"
    cursor.execute(sql)
    for item in cursor.fetchall():
        id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])

    fields = ['label']
    sql = f"SELECT code, {', '.join(fields)} FROM subject_theme"
    cursor.execute(sql)
    theme_dict = {}
    for item in cursor.fetchall():
        theme_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])

    fields = ['label', 'wd_label', 'url']
    # exclude null
    sql = f"SELECT id, {', '.join(fields)} FROM ntcir_publisher WHERE LENGTH(CONCAT(label,sameAs,url)) > 0"
    cursor.execute(sql)
    publisher_dict = {}
    for item in cursor.fetchall():
        if item[1] and len(item[1]) > 0:
            publisher_dict[item[0]] = item[1]
        elif item[2] and len(item[2]) > 0:
            publisher_dict = item[2]

    sql = "SELECT dataset_id, publisher_id, subject FROM ntcir_metadata_aug"
    cursor.execute(sql)
    for item in cursor.fetchall():
        id_text_dict[item[0]] += '\n' + publisher_dict.get(item[1], ' ') + '\n'
        for theme in json.loads(item[2]):
            id_text_dict[item[0]] += theme_dict[theme]

    rtv = Retriever(queries=queries, corpus=id_text_dict)
    run_dict = rtv.retrieval('bm25')
    res = rtv.eval_results(qrels, run_dict, eval_metrics)


if __name__ == "__main__":
    retrieval_bm25_acordar()
    retrieval_bm25_with_enrichment_acordar()
    retrieval_bm25_ntcir()
    retrieval_bm25_with_enrichment_ntcir()
