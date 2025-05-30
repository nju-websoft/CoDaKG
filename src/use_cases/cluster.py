from collections import defaultdict
import json
from graph_tool.all import Graph, load_graph
import graph_tool.all as gt
import pickle
import os
import numpy as np
import pymysql
import pandas as pd
import matplotlib as mpl


MYSQL_CONFIG = {
    "host": "...",
    "user": "...",
    "password": "...",
    "database": "...",
    "charset": "utf8mb4"
}

class TypedGraphBuilder:
    def __init__(self):
        self.graph = Graph(directed=True)
        self.node_type_map = {}  # (type, local_id) -> global_id (global_id here is int vertex index)
        self.node_reverse_map = {}  # global_id -> (type, local_id)
        # 修改：节点type属性存储整数
        self.vertex_prop_type = self.graph.new_vertex_property("int")
        # local_id 保持字符串，因为它可能不是整数
        self.vertex_prop_local_id = self.graph.new_vertex_property("string")
        # 修改：边type属性存储整数
        self.edge_type = self.graph.new_edge_property("int")
        self.edge_weight = self.graph.new_edge_property("float")

        # 新增：字符串类型到整数ID的映射
        self.node_type_to_int = {}
        self.edge_type_to_int = {}
        self._next_node_type_int = 0
        self._next_edge_type_int = 0

        self.type_id_to_vertex = {} # (type, local_id) -> graph_tool vertex object

    def _get_node_type_int(self, node_type_str: str) -> int:
        """获取或创建节点类型的整数ID"""
        if node_type_str not in self.node_type_to_int:
            self.node_type_to_int[node_type_str] = self._next_node_type_int
            self._next_node_type_int += 1
        return self.node_type_to_int[node_type_str]

    def _get_edge_type_int(self, edge_type_str: str) -> int:
        """获取或创建边类型的整数ID"""
        if edge_type_str not in self.edge_type_to_int:
            self.edge_type_to_int[edge_type_str] = self._next_edge_type_int
            self._next_edge_type_int += 1
        return self.edge_type_to_int[edge_type_str]

    def add_nodes(self, node_type: str, local_ids: list):
        # 获取节点类型的整数ID
        node_type_int = self._get_node_type_int(node_type)

        for local_id in local_ids:
            key = (node_type, local_id)
            if key not in self.type_id_to_vertex: # 检查是否已存在
                v = self.graph.add_vertex()
                global_id = int(v) # graph_tool vertex can be cast to int ID

                # 使用原始字符串作为key，整数ID作为vertex属性
                self.node_type_map[key] = global_id
                self.node_reverse_map[global_id] = key

                # 存储整数类型的node_type
                self.vertex_prop_type[v] = node_type_int
                self.vertex_prop_local_id[v] = str(local_id) # local_id仍为字符串

                self.type_id_to_vertex[key] = v
    
    def build_nodes(self, source):
        print('build nodes...')
        connection = pymysql.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()

        node_type = 'dataset'
        if source == 'ntcir':
            sql = "SELECT dataset_id FROM ntcir_metadata"
        elif source == 'acordar':
            sql = "SELECT dataset_id FROM acordar_datasets"
        cursor.execute(sql)
        local_ids = [str(x[0]) for x in cursor.fetchall()]
        self.add_nodes(node_type, local_ids)
        
        node_type = 'datafile'
        if source == 'ntcir':
            sql = "SELECT file_id FROM ntcir_datafile"
        elif source == 'acordar':
            sql = "SELECT dataset_id FROM acordar_datasets"
        cursor.execute(sql)
        local_ids = [str(x[0]) for x in cursor.fetchall()]
        self.add_nodes(node_type, local_ids)

        node_type = 'publisher'
        sql = f"SELECT id FROM {source}_publisher WHERE label != ''"
        cursor.execute(sql)
        local_ids = [str(x[0]) for x in cursor.fetchall()]
        self.add_nodes(node_type, local_ids)

        node_type = 'theme'
        sql = "SELECT code FROM subject_theme"
        cursor.execute(sql)
        local_ids = [str(x[0]) for x in cursor.fetchall()]
        self.add_nodes(node_type, local_ids)

    def add_edges(self, edge_list: list, src_type: str, dst_type: str):
        src_type_int = self._get_node_type_int(src_type) # 确保节点类型也已记录
        dst_type_int = self._get_node_type_int(dst_type) # 确保节点类型也已记录

        for src_id, dst_id, edge_type_str, weight in edge_list:
            src_key = (src_type, src_id)
            dst_key = (dst_type, dst_id)

            # 检查源和目标节点是否存在
            if src_key not in self.type_id_to_vertex or dst_key not in self.type_id_to_vertex:
                # 可以选择打印警告或者跳过，这里选择跳过
                # print(f"Warning: Source or destination node not found for edge {src_key} -> {dst_key}")
                continue

            src_v = self.type_id_to_vertex[src_key]
            dst_v = self.type_id_to_vertex[dst_key]

            # 获取边类型的整数ID
            edge_type_int = self._get_edge_type_int(edge_type_str)

            e = self.graph.add_edge(src_v, dst_v)
            # 存储整数类型的edge_type
            self.edge_type[e] = edge_type_int
            self.edge_weight[e] = weight

    def build_edges(self, source, weights=defaultdict(lambda: 1.0), thresholds=defaultdict(lambda: 0.8), row_cnt_limit=1000):
        print('build edges...')
        connection = pymysql.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()

        edge_type, src_type, dst_type = 'metadata', 'dataset', 'dataset'
        edges = []
        sql = f'SELECT dataset_id1, dataset_id2, relationship FROM {source}_relationship'
        cursor.execute(sql)
        for item in cursor.fetchall():
            dataset_id1, dataset_id2, relationship = item
            edges.append((dataset_id1, dataset_id2, relationship, weights[edge_type]))
            edges.append((dataset_id2, dataset_id1, relationship, weights[edge_type]))
        self.add_edges(edges, src_type, dst_type)
        # id1_list = []
        # id2_list = []
        # relationship_list = []
        # lst = cursor.fetchall()
        # for item in lst:
        #     dataset_id1, dataset_id2, relationship = item
        #     id1_list.append(min(dataset_id1, dataset_id2))
        #     id2_list.append(max(dataset_id1, dataset_id2))
        #     relationship_list.append(relationship)
        # main_df = pd.DataFrame({
        #     'id1': id1_list,
        #     'id2': id2_list,
        #     'relationship': relationship_list,
        #     'j_sim': [1.0] * len(lst)
        # })
        # grouped = main_df.groupby('relationship')
        # result_dict = {
        #     name: group[['id1', 'id2', 'j_sim']].astype({'id1': str, 'id2': str, 'j_sim': float}).reset_index(drop=True)
        #     for name, group in grouped
        # }
        # for relationship, all_data in result_dict.items():
        #     all_data_flitered = self.filter_by_row_cnt(all_data, row_cnt_limit)
        #     edge_type = relationship
        #     link_res = self.generate_sim_pairs(all_data_flitered, edge_type, 0, weights[edge_type])
        #     edges.extend(link_res)
        # self.add_edges(edges, src_type, dst_type)

        edges = []
        edge_type, src_type, dst_type = 'dump', 'dataset', 'datafile'
        if source == 'ntcir':
            sql = "SELECT dataset_id, file_id FROM ntcir_datafile"
        elif source == 'acordar':
            sql = "SELECT dataset_id, dataset_id FROM acordar_datasets"
        cursor.execute(sql)
        for item in cursor.fetchall():
            dataset_id, file_id = item
            edges.append((dataset_id, str(file_id), edge_type, weights[edge_type]))
        self.add_edges(edges, src_type, dst_type)
        
        edges = []
        edge_type, src_type, dst_type = 'publish', 'dataset', 'publisher'
        sql = f"SELECT dataset_id, publisher_id FROM {source}_metadata_aug"
        cursor.execute(sql)
        for item in cursor.fetchall():
            dataset_id, publisher_id = item
            edges.append((dataset_id, str(publisher_id), edge_type, weights[edge_type]))
        self.add_edges(edges, src_type, dst_type)

        edges = []
        edge_type, src_type, dst_type = 'publish', 'datafile', 'publisher'
        if source == 'ntcir':
            sql = f"SELECT file_id, publisher_id FROM {source}_datafile_aug"
            cursor.execute(sql)
            for item in cursor.fetchall():
                file_id, publisher_id = item
                edges.append((str(file_id), str(publisher_id), edge_type, weights[edge_type]))
            self.add_edges(edges, src_type, dst_type)

        edges = []
        edge_type, src_type, dst_type = 'theme', 'dataset', 'theme'
        sql = f"SELECT dataset_id, `subject` FROM {source}_metadata_aug"
        cursor.execute(sql)
        for item in cursor.fetchall():
            dataset_id = item[0]
            themes = json.loads(item[1])
            for theme in themes:
                edges.append((dataset_id, theme,  edge_type, weights[edge_type]))
        self.add_edges(edges, src_type, dst_type)

        edges = []
        edge_type, src_type, dst_type = 'content', 'datafile', 'datafile'
        all_data = pd.DataFrame()
        if source == 'ntcir':
            fmt_list = ['text', 'table', 'json_xml']
        elif source == 'acordar':
            fmt_list = ['rdf']
        for fmt in fmt_list:
            csv_file = f'output/similarity/{fmt}_similarity_results.csv'
            df = pd.read_csv(csv_file, dtype={'id1': str, 'id2': str, 'j_sim': float})
            all_data = pd.concat([all_data, df], ignore_index=True)
        all_data_flitered = self.filter_by_row_cnt(all_data, row_cnt_limit)
        link_res = self.generate_sim_pairs(all_data_flitered, edge_type, thresholds[edge_type], weights[edge_type])
        edges.extend(link_res)
        self.add_edges(edges, src_type, dst_type)
        
        edges = []
        edge_type, src_type, dst_type = 'pattern', 'datafile', 'datafile'
        all_data = pd.DataFrame()
        if source == 'ntcir':
            fmt_list = ['table', 'json_xml']
        elif source == 'acordar':
            fmt_list = ['rdf']
        for fmt in fmt_list:
            csv_file = f'output/similarity/{fmt}_pattern_similarity_results.csv'
            df = pd.read_csv(csv_file, dtype={'id1': str, 'id2': str, 'j_sim': float})
            all_data = pd.concat([all_data, df], ignore_index=True)
        all_data_flitered =self.filter_by_row_cnt(all_data, row_cnt_limit)
        link_res = self.generate_sim_pairs(all_data_flitered, edge_type, thresholds[edge_type], weights[edge_type])
        edges.extend(link_res)
        self.add_edges(edges, src_type, dst_type)

        edges = []
        edge_type, src_type, dst_type = 'keyphrase', 'datafile', 'datafile'
        csv_file = f'output/similarity/{source}_keyphrase_15_similarity_results.csv'
        all_data = pd.read_csv(csv_file, dtype={'id1': str, 'id2': str, 'j_sim': float})
        all_data_flitered = self.filter_by_row_cnt(all_data, row_cnt_limit)
        edges = self.generate_sim_pairs(all_data_flitered, edge_type, thresholds[edge_type], weights[edge_type])
        self.add_edges(edges, src_type, dst_type)

    @staticmethod
    def filter_by_row_cnt(df, row_cnt=1000):
        print(f'row count limit: {row_cnt}')
        if row_cnt is None or row_cnt < 0:
            return df
        filtered_df = df

        # 2. Calculate occurrence counts for id1 and id2
        id1_counts = filtered_df['id1'].value_counts()
        id2_counts = filtered_df['id2'].value_counts()

        # 3. Find id1 and id2 where all j_sim values are 1.0
        id1_all_1 = filtered_df.groupby('id1')['j_sim'].agg(lambda x: x.nunique() == 1 and x.iloc[0] == 1.0)
        id2_all_1 = filtered_df.groupby('id2')['j_sim'].agg(lambda x: x.nunique() == 1 and x.iloc[0] == 1.0)

        # 4. Create a combined count series (sum of id1 and id2 counts for each ID)
        combined_counts = id1_counts.add(id2_counts, fill_value=0)
        
        # 5. Find IDs where combined count > row_cnt and all j_sim are 1.0
        bad_ids = combined_counts[
            (combined_counts > row_cnt) & (
            (combined_counts.index.isin(id1_all_1[id1_all_1].index)) | 
            (combined_counts.index.isin(id2_all_1[id2_all_1].index))
            )
        ].index

        # 6. Filter out rows containing these bad IDs in either id1 or id2
        final_df = filtered_df[
            ~filtered_df['id1'].isin(bad_ids) & 
            ~filtered_df['id2'].isin(bad_ids)
        ]

        # 7. Print and return results
        print("Filtered row count:", len(final_df))
        return final_df

    @staticmethod
    def generate_sim_pairs(csv_file, edge_type, threshold=0.8, weight=1.0):
        """
        从CSV文件生成相似对列表
        
        参数:
            csv_file: CSV文件路径
            threshold: 相似度阈值(默认0.8)
            weight: 边权
        
        返回:
            list: 格式为 [(id1, id2, weight), ...] 的列表
        """
        print(f'threshold: {threshold}')
        if type(csv_file) == str:
            # 读取CSV文件
            df = pd.read_csv(csv_file, dtype={'id1': str, 'id2': str, 'j_sim': float})
        elif type(csv_file) == pd.DataFrame:
            df = csv_file
        else:
            print('Invalid csv_file!')
            return []
        
        # 筛选相似度大于threshold的行
        high_sim_df = df[df['j_sim'] > threshold]
        
        # 生成结果列表
        result = []
        for _, row in high_sim_df.iterrows():
            id1 = row['id1']
            id2 = row['id2']

            if id1 == id2:
                continue
            
            # 添加双向关系
            result.append((id1, id2, edge_type, weight))
            result.append((id2, id1, edge_type, weight))
        
        return result
    
    def build_graph(self, source, weights=defaultdict(lambda: 1.0), thresholds=defaultdict(lambda: 0.8), row_cnt_limit=1000):
        self.build_nodes(source)
        self.build_edges(source, weights, thresholds, row_cnt_limit)
        print(f"图信息: 顶点数={self.graph.num_vertices()}, 边数={self.graph.num_edges()}")

    def finalize_and_save(self, graph_path: str, meta_path: str):
        print('save...')
        # 将内部的属性映射到图对象上
        self.graph.vertex_properties["type"] = self.vertex_prop_type
        self.graph.vertex_properties["local_id"] = self.vertex_prop_local_id
        self.graph.edge_properties["type"] = self.edge_type
        self.graph.edge_properties["weight"] = self.edge_weight

        # 保存图结构
        self.graph.save(graph_path)

        # 保存映射信息，包括新的字符串-整数映射
        metadata = {
            "node_type_map": self.node_type_map,
            "node_reverse_map": self.node_reverse_map,
            "node_type_to_int": self.node_type_to_int, # 新增保存
            "edge_type_to_int": self.edge_type_to_int, # 新增保存
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

    @staticmethod
    def load(graph_path: str, meta_path: str):
        print('load...')
        builder = TypedGraphBuilder()
        builder.graph = load_graph(graph_path)

        # 加载属性映射 (现在它们是整数类型)
        builder.vertex_prop_type = builder.graph.vertex_properties["type"]
        builder.vertex_prop_local_id = builder.graph.vertex_properties["local_id"]
        builder.edge_type = builder.graph.edge_properties["type"]
        builder.edge_weight = builder.graph.edge_properties["weight"]

        # 加载映射信息，包括字符串-整数映射
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
            builder.node_type_map = metadata["node_type_map"]
            builder.node_reverse_map = metadata["node_reverse_map"]
            builder.node_type_to_int = metadata["node_type_to_int"] # 加载
            builder.edge_type_to_int = metadata["edge_type_to_int"] # 加载

            # 根据加载的 node_type_to_int 确定下一个可用的整数ID
            if builder.node_type_to_int:
                 builder._next_node_type_int = max(builder.node_type_to_int.values()) + 1
            else:
                 builder._next_node_type_int = 0

            # 根据加载的 edge_type_to_int 确定下一个可用的整数ID
            if builder.edge_type_to_int:
                 builder._next_edge_type_int = max(builder.edge_type_to_int.values()) + 1
            else:
                 builder._next_edge_type_int = 0

            # 重建 type_id_to_vertex 映射，需要根据加载的图和 node_type_map
            builder.type_id_to_vertex = {
                k: builder.graph.vertex(v_id)
                for k, v_id in builder.node_type_map.items()
            }
        return builder

    def graph_view_by_filter_e_type(self, e_type="keyphrase"):
        if type(e_type) == str:
            e_type = self.edge_type_to_int[e_type]
        if type(e_type) != int:
            print("Invalid e_type!")
            return self.graph

        # 创建一个过滤条件：type不等于e_type的边
        filter_cond = self.edge_type.a != e_type
        # 创建过滤后的图视图
        filtered_graph = gt.GraphView(self.graph, efilt=filter_cond)
        return filtered_graph


def calculate_stats(data_array):
    """
    Helper to calculate min, max, mean, median, variance for a numpy array or PropertyArray,
    ensuring results are standard Python types.
    """
    if len(data_array) == 0:
        # Returning 0 for count=0 stats as before
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "variance": 0, "count_nonzero": 0}

    # Calculate stats using numpy functions
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    mean_val = np.mean(data_array)
    median_val = np.median(data_array)
    variance_val = np.var(data_array)
    count_nonzero_val = np.count_nonzero(data_array)

    # Convert results to standard Python types using .item()
    # .item() works on numpy scalars and 0-dim numpy/Property arrays
    return {
        "count": len(data_array), # len returns standard int
        "min": min_val.item() if hasattr(min_val, 'item') else min_val,
        "max": max_val.item() if hasattr(max_val, 'item') else max_val,
        "mean": mean_val.item() if hasattr(mean_val, 'item') else mean_val,
        "median": median_val.item() if hasattr(median_val, 'item') else median_val,
        "variance": variance_val.item() if hasattr(variance_val, 'item') else variance_val,
        "count_nonzero": count_nonzero_val.item() if hasattr(count_nonzero_val, 'item') else count_nonzero_val,
    }

# Check if a property map is a valid integer type
def is_integer_property(prop: gt.PropertyMap) -> bool:
    if prop is None:
        return False
    try:
        return prop.value_type() == "int32_t"
    except ValueError: # value_type might be invalid or not found
        return False

def get_degree_stats_comprehensive(
    graph: Graph,
    vertex_type_prop: gt.PropertyMap = None,
    edge_type_prop: gt.PropertyMap = None,
    node_type_str_to_int: dict = None,
    edge_type_str_to_int: dict = None,
):
    """
    Provides comprehensive degree distribution statistics.

    Includes overall, by node type (total degree), by edge type (across all nodes),
    and by node type AND edge type.

    Args:
        graph: The graph_tool Graph object.
        vertex_type_prop: Vertex property map for node types (integer IDs).
        edge_type_prop: Edge property map for edge types (integer IDs).
        node_type_str_to_int: Dictionary mapping string node types to integer IDs.
                               Used to provide string labels in output.
        edge_type_str_to_int: Dictionary mapping string edge types to integer IDs.
                               Used to provide string labels in output.

    Returns:
        A dictionary containing comprehensive degree statistics.
    """
    stats = {}
    num_vertices = graph.num_vertices()
    num_edges = graph.num_edges()

    if num_vertices == 0:
        print("Graph has no vertices.")
        return stats

    print(f"Calculating degree statistics for graph with {num_vertices} vertices and {num_edges} edges.")

    # --- Overall Degree Statistics ---
    print("\n--- Overall Degree Statistics ---")
    in_degrees = graph.degree_property_map("in")
    out_degrees = graph.degree_property_map("out")
    total_degrees = graph.degree_property_map("total")

    in_degree_array = in_degrees.get_array()
    out_degree_array = out_degrees.get_array()
    total_degree_array = total_degrees.get_array()

    stats["overall"] = {
        "in": calculate_stats(in_degree_array),
        "out": calculate_stats(out_degree_array),
        "total": calculate_stats(total_degree_array),
    }
    print("Overall statistics calculated.")

    # --- Pre-compute edge type counts per vertex if edge type property exists ---
    in_counts_by_etype = None
    out_counts_by_etype = None
    unique_edge_types_int = None
    edge_int_to_str = {}

    # Corrected check for integer type property
    if is_integer_property(edge_type_prop):
        print("\n--- Pre-computing edge type counts per vertex ---")
        unique_edge_types_int = np.unique(edge_type_prop.get_array())
        edge_int_to_str = {v: k for k, v in edge_type_str_to_int.items()} if edge_type_str_to_int else {}

        # Use arrays of size V to store counts for each edge type across all vertices
        in_counts_by_etype = {t_id: np.zeros(num_vertices, dtype=int) for t_id in unique_edge_types_int}
        out_counts_by_etype = {t_id: np.zeros(num_vertices, dtype=int) for t_id in unique_edge_types_int}

        # Iterate through edges to populate the count arrays
        if num_edges > 0:
             for e in graph.edges():
                 # Corrected vertex index access
                 u_idx = int(e.source())
                 v_idx = int(e.target())
                 e_type_id = edge_type_prop[e]
                 if e_type_id in in_counts_by_etype: # Ensure the type is one we are tracking
                     out_counts_by_etype[e_type_id][u_idx] += 1
                     in_counts_by_etype[e_type_id][v_idx] += 1
        print("Edge type counts per vertex pre-computed.")

    else:
         print("\nSkipping edge type counting: edge_type_prop is not provided or is not an integer property map.")


    # --- Statistics by Node Type (Total Degree for nodes of a specific type) ---
    unique_node_types_int = None
    node_int_to_str = {}

    # Corrected check for integer type property
    if is_integer_property(vertex_type_prop):
        print("\n--- Statistics by Node Type (Total Degree) ---")
        unique_node_types_int = np.unique(vertex_type_prop.get_array())
        stats_by_node_type = {}
        node_int_to_str = {v: k for k, v in node_type_str_to_int.items()} if node_type_str_to_int else {}

        for type_id in unique_node_types_int:
            type_str = node_int_to_str.get(type_id, f"ID_{type_id}")
            print(f" Processing node type: {type_str} (ID: {type_id})...")

            # Create a boolean map for vertices of this type
            # More efficient filtering using numpy array comparison directly
            is_this_type_array = (vertex_type_prop.get_array() == type_id)

            # Filter degree arrays using the boolean array
            in_degrees_for_type = in_degree_array[is_this_type_array]
            out_degrees_for_type = out_degree_array[is_this_type_array]
            total_degrees_for_type = total_degree_array[is_this_type_array]

            type_stats = {}
            type_stats["in"] = calculate_stats(in_degrees_for_type)
            type_stats["out"] = calculate_stats(out_degrees_for_type)
            type_stats["total"] = calculate_stats(total_degrees_for_type)

            stats_by_node_type[f"{type_str} (ID: {type_id})"] = type_stats

        stats["by_node_type"] = stats_by_node_type
        print("Node type statistics calculated.")

    else:
         print("\nSkipping statistics by node type: vertex_type_prop is not provided or is not an integer property map.")

    # --- Statistics by Edge Type (Across all nodes) ---
    if in_counts_by_etype is not None: # Check if edge type counts were computed
         print("\n--- Statistics by Edge Type (Across All Nodes) ---")
         stats_by_edge_type_all_nodes = {}

         for type_id in unique_edge_types_int:
              type_str = edge_int_to_str.get(type_id, f"ID_{type_id}")
              print(f" Processing edge type: {type_str} (ID: {type_id})...")

              in_counts_array = in_counts_by_etype[type_id]
              out_counts_array = out_counts_by_etype[type_id]

              type_stats = {}
              # Stats on IN counts for this edge type across all vertices
              type_stats["in"] = calculate_stats(in_counts_array)
              # Stats on OUT counts for this edge type across all vertices
              type_stats["out"] = calculate_stats(out_counts_array)

              stats_by_edge_type_all_nodes[f"{type_str} (ID: {type_id})"] = type_stats

         stats["by_edge_type_all_nodes"] = stats_by_edge_type_all_nodes
         print("Edge type statistics (across all nodes) calculated.")

    # --- Statistics by Node Type AND Edge Type ---
    # Check if both type properties were valid integer properties
    if is_integer_property(vertex_type_prop) and is_integer_property(edge_type_prop):
         print("\n--- Statistics by Node Type AND Edge Type ---")
         stats_by_node_type_and_edge_type = {}

         # unique_node_types_int and unique_edge_types_int are already computed
         # node_int_to_str and edge_int_to_str are already computed
         # in_counts_by_etype and out_counts_by_etype are already computed

         for node_type_id in unique_node_types_int:
             node_type_str = node_int_to_str.get(node_type_id, f"ID_{node_type_id}")
             print(f" Processing node type: {node_type_str} (ID: {node_type_id})...")

             # Re-create boolean array for this node type using the validated prop
             is_this_node_type_array = (vertex_type_prop.get_array() == node_type_id)

             edge_type_stats_for_this_node_type = {}

             for edge_type_id in unique_edge_types_int:
                 edge_type_str = edge_int_to_str.get(edge_type_id, f"ID_{edge_type_id}")
                 print(f"  Processing edge type: {edge_type_str} (ID: {edge_type_id})...")

                 # Filter the pre-computed counts using the boolean array for node type
                 # Ensure the edge_type_id exists in the pre-computed counts dict
                 if edge_type_id in in_counts_by_etype:
                     in_counts_for_combo = in_counts_by_etype[edge_type_id][is_this_node_type_array]
                     out_counts_for_combo = out_counts_by_etype[edge_type_id][is_this_node_type_array]

                     combo_stats = {}
                     combo_stats["in"] = calculate_stats(in_counts_for_combo)
                     combo_stats["out"] = calculate_stats(out_counts_for_combo)

                     edge_type_stats_for_this_node_type[f"Edge {edge_type_str} (ID: {edge_type_id})"] = combo_stats
                 else:
                     # Should not happen if unique_edge_types_int comes from edge_type_prop
                     print(f"  Warning: Edge type ID {edge_type_id} not found in pre-computed counts.")


             stats_by_node_type_and_edge_type[f"Node {node_type_str} (ID: {node_type_id})"] = edge_type_stats_for_this_node_type

         stats["by_node_type_and_edge_type"] = stats_by_node_type_and_edge_type
         print("Node type AND Edge type statistics calculated.")

    elif not is_integer_property(vertex_type_prop):
        print("\nSkipping Node type AND Edge type statistics: vertex_type_prop is not provided or is not an integer property map.")
    elif not is_integer_property(edge_type_prop):
        print("\nSkipping Node type AND Edge type statistics: edge_type_prop is not provided or is not an integer property map.")

    # Include mappings for clarity if provided
    if node_type_str_to_int:
         stats.setdefault("mappings", {})["node_type_int_to_str"] = {v: k for k, v in node_type_str_to_int.items()}
    if edge_type_str_to_int:
         stats.setdefault("mappings", {})["edge_type_int_to_str"] = {v: k for k, v in edge_type_str_to_int.items()}

    return stats


class Cluster:
    def __init__(self, g: Graph = None):
        self.g = g
        self.state = None  # 存储块模型状态
        self.levels = None  # 存储层次聚类结果

    def nested_clustering(self, deg_corr: bool = True, verbose: bool = True, use_type_prop: bool = False) -> None:
        """
        执行层次化嵌套聚类
        
        参数:
            deg_corr: 是否使用度校正
            verbose: 是否打印详细信息
        """
        if self.g is None:
            raise ValueError("图未加载")
            
        if verbose:
            print("开始层次聚类...")
            print(f"初始图信息: 顶点数={self.g.num_vertices()}, 边数={self.g.num_edges()}")
        
        # 确保图满足基本要求
        if self.g.num_vertices() == 0:
            raise ValueError("图中没有顶点")
        if self.g.num_edges() == 0:
            raise ValueError("图中没有边")
        
        try:
            # 准备状态参数
            state_args = {
                'deg_corr': deg_corr,
            }
            if use_type_prop:
                state_args['clabel'] = self.g.vp["type"]
                print("使用节点 type 作为聚类约束 (clabel)")
                state_args['recs'] = [self.g.ep["type"]]
                state_args['rec_types'] = ['discrete-geometric']
                print(f"使用边 type 作为协变量 (recs)，类型为: {state_args['rec_types']}")

            # 执行聚类
            self.state = gt.minimize_nested_blockmodel_dl(
                self.g,
                state_args=state_args,
            )

            # 存储层次结果
            self.levels = self.state.get_levels()
            if verbose:
                print(f"聚类完成，共发现 {len(self.levels)} 个层级")
                for i, level in enumerate(self.levels):
                    print(f"层级 {i}: {level.get_nonempty_B()} 个社区")
                    
        except Exception as e:
            raise RuntimeError(f"聚类失败: {str(e)}") from e

    def get_clusters(self, level: int = 0) -> gt.PropertyMap:
        """
        获取指定层级的聚类结果
        
        参数:
            level: 层级深度(0为最粗粒度)
        
        返回:
            节点到社区ID的映射
        """
        if self.state is None:
            raise RuntimeError("请先执行聚类")
            
        if level >= len(self.levels):
            raise ValueError(f"请求的层级{level}不存在，最大层级为{len(self.levels)-1}")
            
        return self.levels[level].get_blocks()

    @staticmethod
    def visualize_cluster_numpy(g: Graph,
                            blocks_array: np.array,
                            distinct_color: bool = False,
                            output_path: str = "community_graph.pdf",
                            v_types: list = None):
        assert blocks_array.shape[0] == g.num_vertices()

        # === 1. 节点类型过滤 ===
        if v_types is not None:
            # 创建类型掩码 (1=保留，0=过滤)
            type_mask = np.isin(g.vp["type"].a, v_types)
            filtered_blocks = blocks_array[type_mask]
        else:
            type_mask = np.ones(g.num_vertices(), dtype=bool)
            filtered_blocks = blocks_array

        # 如果没有节点被保留，则无法进行后续操作
        if filtered_blocks.size == 0:
            print("Warning: No nodes match the specified v_types. Cannot visualize community graph.")
            return

        # === 2. 社区规模统计（考虑类型过滤）===
        # 只统计保留节点的社区分布
        unique_comms, comm_sizes = np.unique(filtered_blocks, return_counts=True)
        community_sizes = dict(zip(unique_comms, comm_sizes))

        # === 3. 社区间连接统计（考虑类型过滤）===
        sources = g.get_edges()[:, 0]
        targets = g.get_edges()[:, 1]

        # 仅保留两端节点都在v_types中的边
        if v_types is not None:
            valid_edge_mask = type_mask[sources] & type_mask[targets]
            sources = sources[valid_edge_mask]
            targets = targets[valid_edge_mask]

        # 过滤掉社区对中的自环（同一个社区内部的边，在社区图中不显示为边）
        # 并且只考虑保留节点所在的社区
        valid_sources_comm = blocks_array[sources]
        valid_targets_comm = blocks_array[targets]
        comm_pairs = np.column_stack([valid_sources_comm, valid_targets_comm])
        cross_mask = comm_pairs[:, 0] != comm_pairs[:, 1]

        # 进一步过滤，只保留两端社区ID都在 unique_comms 中的边
        comm_id_set = set(unique_comms)
        valid_comm_pair_mask = np.array([(p[0] in comm_id_set) and (p[1] in comm_id_set) for p in comm_pairs])
        final_edge_mask = cross_mask & valid_comm_pair_mask

        # 对社区对排序并统计
        if np.sum(final_edge_mask) > 0:
            sorted_pairs = np.sort(comm_pairs[final_edge_mask], axis=1)
            unique_pairs, pair_counts = np.unique(sorted_pairs, axis=0, return_counts=True)
            community_edges = {(pair[0], pair[1]): count for pair, count in zip(unique_pairs, pair_counts)}
        else:
            community_edges = {} # 没有边连接不同社区

        # === 4. 构建社区图 ===
        comm_graph = gt.Graph(directed=False)
        # 确保只创建存在于 filtered_blocks 中的社区节点
        comm_id = {c: comm_graph.add_vertex() for c in unique_comms}
        # 创建一个属性映射，从comm_graph顶点到原始社区ID
        comm_to_orig_id = comm_graph.new_vp("int")
        for orig_id, v in comm_id.items():
            comm_to_orig_id[v] = orig_id

        # 添加边（向量化方式）
        edge_weights = comm_graph.new_edge_property("double")
        comm_graph.ep.weight = edge_weights

        for (c1, c2), count in community_edges.items():
            # 确保c1和c2都在unique_comms中
            if c1 in comm_id and c2 in comm_id:
                e = comm_graph.add_edge(comm_id[c1], comm_id[c2])
                edge_weights[e] = count

        # === 5. 连通分量分析和布局 ===
        # 找到连通分量
        comp, hist = gt.label_components(comm_graph)
        # hist[i] 是大小为 i 的连通分量的数量
        # comp 是一个顶点属性map，每个顶点的值是其所属连通分量的标签

        # 找出孤立节点（大小为1的连通分量）的comm_graph顶点
        isolated_verts = [v for v in comm_graph.vertices() if hist[comp[v]] == 1]

        # 找出非孤立节点的comm_graph顶点
        non_isolated_verts = [v for v in comm_graph.vertices() if hist[comp[v]] > 1]

        # 创建最终的位置属性图
        pos = comm_graph.new_vp("vector<double>")

        # 如果存在非孤立节点，对其子图进行布局
        if non_isolated_verts:
            # 创建一个GraphView只包含非孤立节点
            non_isolated_mask = comm_graph.new_vp("bool", val=False)
            for v in non_isolated_verts:
                non_isolated_mask[v] = True
            non_isolated_graph_view = gt.GraphView(comm_graph, vfilt=non_isolated_mask)

            # 对子图进行布局
            non_isolated_pos = gt.sfdp_layout(non_isolated_graph_view, C=1.5)

            # 将布局结果复制到最终位置属性图中
            for i, v in enumerate(non_isolated_graph_view.vertices()):
                # Map the vertex v in the GraphView back to the original comm_graph vertex
                # This requires careful handling of vertex indices if using gt.Graph directly
                # Using GraphView's internal vertex mapping is safer
                # A simpler way is to iterate through the non_isolated_verts list and use the index
                original_v = non_isolated_verts[i]
                pos[original_v] = non_isolated_pos[v]


            # 计算非孤立部分的中心和范围，以便放置孤立点
            non_isolated_coords = np.array([pos[v] for v in non_isolated_verts])
            center = np.mean(non_isolated_coords, axis=0)
            #rough_radius = np.max(np.linalg.norm(non_isolated_coords - center, axis=1)) if non_isolated_coords.size > 0 else 0
            # A more robust radius calculation considering the spread
            # Use the max distance from the center, or a percentile, or related to the bounding box
            if non_isolated_coords.size > 0:
                min_coords = np.min(non_isolated_coords, axis=0)
                max_coords = np.max(non_isolated_coords, axis=0)
                layout_size = np.max(max_coords - min_coords)
                placement_radius = layout_size * 0.7 # Adjust this factor as needed
            else:
                center = np.array([0, 0])
                placement_radius = 100 # Default radius if no non-isolated nodes


        else:
            # 如果没有非孤立节点，中心和半径默认为0
            center = np.array([0, 0])
            placement_radius = 100 # Default radius if only isolated nodes

        # 为孤立节点计算圆形布局
        num_isolated = len(isolated_verts)
        if num_isolated > 0:
            angles = np.linspace(0, 2 * np.pi, num_isolated, endpoint=False)
            isolated_coords = np.array([
                [center[0] + placement_radius * np.cos(a),
                center[1] + placement_radius * np.sin(a)]
                for a in angles
            ])
            for i, v in enumerate(isolated_verts):
                pos[v] = isolated_coords[i]

        # === 6. 可视化属性设置 ===
        # 节点大小（与社区规模对数正比）
        v_size = comm_graph.new_vp("double")
        for v in comm_graph.vertices():
            original_comm_id = comm_to_orig_id[v]
            v_size[v] = np.log(community_sizes[original_comm_id] + 1)  # +1避免log(0)
        v_size = gt.prop_to_size(v_size, mi=10, ma=50) # 调整min/max size

        # 节点颜色
        v_color = comm_graph.new_vp("vector<double>")
        if distinct_color:
            cmap = mpl.colormaps["tab20"]
            for v in comm_graph.vertices():
                # Use the original community ID for consistent coloring if distinct_color is True
                original_comm_id = comm_to_orig_id[v]
                rgba = cmap(original_comm_id % 20)  # 使用社区ID作为索引
                v_color[v] = list(rgba[:3]) + [1.0]  # 确保alpha=1
        else:
            for v in comm_graph.vertices():
                v_color[v] = [0.6, 0.6, 0.8, 1.0]  # 统一浅蓝色

        # 边宽度（对数缩放权重）
        e_width = comm_graph.new_edge_property("double")
        for e in comm_graph.edges():
            e_width[e] = np.log(comm_graph.ep.weight[e] + 1)
        e_width = gt.prop_to_size(e_width, mi=0.5, ma=5)

        # 节点文本（社区ID(规模)）
        v_text = comm_graph.new_vp("string")
        for v in comm_graph.vertices():
            original_comm_id = comm_to_orig_id[v]
            v_text[v] = f"{original_comm_id}({community_sizes[original_comm_id]})"

        # === 7. 绘图 ===
        gt.graph_draw(
            comm_graph,
            pos=pos, # 使用计算好的布局
            vertex_size=v_size,
            vertex_fill_color=v_color,
            edge_pen_width=e_width,
            output=output_path,
            output_size=(1000, 1000),
            vertex_text=v_text, # 使用构建好的文本属性
            vertex_font_size=12,
            edge_color=[0.7, 0.7, 0.7, 0.6]  # 半透明灰色边
        )
        print(f"Community graph visualized and saved to {output_path}")

    def visualize_community_level(self, output_dir: str, level: int = 0, distinct_color: bool = False) -> None:
        """
        绘制社区级别的网络图（每个社区作为一个超级节点）
        
        参数:
            output_dir: 输出目录路径
            level: 要可视化的层级
            distinct_color: 是否为不同社区分配不同颜色
        """
        if self.state is None:
            raise RuntimeError("请先执行聚类")
            
        print(f"正在可视化层级 {level}的社区网络...")
        
        # 获取该层级的聚类状态
        lstate = self.levels[level]
        blocks = lstate.get_blocks()
        
        # === 1. 向量化统计 ===
        blocks_array = blocks.get_array()  # 转换为numpy数组
        if blocks_array.shape[0] != self.g.num_vertices():
            # 创建一个节点到社区 ID 的字典
            node_to_community = {int(v): blocks[v] for v in self.g.vertices()}
            # 将字典转换为 NumPy 数组，保持节点顺序
            blocks_array = np.array([node_to_community[int(i)] for i in self.g.vertices()])
        
        # # 社区规模统计
        # unique_comms, comm_sizes = np.unique(blocks_array, return_counts=True)
        # community_sizes = dict(zip(unique_comms, comm_sizes))
        
        # # 社区间连接统计
        # sources = self.g.get_edges()[:, 0]
        # targets = self.g.get_edges()[:, 1]
        # comm_pairs = np.column_stack([blocks_array[sources], blocks_array[targets]])
        # cross_mask = comm_pairs[:, 0] != comm_pairs[:, 1]
        
        # # 对社区对排序并统计
        # sorted_pairs = np.sort(comm_pairs[cross_mask], axis=1)
        # unique_pairs, pair_counts = np.unique(sorted_pairs, axis=0, return_counts=True)
        # community_edges = {(pair[0], pair[1]): count for pair, count in zip(unique_pairs, pair_counts)}
        
        # # === 2. 构建社区图 ===
        # comm_graph = gt.Graph(directed=False)
        # comm_id = {c: comm_graph.add_vertex() for c in community_sizes}
        
        # # 添加边（向量化方式）
        # edge_weights = comm_graph.new_edge_property("double")
        # comm_graph.ep.weight = edge_weights
        
        # for (c1, c2), count in community_edges.items():
        #     e = comm_graph.add_edge(comm_id[c1], comm_id[c2])
        #     edge_weights[e] = count
        
        # # === 3. 可视化属性设置 ===
        # # 节点大小（与社区规模对数正比，避免极端大小差异）
        # v_size = comm_graph.new_vp("double")
        # for c, v in comm_id.items():
        #     v_size[v] = np.log(community_sizes[c] + 1)  # +1避免log(0)
        # v_size = gt.prop_to_size(v_size, mi=1, ma=10)  # mi=10, ma=50
        
        # # 节点颜色
        # v_color = comm_graph.new_vp("vector<double>")
        # if distinct_color:
        #     cmap = mpl.colormaps["tab20"]
        #     for v in comm_graph.vertices():
        #         rgba = cmap(int(v) / len(community_sizes))
        #         v_color[v] = list(rgba)
        # else:
        #     for v in comm_graph.vertices():
        #         v_color[v] = [0.6, 0.6, 0.8, 1.0]  # 统一浅蓝色
        
        # # 边宽度（对数缩放权重）
        # e_width = comm_graph.new_edge_property("double")
        # for e in comm_graph.edges():
        #     e_width[e] = np.log(comm_graph.ep.weight[e] + 1)
        # e_width = gt.prop_to_size(e_width, mi=0.5, ma=5)
        
        # # === 4. 绘图 ===
        # pos = gt.sfdp_layout(comm_graph)
        # output_path = os.path.join(output_dir, f"community_graph_l{level}.pdf")
        
        # gt.graph_draw(
        #     comm_graph,
        #     pos=pos,
        #     vertex_size=v_size,
        #     vertex_fill_color=v_color,
        #     edge_pen_width=e_width,
        #     output=output_path,
        #     output_size=(1000, 1000),
        #     vertex_text=comm_graph.new_vertex_property(
        #         "string",
        #         vals=[f"{k}({v})" for k, v in community_sizes.items()]
        #     ),
        #     vertex_font_size=12,
        #     edge_color=[0.7, 0.7, 0.7, 0.6]  # 半透明灰色边
        # )

        output_path = os.path.join(output_dir, f"community_graph_l{level}.pdf")
        self.visualize_cluster_numpy(self.g, blocks_array, distinct_color, output_path)

    def analyze(self) -> dict:
        """
        分析聚类结果
        
        返回:
            包含各种统计指标的字典
        """
        if self.state is None:
            raise RuntimeError("请先执行聚类")
            
        stats = {}
        base_state = self.levels[0]
        
        # 基础统计
        stats['n_levels'] = len(self.levels)
        stats['n_blocks'] = [s.get_nonempty_B() for s in self.levels]
        stats['modularity'] = gt.modularity(self.g, self.get_clusters())
        
        # 度相关性
        stats['assortativity'] = gt.scalar_assortativity(self.g, "out")
        
        return stats

    def save_results(self, 
                   output_dir: str = "clustering_results",
                   level: int = 0,
                   formats: list = ["pkl", "csv", "json"],
                   include_metadata: bool = True) -> None:
        """
        保存聚类结果到指定目录
        
        参数:
            output_dir: 输出目录路径
            level: 要保存的层级(0为最粗粒度)
            formats: 保存格式列表，支持["pkl", "csv", "json", "graphml"]
            include_metadata: 是否包含图元数据
        """
        if self.state is None:
            raise RuntimeError("请先执行聚类")
            
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        if level == -1:
            level_lst = range(len(self.levels))
        else:
            level_lst = [level]
        for level in level_lst:
            base_name = f"level_{level}"
            # 保存不同格式的结果
            if "pkl" in formats:
                self._save_pkl(output_dir, base_name, level, include_metadata)
            # if "csv" in formats:
            #     self._save_csv(output_dir, base_name, clusters)
            # if "json" in formats:
            #     self._save_json(output_dir, base_name, clusters, include_metadata)
            # if "graphml" in formats:
            #     self._save_graphml(output_dir, base_name, level)
            
        print(f"聚类结果已保存到 {output_dir}")
    
    def _save_pkl(self, output_dir: str, base_name: str, level: int, include_metadata: bool) -> None:
        """保存Python pickle格式"""
            # 获取该层级的聚类状态
        lstate = self.levels[level]
        blocks = lstate.get_blocks()
        blocks_array = blocks.get_array()  # 转换为numpy数组
        if blocks_array.shape[0] != self.g.num_vertices():
            # 创建一个节点到社区 ID 的字典
            node_to_community = {v: blocks[v] for v in self.g.vertices()}
            # 将字典转换为 NumPy 数组，保持节点顺序
            blocks_array = np.array([node_to_community[v] for v in self.g.vertices()])
        data = {
            'clusters': blocks_array,
            'timestamp': np.datetime64('now')
        }
        
        if include_metadata:
            data['metadata'] = {
                'n_vertices': self.g.num_vertices(),
                'n_edges': self.g.num_edges(),
                'modularity': gt.modularity(self.g, self.get_clusters(0)),
                'levels': len(self.levels),
                'blocks': self.levels[0].get_nonempty_B()
            }
            
        with open(os.path.join(output_dir, f"{base_name}.pkl"), 'wb') as f:
            pickle.dump(data, f)


def filter_edges_by_type_threshold(g, edge_type_prop, threshold):
    # 获取边类型属性
    edge_type = g.edge_properties[edge_type_prop] if isinstance(edge_type_prop, str) else edge_type_prop
    
    # 创建边过滤属性
    edge_filter = g.new_edge_property("bool", val=True)
    
    # 获取所有边的类型数组
    edge_types = edge_type.get_array()
    
    # 预先计算每个顶点的邻居边
    for v in g.vertices():
        # 获取该顶点的所有边(出边+入边)
        all_edges = list(v.out_edges()) + list(v.in_edges())
        if not all_edges:
            continue
            
        # 获取这些边的类型
        edge_indices = [g.edge_index[e] for e in all_edges]
        types = edge_types[edge_indices]
        
        # 使用NumPy的bincount统计每种类型的数量
        if len(types) > 0:
            type_counts = np.bincount(types)
            over_threshold = np.where(type_counts > threshold)[0]
            
            # 标记需要过滤的边
            for e in all_edges:
                if edge_type[e] in over_threshold:
                    edge_filter[e] = False
    
    # 创建并返回过滤后的图视图
    return gt.GraphView(g, efilt=edge_filter)


def get_node_text(source, node_type):
    KEYPHRASE_NUM = 15
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
                id_text_dict[item[0]] = id_text_dict.get(item[0], '')+ '\n'.join([str(item[i]) for i in range(1, len(item))])
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
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item)-1)])
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
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item)-1)])
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
                id_text_dict[str(item[0])] = '\n'.join([str(item[i]) for i in range(1, len(item)) if item[i]])
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
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item)-1)])
                id_text_dict[item[0]] += item[-1].replace(';', '\t')
        elif node_type == 'datafile':
            sql = "SELECT file_id, keyphrase FROM acordar_keyphrase"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\t'.join(json.loads(item[1])[:KEYPHRASE_NUM]) + '\n'
            sql = f"SELECT dataset_id, download FROM acordar_datasets"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = id_text_dict.get(item[0], '') +  '\n'.join(eval(item[1]) + ['rdf'])
        elif node_type == 'publisher':
            fields = ['label', 'wd_label', 'wd_description', 'url']
            # exclude null
            sql = f"SELECT id, {', '.join(fields)} FROM acordar_publisher WHERE LENGTH(CONCAT(label,sameAs,url)) > 0"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[str(item[0])] = '\n'.join([str(item[i]) for i in range(1, len(item)) if item[i]])
        elif node_type == 'theme':
            fields = ['label', 'definition']
            sql = f"SELECT code, {', '.join(fields)} FROM subject_theme"
            cursor.execute(sql)
            for item in cursor.fetchall():
                id_text_dict[item[0]] = '\n'.join([str(item[i]) for i in range(1, len(item))])
    return id_text_dict


def convert_graph_to_data(graph_path, meta_path, source, output_dir):
    tgb = TypedGraphBuilder.load(graph_path, meta_path)

    node_type_text_dict = {}
    for node_type in tgb.node_type_to_int:
        node_type_text_dict[node_type] = get_node_text(source, node_type)
    corpus = {}
    nodes = {}
    for v in tgb.graph.vertices():
        v_type, v_id = tgb.node_reverse_map[int(v)]
        corpus[str(int(v))] = node_type_text_dict[v_type][v_id]
        v_type = tgb.vertex_prop_type[v]
        nodes.setdefault(v_type, []).append((int(v), v_id, v_type))
    with open(os.path.join(output_dir, 'corpus.json'), 'w', encoding='utf-8') as f:
        json.dump(corpus, f)
    with open(os.path.join(output_dir, 'node_temp.dat'), 'w') as f:  # embedding
        for v_type in sorted(nodes):
            for v in nodes[v_type]:
                f.write(f'{v[0]}\t{v[1]}\t{v[2]}\n')

    edges = {}
    for edge in tgb.graph.edges():
        src_id = int(edge.source())
        dst_id = int(edge.target())
        e_type = tgb.edge_type[edge]
        e_weight = tgb.edge_weight[edge]
        edges.setdefault(e_type, []).append((src_id, dst_id, e_type, e_weight))
    skip_e_type = []
    if source == 'acordar_metadata':
        skip_e_type.append(tgb.edge_type_to_int['content'])
        skip_e_type.append(tgb.edge_type_to_int['pattern'])
        skip_e_type.append(tgb.edge_type_to_int['keyphrase'])
    with open(os.path.join(output_dir, 'link.dat'), 'w') as f:
        for e_type in sorted(edges):
            if e_type in skip_e_type:
                continue
            for e in edges[e_type]:
                f.write(f'{e[0]}\t{e[1]}\t{e[2]}\t{e[3]}\n')

    if 'acordar' in source:
        acordar_qrel_dir = '../data/acordar/Splits for Cross Validation'
        for sub_dir in os.listdir(acordar_qrel_dir):
            os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
            for qrel_file in os.listdir(os.path.join(acordar_qrel_dir, sub_dir)):
                qrel_dict = {}
                with open(os.path.join(acordar_qrel_dir, sub_dir, qrel_file), 'r') as f:
                    for line in f:
                        items = line.strip().split('\t')
                        query_id = items[0]
                        node_id = str(tgb.node_type_map[('dataset', items[2])])
                        qrel = int(items[3])
                        qrel_dict[query_id] = qrel_dict.get(query_id, {})
                        qrel_dict[query_id][node_id] = qrel
                filename = 'val.json' if qrel_file == 'valid.txt' else os.path.splitext(qrel_file)[0] + '.json'
                with open(os.path.join(output_dir, sub_dir, filename), 'w') as f:
                    json.dump(qrel_dict, f, indent=2)


def get_dataset_node_cluster_dict(tgb: TypedGraphBuilder, blocks: np.array):
    assert blocks.shape[0] == tgb.graph.num_vertices()
    # 获取所有顶点的类型数组
    types = tgb.graph.vp['type'].get_array()
    # 获取类型为0的顶点索引
    dataset_type = tgb.node_type_to_int['dataset']
    indices = np.where(types == dataset_type)[0]
    # 构建字典
    result_dict = {tgb.node_reverse_map[idx][1]: int(blocks[idx]) for idx in indices}
    return result_dict  # dataset_id: comm_id


def print_comm_id_with_e_type(tgb: TypedGraphBuilder, blocks: np.array, e_types=None):
    g = tgb.graph
    # === 1. 获取所有社区ID ===
    community_ids = np.unique(blocks)

    unique_comms, comm_sizes = np.unique(blocks, return_counts=True)
    community_sizes = dict(zip(unique_comms, comm_sizes))  # 获取社区大小

    edges = g.get_edges()  # 获取所有边的 (源节点, 目标节点) 数组
    edge_types = g.ep['type'].a  # 所有边的 type 数组
    
    if e_types is None:
        e_types = tgb.edge_type_to_int.keys()
    elif type(e_types) == str:
        e_types = [e_types]

    for e_type_str in e_types:
        e_type_int = tgb.edge_type_to_int[e_type_str]
        community_has_type = {comm_id: False for comm_id in community_ids}

        # === 2. 获取所有边，并过滤出 type=e_type_int 的边 ===
        type_mask = (edge_types == e_type_int)  # type=e_type_int 的边的布尔掩码

        # 仅保留 ttype=e_type_int 的边
        type_edges = edges[type_mask]

        # === 3. 检查这些边的两端是否在同一个社区 ===
        src_comms = blocks[type_edges[:, 0]]  # 源节点的社区ID
        tgt_comms = blocks[type_edges[:, 1]]  # 目标节点的社区ID
        internal_edges_mask = (src_comms == tgt_comms)  # 是否为社区内部边

        # 获取社区内部边对应的社区ID（取src_comms或tgt_comms均可，因为此时它们相同）
        internal_comms = src_comms[internal_edges_mask]

        # 统计每个社区的内部边数量
        unique_comms, comm_edge_counts = np.unique(internal_comms, return_counts=True)
        community_internal_edges = dict(zip(unique_comms, comm_edge_counts))

        # 标记存在 type=e_type_int 内部边的社区
        communities_with_type = set(src_comms[internal_edges_mask])
        for comm_id in communities_with_type:
            community_has_type[comm_id] = True

        # === 4. 输出结果 ===
        comm_ids = [int(k) for k, v in community_has_type.items() if v]
        # print(f"类型为 {e_type_str} 的边：", end='')
        for x in comm_ids:
            comm_size = community_sizes[x]
            comm_edge_size = community_internal_edges[x]
            # print(f'{x}({comm_size}/{comm_edge_size})', end=', ')
        # print('\n比例较大的社区：')
        # for x in comm_ids:
        #     comm_size = community_sizes[x]
        #     comm_edge_size = community_internal_edges[x]
        #     if comm_edge_size/comm_size > 0.1 and comm_size > 50 and comm_edge_size < 500:
        #         print(f'{x}({comm_edge_size/comm_size:.4f},{comm_edge_size}/{comm_size})', end=', ')
        # print()
        return {x:(community_sizes[x], community_internal_edges[x]) for x in comm_ids}


def print_comm_edges_with_e_type(tgb, blocks, target_comm_id, e_type):
    if type(e_type) == str:
        e_type_int = tgb.edge_type_to_int[e_type]
    else:
        e_type_int = e_type
    g = tgb.graph
    edges = g.get_edges()  # 获取所有边的 (源节点, 目标节点) 数组
    edge_types = g.ep['type'].a  # 所有边的 type 数组

    # === 1. 找出社区内部的所有边 ===
    src_comms = blocks[edges[:, 0]]
    tgt_comms = blocks[edges[:, 1]]
    internal_mask = (src_comms == target_comm_id) & (tgt_comms == target_comm_id)
    internal_edges = edges[internal_mask]
    internal_edge_types = edge_types[internal_mask]
    
    # === 2. 找出社区内所有有 type=dump 边的节点 ===
    dump_type_int = tgb.edge_type_to_int['dump']
    type0_mask = (internal_edge_types == dump_type_int)
    type0_edges = internal_edges[type0_mask]
    type0_nodes = np.unique(np.concatenate([type0_edges[:, 0], type0_edges[:, 1]]))
    
    
    # === 3. 找出社区内 type={e_type_int} 的边 ===
    target_type_mask = (internal_edge_types == e_type_int)
    target_type_edges = internal_edges[target_type_mask]
    
    # === 4. 筛选：边的两端节点都在 type0_nodes 里（即都有 type=0 的边） ===
    src_nodes = target_type_edges[:, 0]
    tgt_nodes = target_type_edges[:, 1]
    
    # 检查 src 和 tgt 是否都在 type0_nodes 里
    src_has_type0 = np.isin(src_nodes, type0_nodes)
    tgt_has_type0 = np.isin(tgt_nodes, type0_nodes)
    both_have_type0 = src_has_type0 & tgt_has_type0
    
    filtered_edges = target_type_edges[both_have_type0]

    # === 5. 输出结果 ===
    # print(f"社区 {target_comm_id} 内满足条件的 type={e_type_int} 边（两端都有 type=0 的边）：")
    str_ids = []
    for src, tgt in filtered_edges:
        src_str = tgb.node_reverse_map[int(src)][1]
        tgt_str = tgb.node_reverse_map[int(tgt)][1]
        str_ids.extend([src_str, tgt_str])
        # print(f"{src_str} -> {tgt_str}")
    
    # print("\n所有涉及的节点：")
    # print(",".join(set(str_ids)))
    # print()
    # return filtered_edges
    return str_ids



def cluster_run(source):
    save_dir = f'data/cluster/{source}'
    if source == 'ntcir':
        graph_path = os.path.join(save_dir, 'graph_flt.gt')
        meta_path = os.path.join(save_dir, 'meta_flt.pkl')
    else:
        graph_path = os.path.join(save_dir, 'graph.gt')
        meta_path = os.path.join(save_dir, 'meta.pkl')

    tgb = TypedGraphBuilder.load(graph_path, meta_path)
    g = tgb.graph_view_by_filter_e_type(e_type="keyphrase")

    level_num = 5
    use_type_prop = True
    if use_type_prop:
        output_dir = os.path.join(save_dir, "use_type_prop")
    else:
        output_dir = save_dir
    os.makedirs(output_dir, exist_ok=True)

    cluster = Cluster(g)
    cluster.nested_clustering()
    print(cluster.analyze())
    for level in range(level_num):
        cluster.save_results(output_dir=output_dir, formats=["pkl"], level=level)
    for level in range(level_num):
        cluster.visualize_community_level(output_dir=output_dir, level=level, distinct_color=True)


def build_graph_run(source):
    if source == 'ntcir':
        thresholds = {'content': 0.82, 'pattern': 0.83, 'keyphrase': 0.53}
    elif source == 'acordar':
        thresholds = {'content': 0.54, 'pattern': 0.67, 'keyphrase': 0.61}

    save_dir = f'data/cluster/{source}'
    os.makedirs(save_dir, exist_ok=True)
    graph_path = os.path.join(save_dir, 'graph.gt')
    meta_path = os.path.join(save_dir, 'meta.pkl')

    row_cnt_limit = 1000
    print(f'build {source} graph, row_cnt_limit={row_cnt_limit}')

    tgb = TypedGraphBuilder()
    tgb.build_graph(source=source, thresholds=thresholds, row_cnt_limit=row_cnt_limit)
    tgb.finalize_and_save(graph_path, meta_path)




if __name__ == "__main__":
    # test()

    # source = 'ntcir'
    source = 'acordar'

    build_graph_run(source)

    # source = 'ntcir'
    source = 'acordar'
    cluster_run(source)


