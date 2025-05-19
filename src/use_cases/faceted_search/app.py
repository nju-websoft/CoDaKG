# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# 示例原始数据集列表 (用于构建新的字典结构)
# 在实际应用中，你可能会从数据库或文件读取数据，然后构建成下面的字典结构

with open('data.json', 'r') as f:
    data = json.load(f)

dataset = data['dataset']
related_datasets = data['related_datasets']

facet_definitions = {
    "publisher": {"label": "publihser", "type": "categorical"},
    "created": {"label": "created", "type": "date"},
    "modified": {"label": "modified", "type": "date"},
    "license": {"label": "license", "type": "categorical"},
    "themes": {"label": "themes", "type": "categorical"},
}

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        return None

# --- API Endpoints ---

# 1. 获取分面项定义及选项的接口 (修改，遍历 dataset 字典的值)
@app.route('/facets', methods=['GET'])
def get_facets():
    """
    提供分面项的定义和选项列表。
    现在从 dataset 字典的值中统计信息。
    """
    facets_data = []

    # 遍历分面定义
    for key, definition in facet_definitions.items():
        facet_info = {
            "key": key,
            "label": definition["label"],
            "type": definition["type"],
            "options": []
        }

        if definition["type"] == "categorical":
            options_count = {}
            # *** 遍历 dataset 字典的值来统计选项 ***
            for item in dataset:
                value = item.get(key)
                if value is not None:
                    str_value = str(value)
                    options_count[str_value] = options_count.get(str_value, 0) + 1

            facet_info["options"] = [{"value": value, "count": count} for value, count in options_count.items()]

        elif definition["type"] == "numerical":
             facet_info["options"] = []

        elif definition["type"] == "date":
             facet_info["options"] = []

        facets_data.append(facet_info)

    return jsonify(facets_data)


# 2. 搜索和过滤数据的接口 (修改，遍历 dataset 字典的值并构建响应结构)
@app.route('/search', methods=['POST'])
def search_data():
    """
    接收前端发送的过滤条件和搜索查询，并返回符合条件的数据。
    响应结构包含数据集详情和相关数据集列表。
    """
    request_data = request.json
    filters = request_data.get('filters', {})
    query = request_data.get('query', '').strip()

    # 将 dataset 字典的值转换为列表，方便后续过滤和搜索
    current_results_list = dataset

    # --- 先应用分面过滤 ---
    for facet_key, filter_value in filters.items():
        if facet_key in facet_definitions:
            facet_type = facet_definitions[facet_key]["type"]

            if facet_type == "categorical":
                if isinstance(filter_value, list) and filter_value:
                     valid_filter_values = [str(v) for v in filter_value]
                     current_results_list = [
                         item for item in current_results_list
                         if str(item.get(facet_key)) in valid_filter_values
                     ]
                elif isinstance(filter_value, str):
                      current_results_list = [
                         item for item in current_results_list
                         if str(item.get(facet_key)) == str(filter_value)
                     ]

            elif facet_type == "numerical":
                 if isinstance(filter_value, dict):
                    min_val = filter_value.get('min')
                    max_val = filter_value.get('max')

                    is_min_valid = isinstance(min_val, (int, float)) or min_val is None
                    is_max_valid = isinstance(max_val, (int, float)) or max_val is None

                    if is_min_valid or is_max_valid:
                        current_results_list = [
                            item for item in current_results_list
                            if item.get(facet_key) is not None and isinstance(item.get(facet_key), (int, float)) and (
                                (min_val is None or item.get(facet_key) >= min_val) and
                                (max_val is None or item.get(facet_key) <= max_val)
                            )
                        ]

            elif facet_type == "date":
                 if isinstance(filter_value, dict):
                    start_date_str = filter_value.get('start_date')
                    end_date_str = filter_value.get('end_date')

                    start_date = parse_date(start_date_str)
                    end_date = parse_date(end_date_str)

                    if start_date is not None or end_date is not None:
                         current_results_list = [
                            item for item in current_results_list
                            if item.get(facet_key) is not None and isinstance(item.get(facet_key), str) and parse_date(item.get(facet_key)) is not None and (
                                (start_date is None or parse_date(item.get(facet_key)) >= start_date) and
                                (end_date is None or parse_date(item.get(facet_key)) <= end_date)
                            )
                        ]

    # --- 再应用搜索查询过滤 ---
    if query:
        # lower_query = query.lower()
        # text_fields_to_search = ["Title", "Method", "Contribution"]

        # current_results_list = [
        #     item for item in current_results_list
        #     if any(
        #         item.get(field) and isinstance(item.get(field), str) and lower_query in item.get(field).lower()
        #         for field in text_fields_to_search
        #     )
        # ]
        pass

    # --- 构建最终响应结构：为每个结果项添加 related_datasets 列表 ---
    final_response_data = []
    for item in current_results_list:
        item_id = str(item["id"])
        item_with_related = item.copy() # 复制一份，避免修改原始 dataset 字典

        # 从 related_datasets 字典中查找当前项的相关数据集
        related_items_dict = related_datasets.get(item_id, {})
        # 将内层字典的值（相关数据集详情）转换为列表
        item_with_related["related_datasets"] = list(related_items_dict.values())

        final_response_data.append(item_with_related)

    # 返回过滤和搜索后的数据 (包含 related_datasets)
    return jsonify(final_response_data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)