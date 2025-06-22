# CoDaKG: Content-Based Dataset Knowledge Graphs for Dataset Search


This repository contains the code, schema, and resources for **CoDaKG (Content-Based Dataset Knowledge Graphs)**, a project focused on enriching dataset search by modeling fine-grained attributes and semantic relationships derived from both dataset metadata and their actual data file content.

We provide:
1. [CoDaKG instances](https://zenodo.org/records/15398145) for two widely used dataset search test collections: NTCIR and ACORDAR.
1. Source code for constructing CoDaKG in this github repository.
2. An extended DCAT ontology ([schema.owl](schema.owl)) defining the custom relationships used in CoDaKG.


## CoDaKG Instances

The generated CoDaKG instances for the NTCIR and ACORDAR test collections are available as RDF dumps (Turtle format). These resources explicitly model datasets, their distributions, publishers, themes, and various inter-dataset relationships.

*   **Access:** Archived on Zenodo.
    *   **DOI:** `10.5281/zenodo.15398145`
    *   **Direct Link:** [https://zenodo.org/records/15398145](https://zenodo.org/records/15398145)
*   **Statistics:** Detailed statistics for each CoDaKG instance can be found in our paper.

Table 1: Predicate Usage Statistics

| Predicate                          | NTCIR-CoDaKG   | ACORDAR-CoDaKG |
|------------------------------------|----------------|----------------|
| base:dataOverlap                   | 74,395,056     | 6,116          |
| base:variant                       | 41,804,226     | 121,548        |
| base:schemaOverlap                 | 1,761,094      | 45,466         |
| dcat:distribution                  | 92,930         | 31,589         |
| dct:title                          | 46,615         | 31,589         |
| rdf:type                           | 139,725        | 65,718         |
| dct:description                    | 46,613         | 27,244         |
| dcat:downloadURL                   | 92,930         | 34,285         |
| dct:publisher                      | 124,777        | 19,375         |
| dcat:theme                         | 24,831         | 35,305         |
| dcat:keyword                       | 74,019         | 31,573         |
| foaf:homepage                      | 29,910         | 23             |
| dct:format                         | 92,930         | 31,589         |
| dct:created                        | 46,615         | 31,532         |
| dct:modified                       | 46,615         | 19,265         |
| dct:license                        | 3,519          | 3,220          |
| base:replica                       | 1,184          | 11,618         |
| foaf:name                          | 180            | 2,540          |
| dct:creator                        | 590            | -              |
| owl:sameAs                         | 126            | 904            |
| base:subset                        | 62             | 28             |
| base:version                       | 26             | 12             |
| **Total Triples**                  | **118,824,573**| **550,539**    |

Table 2: Term Statistics

| Term                     | NTCIR-CoDaKG                           | ACORDAR-CoDaKG                         |
|--------------------------|----------------------------------------|----------------------------------------|
| Typed IRIs               | 139,725                                | 65,718                                 |
| • `foaf:Agent`           | 180                                    | 2,540                                  |
| • `dcat:Distribution`    | 92,930                                 | 31,589                                 |
| • `dcat:Dataset`         | 46,615                                 | 31,589                                 |
| Untyped IRIs             | 114,175                                | 34,385                                 |
| Literals                 | 136,855                                | 78,048                                 |
| **Total Terms**          | **390,755**                            | **178,151**                            |


## CoDaKG Construction Code

The code used to construct the CoDaKG is provided in the [./sec/construct_graph/](./src/construct_graph/) directory. 
- `metadata` directory: Metadata Enrichment and Metadata-Based Relationship Discovery
- `content` directory: Content-Based Property Population
- `contruct_graph.py` file: Construct CoDaKG

The use cases code is provided in the [./sec/use_cases/](./src/use_cases/) directory.
- `retrieval_with_enrichment.py` file: Ad Hoc Dataset Retrieval with Enriched Metadata
- `graph_based_reranking` directory: Graph-Based Dataset Re-Ranking
- `cluster.py` file and `faceted_search` directory: Exploratory Dataset Search

### Dependencies

To run the code, ensure you have the following dependencies installed:
- Python 3.10
- rank-bm25
- pymysql
- RDFLib
- transformers
- FlagEmbedding
- ydf
- sentence-transformers
- scikit-learn
- graph-tool
- pandas

---

The following section outlines the main components and steps involved in building a CoDaKG.

### 1. Metadata Enrichment

This phase focuses on processing dataset metadata to extract core properties, link entities to external knowledge bases, and classify datasets by theme.

#### 1.1. Entity Linking (Publisher and License)

We link dataset publishers and licenses (from metadata) to Wikidata entities.

*   **Method 1: Wikidata API Search**
    The script [src/construct_graph/metadata/entity_linking_plain.py](src/construct_graph/metadata/entity_linking_plain.py) queries the Wikidata API to find matching entities.
    ```python
    # Example usage from src/construct_graph/metadata/entity_linking_plain.py
    import requests
    base_url = "https://www.wikidata.org/w/api.php"
    name = "Fish and Wildlife Service" # Example publisher string
    params = {
        "action": "wbsearchentities", "format": "json", "search": name,
        "language": "en", "type": "item", "limit": 1,
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if data['search']:
        wikidata_id = data['search'][0]['concepturi']
    ```

*   **Method 2: LLM-based URL Inference**
    If no direct match is found via the API, [src/construct_graph/metadata/entity_linking_aiprompt.py](src/construct_graph/metadata/entity_linking_aiprompt.py) uses a Large Language Model (ChatGPT o3-mini-high via OpenAI API) to infer a Wikidata URL or an official website URL from the entity name. The inferred URL is then validated.
    ```python
    # Example usage from src/construct_graph/metadata/entity_linking_aiprompt.py
    import openai # Ensure your OpenAI API key is configured
    label = "Example Organization Name"
    url = get_wikidata_url(label) # Function call
    ```

#### 1.2. Theme Classification

Datasets are classified into EU Vocabularies data themes.

*   **Option A: Use Pre-trained Model**
    Download the pre-trained theme classification model from [data/models/theme_classification.zip](data/models/theme_classification.zip).
*   **Option B: Train Your Own Model**
    1.  Download the training data [data/datasets/eu_datasets.zip](data/datasets/eu_datasets.zip).
    2.  Place the unzipped data in the same directory as [src/construct_graph/metadata/subject_training.py](src/construct_graph/metadata/subject_training.py).
    3.  Install Scikit-learn: `conda install -c conda-forge scikit-learn`
    4.  Run training: [src/construct_graph/metadata/subject_training.py](src/construct_graph/metadata/subject_training.py)
*   **Apply Classification:**
    Run [src/construct_graph/metadata/subject_eval.py](src/construct_graph/metadata/subject_eval.py) to load the trained model and classify datasets. You'll need to modify the script to point to your dataset metadata file (CSV with `dataset_id`, `title`, `description` fields).
    ```bash
    # Modify input CSV path in subject_eval.py first
    python src/construct_graph/metadata/subject_eval.py
    ```

### 2. Metadata-Based Relationship Discovery

This step identifies provenance-related relationships (e.g., `base:replica`, `base:version`) between datasets based on their metadata.

**Dependencies:**
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install conda-forge::transformers
pip install ydf -U
pip install levenshtein pymysql
```

#### 2.1. Train Relationship Model

*   **Option A: Use Pre-trained GBDT Model**
    Download the pre-trained model from [data/models/metadata_relationship.zip](data/models/metadata_relationship.zip). The unzipped directory will be your `GBDT_MODEL_DIR`.
*   **Option B: Train Your Own GBDT Model**
    1.  Prepare your relationship training data as a CSV file (e.g., `relationships.csv`) with columns `dataset_id1`, `dataset_id2`, `relationship`. Our training data was derived from [K. Lin et al.](https://figshare.com/articles/dataset/Metadata_for_Datasets_and_Relationships/22790810) as described in our paper.
    2.  Set `RELATIONSHIPS_CSV_PATH` in [src/construct_graph/metadata/relationship_training.py](src/construct_graph/metadata/relationship_training.py) to your CSV file.
    3.  Run training: `python src/construct_graph/metadata/relationship_training.py`. The model will be saved to `GBDT_MODEL_DIR` defined in the script.

#### 2.2. Discover Relationships

1.  Prepare your dataset metadata as a CSV file (e.g., `datasets_acordar.csv`) with columns `dataset_id`, `title`, `description`.
2.  Set `DATASET_CSV_PATH` in [src/construct_graph/metadata/relationship.py](src/construct_graph/metadata/relationship.py) to your metadata file. Ensure `GBDT_MODEL_DIR` in this script points to your trained or pre-trained model.
3.  Run discovery: `python src/construct_graph/metadata/relationship.py`. Results are saved to `OUTPUT_CSV_PATH` defined in the script.

#### 2.3. Refine Relationships with Heuristics

Apply heuristic rules to refine the discovered relationships:
```bash
python src/construct_graph/metadata/relationship_refine.py
```
This script will take the output from the previous step and produce a refined CSV of relationships.

### 3. Content Parsing and Categorization

This phase processes the actual data files associated with dataset distributions.

#### 3.1. File Format Detection

**Dependencies:** `pip install pymysql python-magic`

The `detect_file_format(filenames: list)` function in [src/construct_graph/content/detect_format.py](src/construct_graph/content/detect_format.py) detects the MIME type of files.
```python
# Example from src/construct_graph/content/detect_format.py
filenames = ["path/to/your/file1.html", "path/to/your/file2.docx"]
results = detect_file_format(filenames) # Returns a dict: {filepath: format_string}
# e.g. {"path/to/your/file1.html": "html", file2.docx: "docx", ...}
```

#### 3.2. Content Extraction (Textualization)

**Dependencies:** Install [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract), then `pip install PyPDF2 pdfminer.six pdfplumber pytesseract pdf2image`.

*   **HTML to TXT:** Use `convert_html_to_txt(file_path, target_path)` from [src/construct_graph/content/html_doc_docx2txt.py](src/construct_graph/content/html_doc_docx2txt.py).
*   **DOCX to TXT:** Use `convert_docx_to_txt(file_path, target_path)` from [src/construct_graph/content/html_doc_docx2txt.py](src/construct_graph/content/html_doc_docx2txt.py).
*   **PDF to TXT:** Use `process_pdf(filename, gpu_id)` from [src/construct_graph/content/pdf2txt.py](src/construct_graph/content/pdf2txt.py). Modify `INPUT_FOLDER` and `OUTPUT_FOLDER` in the script.
*   **RDF Files:** For parsing RDF files into a structured format suitable for later processing, we reference the methods used in the [VOYAGE project](https://github.com/nju-websoft/VOYAGE).

### 4. Content-Based Keyword Extraction

**Dependencies:**
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install conda-forge::transformers
```
The script [src/construct_graph/content/extract_keywords.py](src/construct_graph/content/extract_keywords.py) contains functions:
*   `clean_text_file`, `clean_table_file`, `clean_json_file`, `clean_xml_file` for preparing text from different file formats.
    ```python
    # Example from src/construct_graph/content/extract_keywords.py
    file_path = "path/to/your/file.txt" # or other format
    file_size = os.path.getsize(file_path) # Check file size
    truncated = file_size > FILE_SIZE_THRESHOLD # Decide on truncation

    # Apply cleaning based on detected format
    if detect_format == "json":
        cleaned_text = clean_json_file(file_path, truncated=truncated)
    elif detect_format == "xml":
        cleaned_text = clean_xml_file(file_path, truncated=truncated)
    elif detect_format in {"xlsx", "xls"}:
        # safe_read_excel handles reading, then clean_table_file handles processing
        cleaned_text = clean_table_file(file_path) # Truncation handled within safe_read_excel if needed (not currently implemented there)
    elif detect_format == "csv":
            cleaned_text = clean_table_file(file_path, truncated=truncated)
    else: # pdf, txt, html, doc, docx treated as plain text
        with open(file_path, "r", encoding="utf8", errors="ignore") as f:
            content = f.read(FILE_SIZE_THRESHOLD) if truncated else f.read()
        cleaned_text = clean_text_file(content)
    ```
*   `extract_keyphrases` (using `bloomberg/KeyBART`) to extract keywords from cleaned text. Initialize workers with `init_worker(GPU_LIST)`.
    ```python
    # Example from src/construct_graph/content/extract_keywords.py
    MODEL_NAME = "bloomberg/KeyBART" # 用于提取关键词的模型
    GPU_LIST = [0, 1, 2, 3, 4] # List of GPU IDs to use
    init_worker(GPU_LIST)
    cleaned_text = "..." 
    keywords = extract_keyphrases(cleaned_text) # ["keyword1", "keyword2", ...]
    ```

### 5. Content-Based Relationship Discovery (Schema & Data Overlap)

This step identifies `base:schemaOverlap` and `base:dataOverlap` relationships.

#### 5.1. Schema Units Extraction

*   **RDF Files:** EDPs (Entity Description Patterns) serve as schema units. Methods for EDP extraction can be found in the [VOYAGE project](https://github.com/nju-websoft/VOYAGE).
*   **JSON/XML Files:** Use `extract_json_edp(file_path)` or `extract_xml_edp(file_path)` from `src/construct_graph/content/extract_schema_and_data_unit.py` to get sets of root-to-leaf paths.
    ```python
    # Example from src/construct_graph/content/extract_schema_and_data_unit.py
    if detect_format == 'json':
        edp_set = extract_json_edp(file_path)
    elif detect_format == 'xml':
        edp_set = extract_xml_edp(file_path)
    ```
*   **Tabular Files:** Column headers are schema units (details in paper). Methods can be found in [src/construct_graph/content/extract_table_header.py](src/construct_graph/content/extract_table_header.py).

#### 5.2. Data Units Extraction

*   **RDF Files:** MSGs (Minimum Self-contained Graphs) serve as data units. Methods for MSG extraction can be found in the [VOYAGE project](https://github.com/nju-websoft/VOYAGE).
*   **Textual Files:** Sentences are data units. Use `generate_text_sentence_data(folder_path)` from [src/construct_graph/content/extract_schema_and_data_unit.py](src/construct_graph/content/extract_schema_and_data_unit.py).
*   **JSON/XML Files:** Root-to-leaf paths paired with their values are data units. Use `generate_json_xml_content_data(file_infos)` from [src/construct_graph/content/extract_schema_and_data_unit.py](src/construct_graph/content/extract_schema_and_data_unit.py).
*   **Tabular Files:** Rows are data units. Use `generate_table_content_data(file_infos)` from [src/construct_graph/content/extract_schema_and_data_unit.py](src/construct_graph/content/extract_schema_and_data_unit.py).

```python
# Example from src/construct_graph/content/extract_schema_and_data_unit.py
folder_path = "path/to/your/txt_folder"
data_iterable = generate_text_sentence_data(folder_path) # iterator of (filename, data_unit_set)
file_infos = [("file_id", "json", "path/to/your/file.json"), ...] # file_id, detect_format, full_path
data_iterable = generate_json_xml_content_data(file_infos) # iterator of (file_id, data_unit_set)
```

#### 5.3. Compute Overlap using MinHash and LSH

**Dependencies:** `pip install datasketch`

1.  **Compute MinHash Signatures:**
    Use `compute_minhash_signatures(data_iterable, ...)` from [src/construct_graph/content/compute_overlap.py](src/construct_graph/content/compute_overlap.py). This function takes an iterable of `(item_id, item_data_units)` and parameters like `num_perm`, `num_processes`, `save_interval`, and paths for saving signatures and state.
    ```python
    # Conceptual usage from src/construct_graph/content/compute_overlap.py
    data_iterable = ... # Your iterator of (id, set_of_units)
    minhash_signatures = compute_minhash_signatures(data_iterable, ...)
    ```
2.  **Compute LSH Similarity:**
    Use `compute_lsh_similarity(minhash_signatures, save_path, num_perm, lsh_threshold)` from [src/construct_graph/content/compute_overlap.py](src/construct_graph/content/compute_overlap.py). Load the MinHash signatures generated in the previous step. The `lsh_threshold` is for LSH candidate pair generation. The output CSV contains `id1, id2, j_sim`.
    ```python
    # Conceptual usage from src/construct_graph/content/compute_overlap.py
    # ... load minhash_signatures ...
    lsh_similarity_results = compute_lsh_similarity(minhash_signatures, ...)
    ```
    The final Jaccard similarity threshold for creating `base:schemaOverlap` or `base:dataOverlap` edges is determined as described in our paper, potentially using `plot_threshold.ipynb` for analysis.

### 6. Constructing the Final CoDaKG RDF Graph

Finally, use [src/construct_graph/construct_graph.py](src/construct_graph/construct_graph.py) to combine all extracted metadata, enriched attributes, discovered relationships, and content-derived links into the final CoDaKG instances, serializing them as Turtle RDF files. 


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.