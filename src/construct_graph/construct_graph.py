from collections import defaultdict
import json
from graph_tool.all import Graph as GTGraph, load_graph # Renamed to avoid clash with rdflib.Graph
import graph_tool.all as gt
import pickle
import os
import pymysql
import re
from urllib.parse import quote, unquote, urlparse
from datetime import datetime

# RDFLib imports
from rdflib import Graph as RdfGraph, Literal, Namespace, BNode, Node, URIRef
from rdflib.namespace import RDF, XSD, FOAF, DCTERMS, DCAT, OWL

MYSQL_CONFIG = {
    "host": "...",
    "port": 3306,
    "user": "...",
    "password": "...",
    "database": "..."
}

KEYWORD_NUM = 15

# Define Namespaces globally for use in fetching functions too
BASE_NS = Namespace("https://github.com/nju-websoft/CoDaKG/blob/main/schema.owl#")
FOAF_NS = Namespace("http://xmlns.com/foaf/0.1/")
DCT_NS = Namespace("http://purl.org/dc/terms/")
DCAT_NS = Namespace("http://www.w3.org/ns/dcat#")
OWL_NS = Namespace("http://www.w3.org/2002/07/owl#")
WD_NS = Namespace("http://www.wikidata.org/entity/")
EUVOC_NS = Namespace("http://publications.europa.eu/resource/authority/data-theme/")
# XSD_NS is available from rdflib.namespace.XSD

class TypedGraphBuilder:
    def __init__(self):
        self.graph = GTGraph(directed=True) # Use GTGraph
        self.node_type_map = {}
        self.node_reverse_map = {}
        self.vertex_prop_type = self.graph.new_vertex_property("int")
        self.vertex_prop_local_id = self.graph.new_vertex_property("string")
        self.edge_type = self.graph.new_edge_property("int")
        self.edge_weight = self.graph.new_edge_property("float")
        self.node_type_to_int = {}
        self.edge_type_to_int = {}
        self._next_node_type_int = 0
        self._next_edge_type_int = 0
        self.type_id_to_vertex = {}

    @staticmethod
    def load(graph_path: str, meta_path: str):
        print('Loading graph-tool graph and metadata...')
        builder = TypedGraphBuilder()
        builder.graph = load_graph(graph_path) # graph-tool load
        builder.vertex_prop_type = builder.graph.vertex_properties["type"]
        builder.vertex_prop_local_id = builder.graph.vertex_properties["local_id"]
        builder.edge_type = builder.graph.edge_properties["type"]
        builder.edge_weight = builder.graph.edge_properties["weight"]
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
            builder.node_type_map = metadata["node_type_map"]
            builder.node_reverse_map = metadata["node_reverse_map"]
            builder.node_type_to_int = metadata["node_type_to_int"]
            builder.edge_type_to_int = metadata["edge_type_to_int"]
            if builder.node_type_to_int:
                 builder._next_node_type_int = max(builder.node_type_to_int.values()) + 1
            else:
                 builder._next_node_type_int = 0
            if builder.edge_type_to_int:
                 builder._next_edge_type_int = max(builder.edge_type_to_int.values()) + 1
            else:
                 builder._next_edge_type_int = 0
            builder.type_id_to_vertex = {
                k: builder.graph.vertex(v_id)
                for k, v_id in builder.node_type_map.items()
            }
        return builder

    def graph_view_by_filter_e_type(self, e_type_str_to_filter="keyphrase"):
        if e_type_str_to_filter in self.edge_type_to_int:
            e_type_int_to_filter = self.edge_type_to_int[e_type_str_to_filter]
            filter_cond = self.edge_type.a != e_type_int_to_filter
            return gt.GraphView(self.graph, efilt=filter_cond)
        print(f"Warning: Edge type '{e_type_str_to_filter}' not found in edge_type_to_int map. Returning unfiltered graph.")
        return self.graph


def parse_ntcir_date_to_xsd_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%B %d, %Y").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None

def extract_q_number(wikidata_uri):
    if not wikidata_uri: return None
    pattern = r'(Q\d+)(?:[#/]|$)'
    match = re.search(pattern, wikidata_uri)
    return match.group(1) if match else None

# --- New functions to fetch structured data from MySQL ---
def fetch_all_node_properties_from_db(source_dataset_name: str):
    """
    Fetches all properties for all relevant nodes from the database
    and returns them in a structured dictionary.
    Key: (node_type_str, local_id_str)
    Value: dict of {property_uri: [value1, value2...]} where value is Literal or URIRef
    """
    print(f'Fetching raw properties from DB for source: {source_dataset_name}...')
    all_props = defaultdict(lambda: defaultdict(list))
    connection = pymysql.connect(**MYSQL_CONFIG)
    cursor = connection.cursor(pymysql.cursors.DictCursor) # Use DictCursor

    # Fetch Dataset properties
    node_type_str = 'dataset'
    if source_dataset_name == 'ntcir':
        sql = "SELECT t1.dataset_id, t1.title, t1.description, t2.creator, t2.date_created, t2.date_modified, t2.homepage, t2.license FROM ntcir_metadata as t1 JOIN ntcir_metadata_aug as t2 ON t1.dataset_id=t2.dataset_id"
        cursor.execute(sql)
        for item in cursor.fetchall():
            key = (node_type_str, item['dataset_id'])
            if item['title']: all_props[key][DCT_NS.title].append(Literal(item['title']))
            if item['description']: all_props[key][DCT_NS.description].append(Literal(item['description']))
            if item['creator']: all_props[key][DCT_NS.creator].append(Literal(item['creator']))
            created_date = parse_ntcir_date_to_xsd_date(item['date_created'])
            if created_date: all_props[key][DCT_NS.created].append(Literal(created_date, datatype=XSD.date))
            modified_date = parse_ntcir_date_to_xsd_date(item['date_modified'])
            if modified_date: all_props[key][DCT_NS.modified].append(Literal(modified_date, datatype=XSD.date))
            if item['homepage']: all_props[key][FOAF_NS.homepage].append(URIRef(item['homepage']))
            if item['license']: all_props[key][DCT_NS.license].append(URIRef(item['license']))
    elif source_dataset_name == 'acordar':
        sql = "SELECT t1.dataset_id, t1.title, t1.description, t2.creator, t2.date_created, t2.date_modified, t2.homepage, t2.license FROM acordar_datasets as t1 JOIN acordar_metadata_aug as t2 ON t1.dataset_id=t2.dataset_id"
        cursor.execute(sql)
        for item in cursor.fetchall():
            key = (node_type_str, item['dataset_id'])
            if item['title']: all_props[key][DCT_NS.title].append(Literal(item['title']))
            if item['description']: all_props[key][DCT_NS.description].append(Literal(item['description']))
            if item['creator']: all_props[key][DCT_NS.creator].append(Literal(item['creator']))
            if item['date_created']: all_props[key][DCT_NS.created].append(Literal(item['date_created'], datatype=XSD.date)) # Assuming YYYY-MM-DD
            if item['date_modified']: all_props[key][DCT_NS.modified].append(Literal(item['date_modified'], datatype=XSD.date)) # Assuming YYYY-MM-DD
            if item['homepage']: all_props[key][FOAF_NS.homepage].append(URIRef(validate_and_fix_uri(item['homepage'])[0]))
            if item['license']: all_props[key][DCT_NS.license].append(URIRef(item['license']))

    # Fetch Datafile properties
    node_type_str = 'datafile'
    if source_dataset_name == 'ntcir':
        sql_kp = "SELECT file_id, keyphrase FROM ntcir_keyphrase"
        cursor.execute(sql_kp)
        temp_kp = {row['file_id']: ', '.join(json.loads(row['keyphrase'])[:KEYWORD_NUM]) for row in cursor.fetchall()}
        
        sql_df = "SELECT file_id, format, data_url FROM ntcir_datafile_aug"
        cursor.execute(sql_df)
        for item in cursor.fetchall():
            key = (node_type_str, str(item['file_id']))
            if item['format']: all_props[key][DCT_NS["format"]].append(Literal(item['format']))
            keywords = temp_kp.get(str(item['file_id']), '')
            if keywords: all_props[key][DCAT_NS.keyword].append(Literal(keywords))
            if item['data_url']: all_props[key][DCAT_NS.downloadURL].append(URIRef(validate_and_fix_uri(item['data_url'])[0]))
    elif source_dataset_name == 'acordar':
        sql_kp = "SELECT file_id, keyphrase FROM acordar_keyphrase" # Assuming acordar_keyphrase.file_id matches acordar_datasets.dataset_id for datafile
        cursor.execute(sql_kp)
        temp_kp = {row['file_id']: ', '.join(json.loads(row['keyphrase'])[:KEYWORD_NUM]) for row in cursor.fetchall()}

        sql_df = "SELECT dataset_id, download FROM acordar_datasets" # 'rdf' format is implicit or defined by type.
        cursor.execute(sql_df)
        for item in cursor.fetchall():
            # For Acordar, dataset_id is used as datafile_id as per previous logic for keyphrases and downloadURLs
            key = (node_type_str, str(item['dataset_id']))
            all_props[key][DCT_NS["format"]].append(Literal("RDF")) # Or other appropriate format literal
            keywords = temp_kp.get(str(item['dataset_id']), '')
            if keywords: all_props[key][DCAT_NS.keyword].append(Literal(keywords))
            if item['download']:
                try:
                    download_urls = eval(item['download']) # eval is risky! Consider json.loads if it's a JSON string array
                    if isinstance(download_urls, list):
                        download_urls_new = []
                        for url in download_urls:
                            download_urls_new.extend(validate_and_fix_uri(url))
                        for url in download_urls_new:
                            if url: all_props[key][DCAT_NS.downloadURL].append(URIRef(url))
                    elif isinstance(download_urls, str) and download_urls: # single URL string
                         all_props[key][DCAT_NS.downloadURL].append(URIRef(download_urls))
                except Exception as e:
                    print(f"Warning: Could not parse download URLs for {key}: {item['download']} (Error: {e})")
                    if isinstance(item['download'], str) and item['download'].startswith("http"): # Failsafe for single URL string
                        all_props[key][DCAT_NS.downloadURL].append(URIRef(item['download']))

    # Fetch Publisher properties
    node_type_str = 'publisher'
    sql = f"SELECT id, label, sameAs, url FROM {source_dataset_name}_publisher WHERE label != ''"
    cursor.execute(sql)
    for item in cursor.fetchall():
        key = (node_type_str, str(item['id']))
        if item['label']: all_props[key][FOAF_NS.name].append(Literal(item['label']))
        q_num = extract_q_number(item['sameAs'])
        if q_num: all_props[key][OWL_NS.sameAs].append(WD_NS[q_num])
        if item['url']:
            url_str = item['url'] if item['url'].startswith('http') else f'https://{item["url"]}'
            url_str = validate_and_fix_uri(url_str)[0]
            all_props[key][FOAF_NS.homepage].append(URIRef(url_str))

    cursor.close()
    connection.close()
    print(f"Finished fetching raw properties. Found properties for {len(all_props)} nodes.")
    return all_props

def gt_node_to_uri(node_type_str: str, local_id: str):
    # Uses globally defined Namespaces
    if node_type_str == 'dataset':
        return BASE_NS[f"Dataset_{local_id}"]
    elif node_type_str == 'datafile':
        return BASE_NS[f"Distribution_{local_id}"]
    elif node_type_str == 'publisher':
        return BASE_NS[f"Publisher_{local_id}"]
    elif node_type_str == 'theme':
        return EUVOC_NS[local_id] # Assuming local_id is the direct code for theme
    return None

# --- Main export function using RDFLib thoroughly ---
def export_to_turtle_rdflib_centric(source_dataset_name: str):
    print(f"Starting RDFLib-centric export for: {source_dataset_name}")
    
    # Prepare paths
    save_dir = f'data/cluster/{source_dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    graph_path = os.path.join(save_dir, 'graph_all.gt')
    meta_path = os.path.join(save_dir, 'meta_all.pkl')
    output_path = os.path.join(save_dir, f"{source_dataset_name}_full_rdflib.ttl")

    if not os.path.exists(graph_path) or not os.path.exists(meta_path):
        print(f"Error: graph_all.gt or meta_all.pkl not found in {save_dir}")
        print("Please ensure these files exist. You might need to run a previous step to generate them.")
        # Create dummy files for basic script execution if needed for testing structure
        # (Similar dummy file creation as in previous version could be added here if essential for standalone run)
        if not os.path.exists(graph_path) and not os.path.exists(meta_path) :
            print(f"Attempting to create minimal dummy graph_all.gt and meta_all.pkl for {source_dataset_name} for script to run.")
            g_dummy = gt.Graph(directed=True)
            v1 = g_dummy.add_vertex()
            g_dummy.vertex_properties["type"] = g_dummy.new_vertex_property("int")
            g_dummy.vertex_properties["local_id"] = g_dummy.new_vertex_property("string")
            g_dummy.edge_properties["type"] = g_dummy.new_edge_property("int")
            g_dummy.edge_properties["weight"] = g_dummy.new_edge_property("float")
            
            g_dummy.vertex_properties["type"][v1] = 0 
            g_dummy.vertex_properties["local_id"][v1] = "dummy_dataset_1"
            g_dummy.save(graph_path)
            
            dummy_meta = {
                "node_type_map": {("dataset", "dummy_dataset_1"): 0},
                "node_reverse_map": {0: ("dataset", "dummy_dataset_1")},
                "node_type_to_int": {"dataset": 0, "datafile":1, "publisher":2, "theme":3, "keyphrase_node_type_placeholder": 4}, # Added placeholder
                "edge_type_to_int": {"Replica":0, "dump":1, "keyphrase":2, "publish": 3, "theme": 4, "content":5, "pattern":6, "version":7, "subset":8, "variant":9} 
            }
            with open(meta_path, 'wb') as f_meta:
                pickle.dump(dummy_meta, f_meta)
            print(f"Created dummy graph_all.gt and meta_all.pkl in {save_dir}")
        # return # Exiting if files are truly missing and not dummied might be safer depending on workflow.

    # 1. Load graph structure from graph-tool
    tgb = TypedGraphBuilder.load(graph_path, meta_path)
    gt_graph_view = tgb.graph_view_by_filter_e_type("keyphrase") # Filter out keyphrase edges

    # 2. Fetch all node properties from DB into a structured format
    # This dictionary stores data fetched from MySQL
    db_node_properties = fetch_all_node_properties_from_db(source_dataset_name)

    # 3. Initialize RDFLib Graph and bind namespaces
    rdf_g = RdfGraph()
    rdf_g.bind("base", BASE_NS)
    rdf_g.bind("foaf", FOAF_NS)
    rdf_g.bind("dct", DCT_NS)
    rdf_g.bind("dcat", DCAT_NS)
    rdf_g.bind("owl", OWL_NS)
    rdf_g.bind("wd", WD_NS)
    rdf_g.bind("euvoc", EUVOC_NS)
    rdf_g.bind("xsd", XSD)

    # Invert gt type maps for easy lookup
    int_to_node_type_str = {v: k for k, v in tgb.node_type_to_int.items()}
    int_to_edge_type_str = {v: k for k, v in tgb.edge_type_to_int.items()}

    # Map edge type strings (from gt) to RDF predicate URIs
    edge_type_to_rdf_predicate = {
        "Replica": BASE_NS.replica, "Subset": BASE_NS.subset,
        "Variant": BASE_NS.variant, "Version": BASE_NS.version,
        "dump": DCAT_NS.distribution, "publish": DCT_NS.publisher,
        "theme": DCAT_NS.theme, "content": BASE_NS.dataOverlap,
        "pattern": BASE_NS.schemaOverlap,
        # "keyphrase" edges are filtered out by gt_graph_view by default
    }

    print("Processing graph-tool nodes and edges to build RDFLib graph...")
    processed_nodes_count = 0
    # 4. Iterate through graph-tool nodes, create RDF counterparts, and add DB properties
    for v_gt in gt_graph_view.vertices(): # v_gt is a graph-tool Vertex object
        v_idx = int(v_gt) # Get integer index if needed by node_reverse_map
        
        # Original way to get type and id:
        # node_type_int_prop = tgb.vertex_prop_type[v_gt]
        # local_id_prop = tgb.vertex_prop_local_id[v_gt]
        # node_type_str = int_to_node_type_str.get(node_type_int_prop)
        
        # Using node_reverse_map which might be more direct if v_idx is the global_id
        if v_idx not in tgb.node_reverse_map:
            # This can happen if the graph view filters vertices in a way not aligned with node_reverse_map
            # Or if node_reverse_map is not exhaustive for all vertices in the view.
            # Fallback to property maps directly from the vertex descriptor v_gt:
            node_type_int_prop = tgb.vertex_prop_type[v_gt]
            local_id_prop = tgb.vertex_prop_local_id[v_gt]
            node_type_str = int_to_node_type_str.get(node_type_int_prop)
            if not node_type_str:
                 print(f"Warning: Vertex {v_idx} has type int {node_type_int_prop} not in int_to_node_type_str map. Skipping.")
                 continue
        else:
            node_type_str, local_id_prop = tgb.node_reverse_map[v_idx]


        if not node_type_str or local_id_prop is None:
            print(f"Warning: Vertex {v_idx} has missing type string or local_id. Type: {node_type_str}, ID: {local_id_prop}. Skipping.")
            continue

        subject_uri = gt_node_to_uri(node_type_str, local_id_prop)
        if not subject_uri:
            print(f"Warning: Could not create URI for node ({node_type_str}, {local_id_prop}). Skipping.")
            continue

        # Add rdf:type
        if node_type_str == 'dataset': rdf_g.add((subject_uri, RDF.type, DCAT_NS.Dataset))
        elif node_type_str == 'datafile': rdf_g.add((subject_uri, RDF.type, DCAT_NS.Distribution))
        elif node_type_str == 'publisher': rdf_g.add((subject_uri, RDF.type, FOAF_NS.Agent))
        # 'theme' nodes are usually objects, not subjects with these types. If they need types, add here.

        # Add properties from the pre-fetched DB dictionary
        props_from_db = db_node_properties.get((node_type_str, local_id_prop))
        if props_from_db:
            for predicate_uri, obj_list in props_from_db.items():
                for obj_val in obj_list: # obj_val is already Literal or URIRef
                    if str(predicate_uri) == 'https://www.Markus.Von Prause@ecy.wa.gov' or str(obj_val) == 'https://www.Markus.Von Prause@ecy.wa.gov':
                        print((node_type_str, local_id_prop))
                        print((subject_uri, predicate_uri, obj_val))
                    rdf_g.add((subject_uri, predicate_uri, obj_val))
        
        processed_nodes_count +=1
        if processed_nodes_count % 10000 == 0:
            print(f"  Processed {processed_nodes_count} nodes for properties...")


    print("Adding relationships (edges) to RDFLib graph...")
    processed_edges_count = 0
    # 5. Iterate through graph-tool edges and add them as RDF triples
    #    The view gt_graph_view already filters out 'keyphrase' edges.
    for e_gt in gt_graph_view.edges(): # e_gt is a graph-tool Edge object
        source_v_gt = e_gt.source()
        target_v_gt = e_gt.target()

        # Similar logic to get type/id for source and target
        source_v_idx = int(source_v_gt)
        if source_v_idx not in tgb.node_reverse_map:
            source_node_type_int = tgb.vertex_prop_type[source_v_gt]
            source_local_id = tgb.vertex_prop_local_id[source_v_gt]
            source_node_type_str = int_to_node_type_str.get(source_node_type_int)
        else:
            source_node_type_str, source_local_id = tgb.node_reverse_map[source_v_idx]
        
        target_v_idx = int(target_v_gt)
        if target_v_idx not in tgb.node_reverse_map:
            target_node_type_int = tgb.vertex_prop_type[target_v_gt]
            target_local_id = tgb.vertex_prop_local_id[target_v_gt]
            target_node_type_str = int_to_node_type_str.get(target_node_type_int)
        else:
            target_node_type_str, target_local_id = tgb.node_reverse_map[target_v_idx]

        if not source_node_type_str or source_local_id is None or \
           not target_node_type_str or target_local_id is None:
            print(f"Warning: Edge {e_gt} has source/target with missing type or ID. Skipping.")
            continue
            
        s_uri = gt_node_to_uri(source_node_type_str, source_local_id)
        o_uri = gt_node_to_uri(target_node_type_str, target_local_id)

        edge_type_int = tgb.edge_type[e_gt]
        edge_type_str = int_to_edge_type_str.get(edge_type_int)

        if not s_uri or not o_uri:
            print(f"Warning: Could not form URIs for edge between ({source_node_type_str},{source_local_id}) and ({target_node_type_str},{target_local_id}). Skipping edge.")
            continue

        if edge_type_str and edge_type_str in edge_type_to_rdf_predicate:
            predicate_rdf_uri = edge_type_to_rdf_predicate[edge_type_str]
            rdf_g.add((s_uri, predicate_rdf_uri, o_uri))
        elif edge_type_str != "keyphrase": # keyphrase should already be filtered, but double check
             print(f"Warning: Edge type '{edge_type_str}' (int: {edge_type_int}) has no mapped RDF predicate or is unexpected. Skipping edge {s_uri} -> {o_uri}.")
        
        processed_edges_count += 1
        if processed_edges_count % 100000 == 0:
            print(f"  Processed {processed_edges_count} edges...")


    # 6. Serialize the RDFLib graph
    print(f"\nSerializing RDFLib graph to {output_path}...")
    try:
        rdf_g.serialize(destination=output_path, format="turtle", encoding="utf-8")
        print(f"Successfully serialized {len(rdf_g)} RDF triples to {output_path}")
    except Exception as e:
        print(f"Error during RDFLib serialization: {e}")



def validate_and_fix_uri(uri_str):
    if ',' in uri_str or '\n' in uri_str:
        uris = [u.strip() for u in re.split(r'[,\s]+', uri_str) if u.strip()]
        valid_uris = [u for u in uris if u.startswith('http')]
        # raise ValueError("Multiple URIs detected, need to split first")
        return valid_uris
    
    uri_str = uri_str.strip()
    uri_str = quote(uri_str, safe=':/?&=')
    
    parsed = urlparse(uri_str)
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"Invalid URI structure: {uri_str}")
    
    return [uri_str]


if __name__ == "__main__":
    # Ensure MYSQL_CONFIG is correct and DB is accessible.
    # Ensure graph_all.gt and meta_all.pkl exist for the chosen source.
    
    # source = 'ntcir'
    source = 'acordar'
    
    # export_to_turtle_rdflib_centric(source)

    source_dataset_name = source
    save_dir = f'data/cluster/{source_dataset_name}'
    output_path = os.path.join(save_dir, f"{source_dataset_name}_full_rdflib.ttl")
    rdf_g = RdfGraph()
    rdf_g.parse(output_path, format="turtle")
    print(len(rdf_g))


