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

The code used to construct the CoDaKG instances is provided in the [./sec](./src) directory. This includes scripts and modules for:


### Dependencies

To run the code, ensure you have the following dependencies installed:
- Python 3.10
- rank-bm25
- RDFLib
- transformers
- FlagEmbedding


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.