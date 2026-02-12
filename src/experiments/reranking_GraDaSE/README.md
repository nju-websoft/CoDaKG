## Overview

This directory contains an optimized version of the [GraDaSE](https://github.com/nju-websoft/GraDaSE) code.

The code in this folder has been modified to improve execution efficiency.

* **Modified Files:** `model.py` and `run.py`
* **Changes:** Performance-related optimizations only. The **core functionality and logic** remain identical to the original implementation.

## Setup

* **Environment:** Follow the environment setup instructions provided in the original repository.
* **Data:** Prepare the dataset according to the original requirements and place it in:
`./data/CoDaKG_tags_annotators/`

## Usage

To run the optimized version, execute the following commands from the root of this directory:

```bash
cd code
bash CoDaKG.sh
```