# PROCS: Uncertainty-Aware Knowledge Graph Embeddings via Conceptual Spaces

Anonymous code and data release for the submission *"PROCS: Uncertainty-Aware Knowledge Graph Embeddings via Conceptual Spaces."*

PROCS represents entities, relations, and semantic types as Gaussian distributions in a conceptual space, with a multi-faceted geometric scoring function (MGS) and scalable, sampling-free inference. A single notebook (`procs.ipynb`) reproduces the link-prediction, triple-classification, and uncertainty results on all datasets; the dataset is selected via the configuration cell. We also release the build script and canonical split for the Defence-Wikidata knowledge graph.

## Repository structure

```
.
├── procs.ipynb                       # Full model + training + evaluation (all datasets)
├── wikidata_defense_extraction.ipynb # Build script for the Defence-Wikidata KG
├── data/
│   ├── FB15k-237/                    # train.txt / valid.txt / test.txt
│   ├── WN18RR/                       # train.txt / valid.txt / test.txt (+ WordNet license)
│   └── defense/                      # Canonical Defence-Wikidata split
│       ├── train.txt                 # 114,701 triples
│       ├── valid.txt                 #  14,338 triples
│       └── test.txt                  #  14,338 triples
├── requirements.txt
└── README.md
```

Every dataset uses the same format: each split file is tab-separated, one triple per line, `head <TAB> relation <TAB> tail`. The Defence-Wikidata split uses original Wikidata entity (`Q…`) and property (`P…`) identifiers.

## Setup

```bash
pip install -r requirements.txt
```

Experiments were run on a single NVIDIA RTX 6000 Ada GPU; a CUDA-capable GPU is recommended (the code falls back to CPU if none is available).

## Reproducing the results

`procs.ipynb` runs any dataset — set the dataset in the **configuration cell** (`DATASET SWITCH`) and run all cells:

| Dataset | `data_dir` | `dataset_name` | `margin` | `adversarial_temperature` |
|---|---|---|---|---|
| FB15k-237 | `data/FB15k-237` | `FB15k-237` | 9.0 | 2.0 |
| WN18RR | `data/WN18RR` | `WN18RR` | 6.0 | 1.0 |
| Defence-Wikidata | `data/defense` | `Defence-Wikidata` | — | — |

Other hyperparameters (embedding dimension 400, 256 self-adversarial negatives, effective batch size 1024, loss weights, cosine-annealing schedule) are fixed in the configuration cell.

All three datasets are included under `data/` in the same tab-separated format, so the notebook runs out of the box. FB15k-237 and WN18RR are the standard public benchmarks; WN18RR is distributed under the WordNet 3.0 license (`data/WN18RR/Wordnet3.0-LICENSE`). The Defence-Wikidata split is our contribution.

## Defence-Wikidata dataset

The Defence-Wikidata knowledge graph contains 68,826 defence-relevant entities and 143,532 triples across 355 relation types, extracted from Wikidata. `wikidata_defense_extraction.ipynb` documents the extraction pipeline; the canonical train/valid/test split used in all reported experiments is provided under `data/defense/`.

## License

Released under the MIT License for research use.
