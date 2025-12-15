# Evaluating Hybrid Sampling Strategies for Large Graphs

This repository contains our course project for Social Network Analysis. We reproduce and extend the evaluation framework of Leskovec & Faloutsos (2006) by implementing two-phase hybrid samplers that combine node selection (RN/RPN/RDN) with exploration (RW/FF), and evaluating them under the Scale-down and Back-in-Time objectives.

---
## Project Structure


```text
.
├── data/                 # downloaded datasets
├── results/              # experiment outputs 
├── figures/              # generated plots
├── src/
│   ├── data_loader.py    # load SNAP datasets
│   ├── samplers.py       # baseline + hybrid sampling implementations
│   ├── evaluator.py      # compute S1-S9 and KS D-statistics
│   ├── temporal_metrics.py  # compute T1-T5 for temporal sequences
│   ├── temporal_utils.py    # snapshot construction, temporal pipeline utils
│   ├── visualizer.py     # plot/table generation 
│   ├── experiment.py     # experiment runners (SD + BiT)
│   └── utils.py          # shared utilities
├── config.py             # global settings
├── download_data.py      # dataset download script
├── main.py               # CLI entrypoint
├── requirements.txt
└── README.md

```


---
## Environment Setup

```bash
python -m venv venv
source venv/bin/activate       
pip install -r requirements.txt
```

---

## Data

We use directed SNAP datasets.

### Scale-Down datasets (5)
- `cit-HepTh`, `cit-HepPh`, `soc-Epinions1`, `wiki-Vote`, `p2p-Gnutella31`

### Back-in-Time datasets (2)
- `cit-HepTh`, `cit-HepPh`

To download datasets:
```bash
python download_data.py
```
---

## Quick Start

### Quick test 
```bash
python main.py --quick
```

### Full Scale-Down experiment
Runs SD at sampling ratios `r ∈ {0.10, 0.15, 0.20}`, repeats 3 seeds, and writes aggregated CSV results.
```bash
python main.py --full
```

### Back-in-Time experiment
Constructs a sequence of historical snapshots and evaluates static + temporal metrics.
```bash
python main.py --temporal
```

---
### Results
Output tables are stored under `results/`.  

---
### Figures
Plots and formatted tables used in the report are stored under `figures/` .



---
## Reference

This project follows the experimental framing of:
- Leskovec, J. & Faloutsos, C. (2006). *Sampling from Large Graphs*.










