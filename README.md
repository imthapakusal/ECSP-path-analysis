# Edge-Central Shortest Paths (ECSP)

This repository contains a complete experiment pipeline for studying the **Edge-Central Shortest Path (ECSP)** problem — which selects, among all shortest *s–t* paths, the one that maximizes cumulative edge centrality.

ECSP is evaluated using four edge-centrality measures:

- **EBC** — Edge Betweenness Centrality  
- **ECL** — Edge Closeness Centrality (via line graph)  
- **GRAV** — Gravity-style centrality (sampled all-pairs SP usage)  
- **ECHO** — ECHO edge centrality (linear system)

The repository includes:

---

## 🔹 1. Core ECSP Implementation
- Shortest-path DAG construction (implicitly via BFS layers)  
- Dynamic programming solver for ECSP  
- Normalization utilities  
- Path reconstruction  

---

## 🔹 2. Centrality Computation Modules
- Edge betweenness via igraph  
- Edge closeness through the line graph  
- Sampled gravity centrality  
- ECHO centrality implementation  

All centrality scores are normalized for comparison.

---

## 🔹 3. Synthetic Graph Experiments  
ECSP experiments on:
- **Erdős–Rényi (ER)** graphs  
- **Barabási–Albert (BA)** preferential attachment graphs  

For each graph:
- Sampled source–target pairs  
- Compute ECSP paths for all four centralities  
- Path agreement and score-difference analysis  
- Path-overlap metrics (Jaccard edge overlap)  
- Attack-robustness experiments:
  - Remove edges in decreasing ECSP usage frequency
  - Evaluate effects on:
    - Global efficiency  
    - Giant component fraction  

Results saved as CSV + figures.

---

## 🔹 4. Real Network Experiments  
Dataset loaders + ECSP experiments for:
- **CA-GrQc** (collaboration network)  
- **Email-Eu-core** (symmetrized)  
- **US Power Grid**  

Includes:
- ECSP computation  
- Path-usage based robustness experiments  
- Agreement matrices  
- Score-difference heatmaps  
- Centrality-correlation matrices  
- Path-overlap metrics  
- Robustness curves  

---

## 🔹 5. Analysis & Plotting  
The code produces:
- Agreement heatmaps  
- Score-difference matrices  
- Jaccard path-overlap matrices  
- Robustness curves (efficiency + GCC)  
- Combined cross-network robustness figure  

All figures are reproducible.

---

## 🔹 6. Repository Structure

ECSP-path-analysis/
│
├── src/
│ ├── ecsp_full_pipeline.py # Main experiment script
│ ├── utils/ # Centrality + robustness utilities
│ └── plots/ # Figure generation
│
├── datasets/
│ ├── CA-GrQc.txt
│ ├── email-Eu-core.txt
│ └── powergrid.txt
│
├── requirements.txt
└── README.md ← You are reading this


---

## 🔹 7. Dependencies
Install via:

```bash
pip install -r requirements.txt
```
## 🔹 8. Running the Experiments

Run all synthetic + real datasets + plots:
python src/ecsp_full_pipeline.py

This will generate:

-CSV files for ECSP and robustness results
-All plots in the working directory

## 🔹 9. Citation

If using this code in research, please cite:
Kusal Thapa, “Edge-Central Shortest Paths: A Comparative Evaluation of Edge Centrality Measures Through Path Selection and Robustness Analysis.”

## 🔹 10. Author

Kusal Thapa
MSc Mathematics
Tribhuvan University
(Currently preparing for PhD research in optimization and network science)
