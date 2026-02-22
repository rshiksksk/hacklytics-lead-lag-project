# Dataset Analysis & Project Plan Synthesis Report

## 1. Context & Objective
Based on the `Project_Proposal.pdf` and `project_plan.txt`, the goal is to develop a **Hybrid-Temporal Graph Neural Network (HT-GNN)** to model lead-lag information diffusion in the Chinese A-share market. The core hypothesis is that economic shocks propagate through supply chain networks with a temporal delay. The objective is framed as a **three-class classification task** to predict whether a stock will **outperform**, **underperform**, or remain **neutral** relative to the equal-weighted market average over various horizons ($h = 1, 3, 5, 20$ days).

## 2. Dataset Analysis & Integration

### A. Supply Chain Network Dataset (`Supply_Chain_data.csv`)
This dataset acts as the **Dynamic Edge List** and network structure.
* **File Size:** ~6.3 MB (63,262 records)
* **Design Decision (from Plan):** The graph is directed, bidirectional (differentiating supplier->customer and customer->supplier rules), and incorporates both listed and unlisted firms.
* **Temporal Alignment:** The graph updates annually each May 1st. To prevent look-ahead bias, interactions dated December 31st of Year $Y-1$ are applied for the period starting May 1st of Year $Y$.
* **Edge Features:** Relationship Type and Partner Rank (1-5).
* **Handling Unlisted Nodes:** As planned, unlisted/private firms will only be represented by a 64-dimensional **Node2Vec structural embedding** since they lack daily market features.

### B. Daily Market Dataset (`LIQ_TOVER_D_combined.csv` + Additional Returns)
This dataset provides the **Time-Series Node Features** for the public entities.
* **Design Decision (from Plan):** A rolling window of $T=20$ past trading days is utilized.
* **Key Node Features for Listed Firms:**
  1. **Abnormal Turnover:** Formulated as $ToverOs_t / \text{20-day mean} - 1$. This acts as the direct proxy for "liquidity shocks" and information arrival.
  2. **Demeaned Daily Return:** The raw return minus the equal-weighted market mean.
  3. **Log Market Cap:** Extracted from raw data (scaled from thousands of RMB).
  4. **Price Limit Status:** Explicit indicator ($\{1, 0, -1\}$) to capture the proposed regulatory friction bottlenecks.
  5. **Node2Vec Embedding:** Concatenated to the temporal features.

## 3. Target Variable & Thresholding
A robust mechanism is defined to create the 3-class target variable automatically scaling with market volatility regimes:
* Computed over an $h$-day cumulative demeaned future return.
* **Outperform (2):** return > $+0.5 \times \text{expanding\_std}(t)$
* **Neutral (1):** return between $\pm 0.5 \times \text{expanding\_std}(t)$
* **Underperform (0):** return < $-0.5 \times \text{expanding\_std}(t)$
* The expanding standard deviation utilizes data from start to date $t$ to strictly prevent data leakage. The $0.5$ threshold optimally balances class distribution to $\sim \frac{1}{3}$ each.

## 4. Modeling Architecture (HT-GNN)
The architecture perfectly aligns spatial graph processing with temporal sequence learning:
1. **GraphSAGE:** Aggregates neighbor signals independently at each of the $T=20$ timesteps.
2. **LSTM:** Processes the sequence of $T=20$ aggregated representations to capture the temporal lag structure of information diffusion.
3. **Linear Head:** Outputs the multi-class cross-entropy predictions for the 3 classes.

## 5. Implementation & Next Steps Ideas
Based on the defined file structure (`data_prep.py`, `supply_chain.py`, `node2vec_train.py`, etc.), here are ideas/recommendations for implementation:

### A. Data Parsing & Deduplication
* `EndDate` in the supply chain data defaults to `MM/DD/YY`; requires strict datetime parsing to ensure the look-ahead logic ($Dec 31_{Y-1} \rightarrow May 1_Y$) aligns safely perfectly with the daily timeline.
* **StateTypeCode Deduplication:** Prioritize consolidated statements (Code 1), but ensure robust fallback logic to parent-only statements (Code 2) during edge construction.

### B. Node2Vec Embedding Alignment
* As noted in the plan, independent annual Node2Vec trainings will shift the embedding coordinate spaces, meaning the $64$-dim vector for Firm A in 2017 cannot be directly compared to Firm A in 2018 in Euclidean space. 
* **Recommendation:** Incorporate Procrustes Analysis (Orthogonal Mapping) early in `node2vec_train.py` to align year $Y$'s embeddings to year $Y-1$, significantly stabilizing the LSTM's learning path across year boundaries.

### C. Event Study Evaluation
* The event study proposed in `evaluate.py` calculating **Cumulative Abnormal Returns (CARs)** is fundamentally excellent for a finance dissertation, clearly showing the temporal delay curve (the essence of a "Lead-Lag" effect).
* **Recommendation:** Track CARs for $+1$ to $+30$ days post-signal. Plotting the divergence curve of *Outperformers vs. Underperformers* will visually prove whether the HT-GNN succeeds in capturing the propagation timeline.
