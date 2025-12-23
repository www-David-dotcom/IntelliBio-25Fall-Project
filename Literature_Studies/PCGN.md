# PGCN: Disease gene prioritization by disease and gene embedding through graph convolutional neural networks

## 1 Introduction
Exusting disease-gene prioritization methods:
- Filter methods: simply filter out irrelevant genes based on prior knowledge.
- Text Mining: Score the candidate genes using literature data (co-occurrence of gene and disease names).
- Similarity Profiling & Data Fusion: Similar genes should belong to similar diseases.
- Network-based methods: Represent diseases and genes as nodes in a heterogeneous network, while similarities are represented as edges.
- Matrix Completion: represent disease-gene association with an incomplete matrix and try to complete it.

Limitations:
- Similarity-based methods fail to handle unknown diseases with no known associated genes.
- Network-based methods are biased by network topology and cnnot integrate multiple data sources effectively.
- Matrix completion methods assumes the relationship is linear, which is not true in biological systems.

## 2. Takeaway
- **Embedding Model**: Learn the potential representation of nodes from initial raw representations(information encoded from different sources), considering the graph's topological structure and the nodes' neighborhood. Combine these into an embedding vector.
- **Decoding Model**: Predict whether a disease and a gene are associated based on their embedding vectors. (edge prediction)
- These two models are trained **end-to-end**, which means we combine the two models into a same pipeline, using the same loss function to optimize both models simultaneously, and to regularize each other.

## 3. Prerequisites
### 3.1 GCNs
**Graph-based Semi-Supervised Learning**: node classification problem in a graph where labels are only available for a small subset of nodes.

**Model Representation**:
\[f(X, A)\]

- $X \in \mathbb{R}^{N \times F}$ is the feature matrix, where $N$ is the number of nodes and **$F$ is the number of features per node.**
- $A \in \mathbb{R}^{N \times N}$ is the adjacency matrix of the graph.

**Propagation Rule**:
\[H^{(\ell + 1)} = \sigma\left(\tilde{D}^{-\frac12}\tilde{A}\tilde{D}^{-\frac12}H^{(\ell)}W^{(\ell)}\right)\]

- $\tilde{A} = A + I$ representing the adjacency matrix with added self loops.

- $\tilde{D}$ is the degree matrix of $\tilde{A}$, i.e.
  \[\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}\]

- $W^{(\ell)}$ is the **trainable weight matrix**(parameter) at the $\ell$-th layer.

- $\sigma(\cdot)$ is an activation function.

- $H^{(\ell)}$ is the matrix of activations at the $\ell$-th layer, and $H^{(0)} = X$.

- Final output: \[Z = H^{(L)}\]
where $L$ is the total number of layers.

This is called graph convolutional layer because the expression comes from a much-simplified version of spectral graph convolutions. The mathematical details are elaborated [here](https://arxiv.org/abs/1609.02907).

**BackPropagation**:
Suppose we have a loss function $\mathcal{L}$ for the GCN, we first compute:
\[\frac{\partial \mathcal{L}}{\partial Z}\]
where $Z$ is the output of the GCN model. Then we can use the chain rule to compute the gradients of all parameters $W^{(\ell)}$ in each layer:
\[\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial Z} \cdot \frac{\partial Z}{\partial W^{(\ell)}}\] 
\[\ell = L-1, L-2, \ldots, 0\]
Use this to update the parameters using gradient descent or other optimizers.

**Solving the graph-based semi-supervised learning problem using GCNs**:
- Input: feature matrix $X$ and adjacency matrix $A$.
- Output: label predictions for all nodes.
- **Loss function**: cross-entropy loss over all labeled nodes.
\[\mathcal{L} = -\sum_{l \in \mathcal{Y}_L}\sum_{f=1}^F Y_{lf}\ln Z_{lf}\]
where $\mathcal{Y}_L$ is the set of labeled nodes, $Y$ is the real label, and $Z$ is the output of the GCN model:
\[Z = f(X, A)\]
Recall $F$ is the number of features per node.

### 3.2 TF-IDF
- **TF(Term Frequency)**: 
  \[TF = \frac{\text{Number of times term t appears in a document}}{\text{Total number of terms in the document}}\]
- **IDF(Inverse Document Frequency)**:
  \[IDF = \log\frac{\text{Total number of documents}}{\text{Number of documents with term t in it}}\]
- **TF-IDF**:
    \[TF-IDF = TF \times IDF\]

### 3.4 Ontology Similarities
#### 3.4.1  Resnik pairwise similarity
- Calculate occurrence frequency of a term $c$ in a corpus: $p(c)$
- Calculate information content of term $c$:
  \[IC(c) = -\log p(c)\]
- For two terms $c_1$ and $c_2$, find their common ancestors in the ontology graph, and select the one with the highest information content:
  \[\text{sim}_{\text{Resnik}}(c_1, c_2) = \max_{c \in \text{Anc}(c_1, c_2)} IC(c)\]
where $Anc(c_1, c_2)$ is the set of common ancestors of $c_1$ and $c_2$.
- For two diseases $d_1$ and $d_2$ each connected with a set of terms $s(d_1)$ and $s(d_2)$ respectively, calculate the pairwise similarity as:
  \[\text{sim}_{\text{Resnik}}(d_1, d_2) =  \max_{c_1 \in s(d_1), c_2 \in s(d_2)} \text{sim}_{\text{Resnik}}(c_1, c_2)\]

#### 3.4.2 Best Match Average(BMA) similarity
- **Representation**: Ontology databases are presented as directed acyclic graphs (DAGs) in which the terms form nodes and the two kinds of semantic relations (‘is-a’ and
‘part-of ’) form edges. ‘is-a’ is a simple class-subclass relation, where A is-a B means that A is a subclass of B. ‘part-of ’ is a partial ownership relation; C part-of D means that whenever C is present, it is always a part of D, but C need not always be present.


## 4. Method
### 4.1 Problem Formalization
- Nodes: a disease or a gene
- Edges: a specific kind of interaction.
- Supplemented Information
- Goal: Predict potential associations between diseases and genes.

### 4.2 Data Sources
- Hetrogeneous network: [gene network](https://www.inetbio.org/humannet/), [disease-similarity network](https://discover.nci.nih.gov/mim/), [disease-gene association network](https://www.omim.org/).
- Additional Information:
  - Diseases:
    - [Disease Ontology(DO)](https://www.disease-ontology.org/similarities). Use Gene Ontology Similarities to represent each disease as a vector.
    - Clinical Features from [OMIM](https://www.omim.org/). First delete most frequent and rarest terms. Use **TF-IDF** to represent each disease as a vector.
  - Genes:
    - Microarray measurement from [BioGPS](http://biogps.org/dataset/) and [CLUE](https://clue.io/) of gene expression level: Apply **PCA**(principal Component Analysis) to reduce dimension to 100.