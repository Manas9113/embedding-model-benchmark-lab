# Technical Interview Preparation Guide
## Embedding Model Benchmark Lab

This guide contains senior-level technical interview questions, mathematical formulations, and engineering explanations directly based on the implementation of this project.

---

### Category 1: Deep Learning & Embedding Architectures

#### Q1: What is the fundamental trade-off between a Bi-Encoder and a Cross-Encoder?
- **Bi-Encoder**: Encodes query $\mathbf{q}$ and document $\mathbf{d}$ independently:
  $$\mathbf{u} = f_{\theta}(\mathbf{q}), \quad \mathbf{v} = f_{\theta}(\mathbf{d})$$
  Similarity is computed via inner product: $s = \mathbf{u}^\top \mathbf{v}$.
  - *Pros*: Scalability. Document vectors can be computed once offline and indexed into an Approximate Nearest Neighbor (ANN) vector database (FAISS, HNSW, ScaNN). Retrieval over millions of documents takes $<5$ ms.
  - *Cons*: Information bottleneck. Because query and document do not interact in the self-attention layers, subtle cross-token nuances are compressed into a single vector.
- **Cross-Encoder**: Concatenates query and document into a single sequence:
  $$s = g_{\phi}([\mathbf{q}; \mathbf{d}])$$
  - *Pros*: Full all-to-all cross-attention across all tokens produces superior semantic matching and ranking accuracy.
  - *Cons*: Cannot be pre-indexed. Evaluating a corpus of $N$ documents requires $N$ forward passes through the transformer at query time, making it computationally infeasible for first-stage candidate retrieval.
- **Production Standard**: Two-stage retrieval. A Bi-Encoder retrieves the Top-100 candidates in 5ms, and a Cross-Encoder reranks those 100 candidates in 30ms.

---

#### Q2: Why did we normalize embeddings with $L_2$ normalization before computing cosine similarity?
- If vectors $\mathbf{u}$ and $\mathbf{v}$ are unit-normalized ($\|\mathbf{u}\|_2 = 1, \|\mathbf{v}\|_2 = 1$), cosine similarity simplifies directly to the inner product (dot product):
  $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u} \cdot \mathbf{v}$$
- On modern hardware (both SIMD/AVX CPU registers and GPU Tensor Cores), matrix-matrix multiplication (GEMM) is one of the most heavily optimized operations. Computing dot products avoids expensive square roots and divisions during top-$K$ search.
- Furthermore, squared Euclidean distance on unit-normalized vectors is strictly monotonic to dot product:
  $$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 2 - 2(\mathbf{u} \cdot \mathbf{v})$$
  Therefore, sorting by largest dot product yields the exact same ranking as sorting by smallest Euclidean distance.

---

#### Q3: How does MiniLM achieve distillation compared to standard BERT?
- Standard knowledge distillation (Hinton et al.) matches the teacher's soft output probabilities or intermediate hidden states via MSE loss.
- **MiniLM (Wang et al., 2020)** distills the **self-attention relation matrices** (the dot products of queries and keys) and value-relation matrices from the teacher's last transformer layer.
- Because relation matrices capture structural syntactic and semantic dependencies independent of hidden state dimensions, MiniLM allows the student model to have fewer layers (6 instead of 12) and a smaller hidden dimension (384 instead of 768) without layer-to-layer dimension projection constraints.

---

### Category 2: Information Retrieval & Evaluation Metrics

#### Q4: Why is nDCG@K preferred over Recall@K or MRR@K for search evaluation?
- **Recall@K** is unweighted: it treats a relevant document at rank 1 identical to a relevant document at rank 10.
- **MRR@K** only considers the rank of the *first* relevant document:
  $$\text{MRR}@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  If the first hit is at rank 1, MRR is 1.0, regardless of whether ranks 2–10 contain relevant documents or complete junk.
- **nDCG@K (Normalized Discounted Cumulative Gain)** accounts for:
  1. **Graded relevance**: Highly relevant documents ($rel=2$) contribute exponentially more gain ($2^{rel} - 1 = 3$) than marginally relevant documents ($rel=1$, gain $= 1$).
  2. **Position discount**: Gain is discounted logarithmically: $\log_2(\text{rank} + 1)$, heavily penalizing systems that bury relevant results at the bottom.
  3. **Normalization**: Dividing by $\text{IDCG}@K$ bounds the score between $0.0$ and $1.0$, enabling fair macro-averaging across queries with different numbers of relevant documents.

---

### Category 3: Measurement Hygiene & Systems Engineering

#### Q5: Why do we discard warm-up runs when benchmarking inference latency?
- In Python and PyTorch, initial inference passes incur:
  1. Model graph construction and dynamic memory allocations.
  2. CPU instruction cache (I-cache) misses and OS memory paging.
  3. PyTorch internal JIT tracing or cuDNN auto-tuner benchmark passes.
- Discarding the first 5 warm-up iterations isolates steady-state production latency from transient initialization overhead.

---

#### Q6: Why did batch throughput peak at batch size 8 or 16 on CPU instead of scaling indefinitely?
- On GPUs, throughput scales with larger batch sizes (32, 64, 128) because thousands of CUDA cores require massive parallelism to reach full compute saturation.
- On CPUs, cache hierarchy governs performance. As batch size increases, working memory buffers for intermediate attention matrices ($B \times H \times L \times L$) exceed the CPU's fast L2/L3 on-chip cache and spill into slower main system RAM (DDR4/DDR5). Memory bandwidth becomes the primary bottleneck, causing throughput to plateau or slightly decrease.

---

#### Q7: If you had 10 million documents and 100 queries/second, how would you calculate the vector index memory requirement?
- Storage per vector for float32:
  $$\text{Bytes} = \text{Dimension} \times 4 \text{ bytes}$$
- For a **384-dimensional** model (`all-MiniLM-L6-v2` or `bge-small-en-v1.5`):
  $$10,000,000 \times 384 \times 4 \text{ bytes} \approx 15.36 \text{ GB}$$
- For a **768-dimensional** model (`all-mpnet-base-v2`):
  $$10,000,000 \times 768 \times 4 \text{ bytes} \approx 30.72 \text{ GB}$$
- **Interview Takeaway**: Choosing a 384-dimensional model cuts RAM and vector database hosting costs by **50%** ($15.36$ GB vs $30.72$ GB). Furthermore, techniques like scalar quantization (SQ8) can reduce this to 1 byte per dimension (3.84 GB vs 7.68 GB) with $<1\%$ loss in nDCG.
