# CortexQuant: Online Vector Quantization with Near-optimal Distortion Rate

---

## Abstract

Vector quantization, a problem rooted in Shannon's source coding theory, aims to quantize high-dimensional Euclidean vectors while minimizing distortion in their geometric structure. We propose **CortexQuant** to address both mean-squared error (MSE) and inner product distortion, overcoming limitations of existing methods that fail to achieve optimal distortion rates. Our data-oblivious algorithms, suitable for online applications, achieve near-optimal distortion rates (within a small constant factor) across all bit-widths and dimensions. CortexQuant achieves this by randomly rotating input vectors, inducing a concentrated Beta distribution on coordinates, and leveraging the near-independence property of distinct coordinates in high dimensions to simply apply optimal scalar quantizers per each coordinate. Recognizing that MSE-optimal quantizers introduce bias in inner product estimation, we propose a two-stage approach: applying an MSE quantizer followed by a 1-bit Quantized JL (QJL) transform on the residual, resulting in an unbiased inner product quantizer. We also provide a formal proof of the information-theoretic lower bounds on best achievable distortion rate by any vector quantizer, demonstrating that CortexQuant closely matches these bounds, differing only by a small constant ($\approx 2.7$) factor. Experimental results validate our theoretical findings, showing that for KV cache quantization, we achieve absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits per channel. Furthermore, in nearest neighbor search tasks, our method outperforms existing product quantization techniques in recall while reducing indexing time to virtually zero.

---

## 1. Introduction

Vector quantization (VQ) in Euclidean space is crucial for efficiently handling high-dimensional vectors across a spectrum of computational domains, from training and deploying large-scale AI and deep learning models to powering vector databases for search/retrieval systems. The core objective is to compress high dimensional vectors by quantizing them — converting floating-point coordinate values to low-bitwidth integers — while minimizing distortion, quantified by metrics such as mean-squared error (MSE) or inner product errors. By preserving these properties, inner product queries can be answered rapidly, with minimal latency, and using reduced computational and communication resources.

This problem's roots trace back to Shannon's seminal work on Source Coding theory [shannon1948mathematical, shannon1959coding], which established that the least distortion achievable by block source codes, now known as vector quantizers, is defined by the Shannon distortion-rate function, determined by the statistical properties of the source and the chosen distortion measure, such as MSE. Today, VQ plays a critical role in fundamental computational domains, including AI, deep learning, and search systems.

A key application of VQ is in the deployment of AI models, including large language models (LLMs). As LLM capabilities depend heavily on their model size and context length, serving them requires substantial memory demands and increased inference latency. This latency is primarily attributed to communication bottlenecks between HBM and SRAM on accelerators, or across distributed clusters. By compressing or quantizing model weights and activations, we can effectively mitigate these bottlenecks, resulting in significant reductions in inference costs. Inner product operations between activations and weights is at the core of deep learning models. Thus, model quantization schemes strive to compress weights and/or activation vectors while accurately preserving these inner products.

Decoder-based transformer models present another compelling use case. These models must store key/value (KV) embeddings from previously generated tokens in the KV cache, the size of which scales with both model size (number of layers and attention heads) and context length. This scaling is a significant bottleneck in terms of memory usage and computational speed, especially for long context models. Therefore, reducing the KV cache size without compromising accuracy is essential. In this context, the preservation of the Euclidean structure of these embedding vectors — their inner products and distances — is crucial for maintaining model performance. VQ emerges as the most suitable framework for addressing this challenge, offering a robust approach to compressing high-dimensional embeddings while preserving their essential geometric properties.

Additionally, nearest neighbor (NN) search in high-dimensional spaces with inner product or cosine similarity is a cornerstone of vector databases. These databases are fundamental for retrieval-augmented generation and information retrieval. VQ, a.k.a. product quantization (PQ), plays a critical role in these applications. It enables efficient compression of database vectors, optimizes memory usage, and facilitates low-latency, accurate estimations of inner products with query vectors, thereby enabling fast and precise nearest neighbor searches.

Existing VQ algorithms present a trade-off: either they lack accelerator (vectorization) compatibility and exhibit slow computation, making them unsuitable for real-time AI applications like KV cache quantization, or they suffer from suboptimal distortion bounds relative to bit-width. Our objective is to introduce an algorithm that addresses these limitations. Specifically, we design **CortexQuant**: a lightweight, capable of online application (crucial for scenarios like KV cache quantization), and highly accelerator-friendly — a critical attribute for modern AI workloads.

The core of CortexQuant is a two-stage process. First, we develop a vector quantizer with optimal distortion rate in terms of mean-squared error (MSE). Subsequently, we apply a 1-bit quantizer to the residual, resulting in an unbiased and low-distortion inner product quantizer. We demonstrate that quantizers optimized for MSE do not produce unbiased estimators for inner products, and our two-stage solution effectively bridges this gap. Our MSE-optimal quantizer starts by randomly rotating $d$-dimensional input vectors. Observing the key fact that each coordinate in the rotated vectors follows a Beta distribution, we design optimal Lloyd-Max quantizer for each coordinate by solving a continuous k-means problem. This method gives optimal MSE distortion bound and minimizes the L2 norm of the residual. To obtain an unbiased and low-distortion quantizer for inner products, we compose our quantizer with the recently developed Quantized Johnson-Lindenstrauss (QJL) transform [qjl], which quantizes each coordinate of the residual vector to a single bit. Our algorithm offers provably optimal distortion bounds for both MSE and inner products, achieving an exponential improvement over existing methods in terms of bit-width dependence.

### 1.1 Problem Definition

Formally, our goal is to design a quantization map, denoted as $Q: \mathbb{R}^d \to \{ 0, 1 \}^B$, that transforms $d$-dimensional vectors to a binary string of $B$ bits. If we set $B = b \cdot d$ for some $b \ge 0$, this quantizer will have a bit-width of $b$, representing the average number of bits used to encode each real-valued coordinate of $\mathbb{R}^d$. Crucially, we require an inverse map, $Q^{-1}: \{ 0, 1 \}^B \to \mathbb{R}^d$ that performs dequantization, approximately reconstructing original vectors from their quantized representations. Of course, this transformation is inherently lossy, as $Q$ is not a bijection. So, our primary objective is to minimize distortion, with a specific focus on mean-squared error (MSE) and inner product distortion.

We make no assumptions about the input vector dataset, considering the worst-case scenario. We let the quantizer $Q(\cdot)$ to be randomized, leading to stochastic outputs. Considering randomized quantizers, it is more appropriate to define the expected distortion over the randomness of the quantizer's output. Thus, we aim to design quantizers that for any desired bit-width $b$ minimize the following expected distortion measures for any (worst-case) vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$:

$$\textbf{(MSE)} \quad D_{\tt mse} := \mathbb{E}_{Q}\left[\left\| \mathbf{x} - Q^{-1}\left( Q(\mathbf{x}) \right) \right\|_2^2 \right] \tag{1}$$

$$\textbf{(inner-prod error)} \quad D_{\tt prod} := \mathbb{E}_{Q}\left[\left| \langle \mathbf{y}, \mathbf{x} \rangle - \langle \mathbf{y}, Q^{-1}\left( Q(\mathbf{x}) \right) \rangle \right|^2 \right] \tag{2}$$

The expectations above are taken with respect to the randomness of the quantizer $Q(\cdot)$. Furthermore, for inner-product quantizers, we require unbiasedness of the inner product estimator, a desirable property for numerous applications. More precisely, we require:

$$\textbf{(unbiased inner-prod)} \quad \mathbb{E}_{Q}\left[ \langle \mathbf{y}, Q^{-1}\left( Q(\mathbf{x}) \right) \rangle\right] = \langle \mathbf{y}, \mathbf{x} \rangle$$

We aim to design computationally efficient quantizers $Q_{\tt mse}$ and $Q_{\tt prod}$, that achieve optimal bounds for the distortion measures defined above, for any given bit-width $b$. Additionally, we aim for $Q_{\tt prod}$ to provide unbiased inner product estimates. In particular, assume that we are given $n$ real-valued vectors $x_1, x_2, \ldots x_n \in \mathbb{R}^d$. We design the following primitives:

- **QUANT**: efficiently quantizes the dataset and computes $Q(\mathbf{x}_1), Q(\mathbf{x}_2), \ldots Q(\mathbf{x}_n)$.
- **DEQUANT**: given a quantized dataset, can efficiently reconstruct original vectors by computing $Q^{-1}\left( Q(\mathbf{x}_i) \right)$ for any $i \in [n]$.

### 1.2 Related Work

**Beginnings of VQ.** The vector quantization theory started by Shannon's seminal work on achievable distortion-rate functions. In 1963, Zador made significant advances by employing high-resolution methods to derive the limiting operational distortion-rate function for fixed-rate quantization at high rates that closely matches Shannon's distortion-rate function. However, Zador did not specifically consider implementable algorithms. Gersho's influential paper further advanced the vector quantization by popularizing high-resolution theory, simplifying Zador's results, introducing lattice vector quantization, and proposing a key conjecture that shaped the field. Despite these theoretical advancements, the practical applicability of vector quantization remained unclear in early years. The most straightforward encoding method, brute-force nearest neighbor search, was computationally expensive, hindering the adoption of VQ in practice.

**Online vs Offline Quantization.** Online (data-oblivious) quantization methods apply instantly without needing data-specific tuning or calibrations. In contrast, offline (data-dependent) methods require heavy preprocessing and learning to adapt the quantization map to the data, making them unsuitable for dynamic data scenarios. For instance, methods such as those using second-order (Hessian) information to tune the quantization map require heavy preprocessing and even in some cases post processing as well.

**Online KV Cache Compression.** Several approaches have been proposed to compress the KV cache. These include architectural modifications which restructure the transformer to minimize the number of stored key-value pairs. Additionally, pruning or evicting redundant or less critical tokens has emerged as another approach.

A simple yet effective approach to reducing KV cache size is quantizing the KV cache. Several quantization techniques have been developed specifically for this purpose. Recently, a new quantization called QJL introduced an efficient, data-oblivious 1-bit quantization approach based on sketching techniques, which provides unbiased estimates for inner product queries. This method does not require tuning or adaptation to the input data and we make use of this technology in our quantizer optimized for inner product distortion.

**Product Quantization (PQ).** In Near Neighbor (NN) search problem with Euclidean datasets, the index size poses a significant memory bottleneck, often mitigated by quantization techniques, commonly referred to as Product Quantization (PQ) in the NN literature. Many of these algorithms rely on constructing a quantization codebook using variations of k-means during the indexing phase. Therefore, these methods are ill-suited for online settings due to their requirement for extensive preprocessing.

Recently, a grid-based PQ method was introduced, eliminating the need for preprocessing. This approach operates by projecting a uniform grid onto the unit sphere and conducting a search to identify the nearest projection to the data points. While the paper's theoretical guarantees are suboptimal, likely due to loose analysis — as practical performance surpasses theoretical bounds — the grid projection and binary search algorithm is also computationally slow and particularly inefficient on accelerators like GPU because of their algorithm's inherent lack of vectorization, which prevents parallel processing.

### 1.3 Overview of Techniques and Contributions

**MSE Optimized CortexQuant.** Our first VQ algorithm is designed to minimize MSE distortion. To achieve this, we apply a random rotation to the input vectors, thereby inducing a Beta distribution on each coordinate, irrespective of the input vectors themselves. In high dimensions $d$, the distribution of each coordinate converges to a Gaussian distribution $\mathcal{N}(1, 1/d)$ due to concentration of measure and the central limit theorem. Furthermore, any two distinct coordinates become nearly uncorrelated and, more importantly, almost independent (a deeper result that goes beyond just correlation). This near-independence is a crucial aspect that simplifies our quantization design. It allows us to quantize each coordinate using optimal scalar quantization, disregarding interactions or correlations between different coordinates, while still achieving near-optimal distortion.

We find optimal scalar quantizers for random variables with Beta distributions by solving a continuous $1$-dimensional k-means problem using the Max-Lloyd algorithm. We precompute and store these optimal codebooks for a range of practically useful bit-widths, to enable efficient subsequent invocations of our CortexQuant algorithm.

In Theorem 1 we prove that the $b$-bit MSE optimized CortexQuant $Q_{\tt mse}: \mathbb{R}^d \to \{ 0, 1 \}^{b \cdot d}$ achieves the following distortion for any worst-case vector $\mathbf{x} \in \mathbb{R}^d$ with $\|\mathbf{x}\|=1$:

- $D_{\tt mse}(Q_{\tt mse}) := \mathbb{E}\left[\left\| \mathbf{x} - Q_{\tt mse}^{-1}\left( Q_{\tt mse}(\mathbf{x}) \right) \right\|_2^2 \right] \le \frac{\sqrt{3} \pi}{2} \cdot \frac{1}{4^{b}}$ for any $b \ge 0$.
- For small bit-widths the above distortion upper bound can be further refined. Specifically, for $b = 1, 2, 3, 4$ we have $D_{\tt mse}(Q_{\tt mse}) \approx \mathbf{0.36}, \mathbf{0.117}, \mathbf{0.03}, \mathbf{0.009}$, respectively.

Note that the unit norm assumption, $\|\mathbf{x}\|_2=1$, is standard and not restrictive. For datasets that do not satisfy this assumption we can compute and store the $L2$ norms in floating-point precision and rescale the dequantized points using these stored norms.

**Inner Product CortexQuant.** We show that the MSE optimized quantizers are biased for inner product estimation and thus a different VQ scheme is needed to get an unbiased inner product quantizer. Our solution is a two stage algorithm that first applies the abovementioned $Q_{\tt mse}$ with a bit-width one less than our target budget and then apply a QJL on the residual error. This is proved to be unbiased and also has nearly optimal inner product error rate.

In Theorem 2 we prove that the $b$-bit inner product optimized CortexQuant $Q_{\tt prod}: \mathbb{R}^d \to \{ 0, 1 \}^{b \cdot d}$ achieves the following distortion for any worst-case vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ with $\|\mathbf{x}\|=1$:

- $\mathbb{E}\left[ \left\langle \mathbf{y}, Q_{\tt prod}^{-1}\left( Q_{\tt prod}(\mathbf{x}) \right) \right\rangle \right] = \langle \mathbf{y}, \mathbf{x} \rangle$
- $D_{\tt prod}(Q_{\tt prod}) := \mathbb{E}\left[\left| \langle \mathbf{y}, \mathbf{x} \rangle - \langle \mathbf{y}, Q_{\tt prod}^{-1}\left( Q_{\tt prod}(\mathbf{x}) \right) \rangle \right|^2 \right] \le \frac{\sqrt{3} \pi^2 \cdot \|\mathbf{y}\|_2^2}{d} \cdot \frac{1}{4^{b}}$ for any $b \ge 0$.
- For small bit-widths the above distortion upper bound can be further refined. Specifically, for $b = 1, 2, 3, 4$ we have $D_{\tt prod}(Q_{\tt prod}) \approx \frac{\mathbf{1.57}}{d}, \frac{\mathbf{0.56}}{d}, \frac{\mathbf{0.18}}{d}, \frac{\mathbf{0.047}}{d}$, respectively.

**Lower Bound.** In Theorem 3, we leverage Shannon's lower bound and Yao's minimax principle to prove that for any randomized quantization algorithm $Q: \mathbb{R}^d \to \{ 0, 1 \}^{b \cdot d}$ with bit-width $b$, there exist hard input instances $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ with $\|\mathbf{x}\| = 1$ such that the following lower bounds hold:

- $D_{\tt mse}(Q) := \mathbb{E}\left[\left\| \mathbf{x} - Q^{-1}\left( Q(\mathbf{x}) \right) \right\|_2^2 \right] \ge \frac{1}{4^{b}}$
- $D_{\tt prod}(Q) = \mathbb{E}\left[\left| \langle \mathbf{y}, \mathbf{x} \rangle - \langle \mathbf{y}, Q^{-1}\left( Q(\mathbf{x}) \right) \rangle \right|^2 \right] \ge \frac{\|\mathbf{y}\|_2^2}{d} \cdot \frac{1}{4^{b}}$

As demonstrated by our lower bounds, CortexQuant's MSE distortion is provably within a factor of at most $\frac{\sqrt{3} \pi}{2} \approx \mathbf{2.7}$ of the information-theoretical lower bound. Notably, for smaller bit-widths, this factor significantly decreases. For instance, at a bit-width of $b=1$ CortexQuant achieves a distortion that is only a factor of approximately $\mathbf{1.45}$ away from the optimal which is also confirmed by our experimental results, indicating its efficiency in low-bit-width scenarios.

**Experimental Results.** In Section 5.1, we empirically validate our theoretical distortion bounds, demonstrating that CortexQuant's observed distortions closely align with our predictions across various real-world datasets, approaching the established lower bounds.

Furthermore, in Sections 5.2 and 5.3, we showcase CortexQuant's efficacy in online KV cache quantization. Specifically, we achieve perfect long-context retrieval in needle-in-a-haystack tasks and maintain high performance on other long-context downstream tasks, all while compressing the KV cache by a factor exceeding $5\times$.

Finally in Section 5.4 we apply CortexQuant to various high-dimensional near neighbor search tasks. CortexQuant consistently outperforms data-dependent product quantization (PQ), while reducing the indexing time to essentially zero.

---

## 2. Preliminaries

We use boldface lowercase letters, such as $\mathbf{x}$ and $\mathbf{y}$, to denote vectors, and boldface uppercase letters, like $\mathbf{M}$, to denote matrices. To denote a slice of a vector $\mathbf{x}$ between the coordinate indices $i$ and $j$ inclusive of the endpoints, we use the notation $\mathbf{x}_{i:j}$. For a matrix $\mathbf{M}$, we write $\mathbf{M}_{i,:}$ to denote its $i$-th row vector, which we will simply refer to as $\mathbf{M}_i$.

We use the notation $\mathbb{S}^{d-1}$ to denote the hypersphere in $\mathbb{R}^d$ of radius $1$. For a random variable $x$ we denote its differential entropy as $h(x)$. For random variables $x$ and $y$, the mutual information between them is denoted as $I(x; y) = h(x) - h(x|y)$.

Given that CortexQuant employs random rotation to mitigate worst-case input scenarios, understanding the statistical properties of random points on a hypersphere is essential. The following lemma outlines one such property that we will need for analysis and design purposes:

**Lemma 1** (Coordinate distribution of random point on hypersphere). _For any positive integer $d$ if $\mathbf{x} \in \mathbb{S}^{d-1}$ is a random variable uniformly distributed over the unit hypersphere, then for any $j \in [d]$ the coordinate $\mathbf{x}_j$ follows the following (scaled/shifted) Beta distribution:_

$$\mathbf{x}_j \sim f_{X}(x) := \frac{\Gamma(d/2)}{\sqrt{\pi} \cdot \Gamma((d-1)/2)} \left( 1 - x^2 \right)^{(d-3)/2}$$

_In high dimensions this Beta distribution converges to the normal distribution $f_{X}(\cdot) \to \mathcal{N}(0, 1/d)$.\_

**Proof.** $f_X(x)$ equals the ratio of the area of a sphere with radius $\sqrt{1-x^2}$ in dimension $d-1$ to the volume of a unit sphere in dimension $d$ scaled down by $1/\sqrt{1-x^2}$ (by Pythagorean theorem). Therefore,

$$f_X(x) = \frac{\frac{2 \pi^{(d-1)/2}}{\Gamma((d-1)/2)} \cdot (1-x^2)^{(d-2)/2}}{\frac{2 \pi^{d/2}}{\Gamma(d/2)}} \cdot 1/\sqrt{1-x^2}= \frac{\Gamma(d/2)}{\sqrt{\pi} \cdot \Gamma((d-1)/2)} \left( 1 - x^2 \right)^{(d-3)/2}$$

### 2.1 Shannon Lower Bound on Distortion

The Shannon Lower Bound (SLB) is a powerful tool, derived from Shannon's lossy source coding theorem, that provides a universal lower bound on the optimal achievable distortion rate for any lossy compression scheme. Specifically, we use a version of SLB tailored for the mean-squared error (MSE) distortion measure applied to general $d$-dimensional sources.

**Lemma 2** (SLB). _Let $\mathbf{x} \in \mathbb{R}^d$ be a random vector with an arbitrary probability distribution $p_X$ and finite differential entropy $h(\mathbf{x})$. Define the MSE distortion-rate function $D(B)$ for total bit complexity $B \ge 0$ as:_

$$D(p_X, B) := \inf \left\{ \mathbb{E} \left[ \|\mathbf{x} - \mathbf{y}\|_2^2 \right] : I(\mathbf{x}; \mathbf{y}) \le B \right\}$$

_where the infimum is taken over all joint distributions of $\mathbf{x}$ and a reconstruction random vector $\mathbf{y} \in \mathbb{R}^d$ such that the mutual information $I(\mathbf{x}; \mathbf{y})$ is at most $B$ and $\mathbb{E} \left[ \|\mathbf{x} - \mathbf{y}\|_2^2 \right]$ is the expected MSE distortion, calculated with respect to the joint distribution of $\mathbf{x}$ and $\mathbf{y}$. Then, for any bit complexity $B \ge 0$, the following Shannon Lower Bound holds:_

$$D(p_X, B) \ge \frac{d}{2 \pi e} \cdot 2^{(2/d) (h(\mathbf{x}) - B)}$$

This is a classic result proved using backward Gaussian test channel. Our lower bound result uses a corollary of SLB that corresponds to the uniformly distributed random points on the unit hypersphere. We present this in the following lemma:

**Lemma 3** (SLB for random point on hypersphere). _Let $\mathbf{x} \in \mathbb{S}^{d-1}$ be a random variable uniformly distributed over the unit hypersphere and define the MSE distortion-rate function $D(B)$ for total bit complexity $B$ as per Lemma 2. Then, for any bit complexity $B \ge 0$, the following distortion lower bound holds:_

$$D(B) \ge 2^{-2B/d}$$

**Proof.** If we let $A_d$ denote the area of the hypersphere $\mathbb{S}^{d-1}$, the entropy of uniform distribution over hypersphere is $h(\mathbf{x}) = \log_2 A_d$. Plugging this into the SLB from Lemma 2 we get $D(B) \ge \frac{d}{2 \pi e} \cdot {A_d}^{2/d} \cdot 2^{-2B/d}$. Using Stirling's approximation formula for Gamma function we have $A_d = \frac{2 \pi^{d/2}}{\Gamma(d/2)} \ge \left(\frac{2 \pi e}{d} \right)^{d/2} \cdot \sqrt{\frac{2d}{\pi}} \cdot (1 - O(1/d))$. By substituting this into the inequality obtained from Lemma 2 we get the desired lower bound.

### 2.2 QJL: 1-bit Inner Product Quantization

As previously stated, we design two VQ algorithms: one optimized for minimizing MSE and the other for minimizing inner product error. We show that MSE-optimal quantizers do not necessarily provide unbiased inner product estimates, particularly exhibiting significant bias at lower bit-widths. Our solution for inner product quantization is a two-stage algorithm. First, we apply the MSE-optimal quantizer using one less bit than the desired bit-width budget, thus minimizing the L2 norm of the residuals. Next we apply an unbiased and optimal single-bit quantizer to the residual. For the single-bit inner product quantizer, we utilize the recently proposed Quantized Johnson-Lindenstrauss (QJL) algorithm [qjl], which is an optimal inner product quantizer with a bit-width of one. Here, we present the QJL algorithm and its essential theoretical guarantees.

**Definition 1** (QJL). For any positive integer $d$ the QJL map $Q_{\tt qjl}: \mathbb{R}^d \to \{ -1, +1 \}^d$ is defined as:

$$Q_{\tt qjl}(\mathbf{x}) := \mathtt{sign} \left( \mathbf{S} \cdot \mathbf{x} \right) \quad \text{ for any } \mathbf{x} \in \mathbb{R}^d$$

where $\mathbf{S} \in \mathbb{R}^{d \times d}$ is a random matrix with i.i.d. entries sampled from the normal distribution $\mathcal{N}(0, 1)$ and the $\mathtt{sign}$ function is applied entry-wise to its vector input. The inverse/dequantization map $Q_{\tt qjl}^{-1}: \{ -1, +1\}^d \to \mathbb{R}^d$ is defined as:

$$Q_{\tt qjl}^{-1}(\mathbf{z}) := \frac{\sqrt{\pi/2}}{d} \cdot \mathbf{S}^\top \cdot \mathbf{z} \quad \text{ for any } \mathbf{z} \in \{ -1, +1\}^d$$

In the next lemma we restate the results from [qjl] that show the QJL is unbiased and also has small inner product distortion:

**Lemma 4** (Performance guarantee: QJL). _Let $Q_{\tt qjl}$ and $Q_{\tt qjl}^{-1}$ be defined as per Definition 1. For any vector $\mathbf{x} \in \mathbb{S}^{d-1}$ and any $\mathbf{y} \in \mathbb{R}^d$ we have the following:\_

- _Unbiased:_ $\mathbb{E}\left[ \left\langle \mathbf{y}, Q_{\tt qjl}^{-1}\left( Q_{\tt qjl}(\mathbf{x}) \right) \right\rangle \right] = \langle \mathbf{y}, \mathbf{x} \rangle$
- _Variance Bound:_ $\mathtt{Var} \left( \left\langle \mathbf{y}, Q_{\tt qjl}^{-1}\left( Q_{\tt qjl}(\mathbf{x}) \right) \right\rangle \right) \le \frac{\pi}{2 d} \cdot \|\mathbf{y}\|_2^2$

**Proof.** The unbiasedness immediately follows from Lemma 3.2 of [qjl]. To show the variance bound let $\mathbf{s}_1, \mathbf{s}_2, \ldots \mathbf{s}_m$ denote the rows of the random matrix $\mathbf{S}$ in Definition 1. We have:

$$\left\langle \mathbf{y}, Q_{\tt qjl}^{-1}\left( Q_{\tt qjl}(\mathbf{x}) \right) \right\rangle = \frac{1}{d} \sum_{i\in[d]} \sqrt{\pi/2} \cdot \mathbf{s}_i^\top \mathbf{y} \cdot \mathtt{sign}(\mathbf{s}_i^\top \mathbf{x})$$

Since $\mathbf{s}_i$'s are i.i.d. the above is indeed the average of $d$ i.i.d. random samples defined as $z_i := \sqrt{\pi/2} \cdot \mathbf{s}_i^\top \mathbf{y} \cdot \mathtt{sign}(\mathbf{s}_i^\top \mathbf{x})$ for $i \in [d]$. Let us now upper bound the variance of a single $z_i$ using Fact 3.4 from [qjl]:

$$\mathtt{Var} \left( z_i \right) = \pi/2 \cdot \mathtt{Var} \left( \mathbf{s}_i^\top \mathbf{y} \cdot \mathtt{sign}(\mathbf{s}_i^\top \mathbf{x}) \right) \le \pi/2 \cdot \mathbb{E} \left[ (\mathbf{s}_i^\top \mathbf{y})^2 \right] = \pi/2 \cdot \|\mathbf{y}\|_2^2 \tag{3}$$

where the last equality above follows because $\mathbf{s}_i^\top \mathbf{y}$ is a Gaussian random variable with mean zero and variance $\|\mathbf{y}\|_2^2$. Now the variance of the average of $d$ i.i.d. random samples $z_1, z_2, \ldots z_d$ is:

$$\mathtt{Var} \left( \left\langle \mathbf{y}, Q_{\tt qjl}^{-1}\left( Q_{\tt qjl}(\mathbf{x}) \right) \right\rangle \right) = \frac{1}{d^2} \sum_{i\in[d]} \mathtt{Var} ( z_i ) \le \frac{\pi}{2 d} \cdot \|\mathbf{y}\|_2^2$$

---

## 3. CortexQuant: High Performance Quantization

We developed two VQ algorithms, each tailored to a specific objective. The first algorithm is designed to minimize the MSE between the original and reconstructed vectors after quantization. The second algorithm is optimized for unbiased inner product estimation, addressing the bias inherent in MSE-optimal quantizers. These algorithms are detailed in the following subsections.

Furthermore, in Section 3.3, we establish information-theoretic lower bounds on the best achievable distortion rates for any vector quantizer. This analysis demonstrates that CortexQuant achieves near-optimality, differing from the lower bound by only a small constant factor across all bit-widths.

### 3.1 MSE Optimal CortexQuant

Let $\mathbf{x} \in \mathbb{S}^{d-1}$ be a (worst-case) vector on the unit sphere in dimension $d$. We aim to quantize $\mathbf{x}$ to $b$ bits per coordinate while minimizing the reconstruction MSE. We start by randomizing this vector by multiplying it with a random rotation matrix $\boldsymbol{\Pi} \in \mathbb{R}^{d \times d}$. We can generate $\boldsymbol{\Pi}$ by applying QR decomposition on a random matrix with i.i.d Normal entries.

The resulting rotated vector, $\boldsymbol{\Pi} \cdot \mathbf{x}$, is uniformly distributed on the unit sphere $\mathbb{S}^{d-1}$. As shown in Lemma 1, each coordinate of $\boldsymbol{\Pi} \cdot \mathbf{x}$ follows a Beta distribution, which converges to a normal distribution in high dimensions. Furthermore, in high dimensions, distinct coordinates of $\boldsymbol{\Pi} \cdot \mathbf{x}$ become nearly independent, allowing us to apply optimal scalar quantizers to each coordinate independently. Therefore, by Lemma 1, our task reduces to designing a scalar quantizer for random variables with the distribution $f_{X}(x) = \frac{\Gamma(d/2)}{\sqrt{\pi} \cdot \Gamma((d-1)/2)} \left( 1 - x^2 \right)^{(d-3)/2}$ for $x \in [-1, 1]$.

The optimal scalar quantization problem, given a known probability distribution, can be framed as a continuous k-means problem in dimension one. Specifically, we aim to partition the interval $[-1, 1]$ into $2^b$ clusters/buckets. The optimal solution adheres to a Voronoi tessellation, meaning interval boundaries are the midpoints between consecutive centroids, when arranged in sorted order. Therefore, with $c_i$'s denoting the centroids in ascending order, we can formulate the scalar quantization as the following k-means optimization problem:

$$\mathcal{C}(f_X, b) := \min_{-1 \le c_1 \le c_2 \le \ldots \le c_{2^b} \le 1} \sum_{i=1}^{2^b} \int_{\frac{c_{i-1} + c_i}{2}}^{\frac{c_i + c_{i+1}}{2}} |x - c_i|^2 \cdot f_X(x)\,dx \tag{4}$$

Note that $\mathcal{C}(f_X, b)$ in (4) denotes the optimal MSE cost function for bit-width $b$, a quantity we will bound to prove the upper bound on the end-to-end MSE of CortexQuant. The problem in (4) can be solved using iterative numerical methods to achieve any desired precision. We solve (4) for a range of practically relevant bit-widths $b$ once, and store the results for future uses by the quantizer.

For example, in moderately high dimensions $d$, where the distribution $f_X(x)$ closely approximates a normal distribution, the optimal quantization centroids for bit-widths $b = 1, 2$ are $\left\{ \pm \frac{\sqrt{2/\pi}}{\sqrt{d}} \right\}$ and $\left\{ \pm \frac{0.453}{\sqrt{d}}, \pm \frac{1.51}{\sqrt{d}} \right\}$, respectively.

Therefore the quantizer $Q_{\tt mse}: \mathbb{R}^d \to \{ 0, 1 \}^{b \cdot d}$ first computes $\boldsymbol{\Pi} \cdot \mathbf{x}$ and then computes and stores the indices of the nearest centroids to each coordinate of this vector. The dequantization map $Q_{\tt mse}^{-1}: \{ 0, 1 \}^{b \cdot d} \to \mathbb{R}^d$ reconstructs the vector by retrieving the centroids corresponding to the stored indices and then rotating the result back to the original basis through multiplication with $\boldsymbol{\Pi}^\top$. A pseudocode for these procedures is given in Algorithm 1.

---

**Algorithm 1:** $\text{CortexQuant}_{\tt mse}$: optimized for MSE

**Input:** dimension $d$ and bit-width $b$

> _// Global Parameters for Setting up $\text{CortexQuant}_{\tt mse}$\_

1. Generate a **random rotation matrix** $\boldsymbol{\Pi} \in \mathbb{R}^{d \times d}$
2. Construct **codebook** by finding centroids $c_1, c_2, \ldots c_{2^b} \in [-1, 1]$ that minimize MSE cost in (4)

---

**Procedure** $\textsc{Quant}_{\tt mse}(\mathbf{x})$

3. $\mathbf{y} \gets \boldsymbol{\Pi} \cdot \mathbf{x}$
4. $\mathtt{idx}_j \gets \arg\min_{k \in [2^b]} |\mathbf{y}_j - c_k|$ for every $j \in [d]$ &emsp; _(idx$_j$'s are $b$-bit integers)_
5. **output:** $\mathtt{idx}$

---

**Procedure** $\textsc{DeQuant}_{\tt mse}(\mathtt{idx})$

6. $\tilde{\mathbf{y}}_j \gets c_{\mathtt{idx}_j}$ for every $j \in [d]$
7. $\tilde{\mathbf{x}} \gets \boldsymbol{\Pi}^\top \cdot \tilde{\mathbf{y}}$
8. **output:** $\tilde{\mathbf{x}}$

---

We are now ready to prove our main theorem for $\text{CortexQuant}_{\tt mse}$.

**Theorem 1** (Performance guarantee: $\text{CortexQuant}_{\tt mse}$). _For any bit-width $b \ge 1$ and any vector $\mathbf{x} \in \mathbb{S}^{d-1}$, the procedure $\textsc{Quant}_{\tt mse}(\mathbf{x})$ in Algorithm 1 outputs an index vector $\mathtt{idx} \in [2^b]^d$. When this index vector is passed to the primitive $\textsc{DeQuant}_{\tt mse}(\mathtt{idx})$, it produces a reconstructed vector $\tilde{\mathbf{x}} \in \mathbb{R}^d$ that satisfies the following distortion bounds:\_

- _MSE defined as $D_{\tt mse} := \mathbb{E}_{\tilde{\mathbf{x}}}[ \|\mathbf{x} - \tilde{\mathbf{x}}\|_2^2 ]$ is bounded by $D_{\tt mse} \le \frac{\sqrt{3} \pi}{2} \cdot \frac{1}{4^{b}}$ for any $b \ge 0$.\_
- _For small bit-widths, specifically $b = 1, 2, 3, 4$ the MSE exhibits finer-grained distortion values: $D_{\tt mse} \approx \mathbf{0.36}, \mathbf{0.117}, \mathbf{0.03}, \mathbf{0.009}$, respectively.\_

**Proof.** We start the proof by showing that $D_{\tt mse} = d \cdot \mathcal{C}(f_X, b)$, where $\mathcal{C}(f_X, b)$ is the optimal MSE cost for scalar quantizer defined in (4). Let $\tilde{\mathbf{y}}$ be defined as per Algorithm 1. Since $\boldsymbol{\Pi}$ is a rotation matrix we can write: $\|\mathbf{x} - \tilde{\mathbf{x}}\|_2 = \|\boldsymbol{\Pi} \cdot \mathbf{x} - \tilde{\mathbf{y}}\|_2$. Using the notation $\mathbf{y} = \boldsymbol{\Pi} \cdot \mathbf{x}$ and plugging this into the definition of $D_{\tt mse}$ we can write:

$$
\begin{align}
D_{\tt mse} &= \mathbb{E} [\|\mathbf{y} - \tilde{\mathbf{y}}\|_2^2] \\
&= \sum_{j \in [d]} \mathbb{E}\left[ |\mathbf{y}_j - \tilde{\mathbf{y}}_j|^2 \right] \\
&= \sum_{j \in [d]} \mathbb{E}\left[ |\mathbf{y}_j - c_{\mathtt{idx}_j}|^2 \right]\\
&= d \cdot \mathbb{E}\left[ |\mathbf{y}_1 - c_{\mathtt{idx}_1}|^2 \right] \\
&= d \cdot \min_{-1 \le c_1 \le \ldots \le c_{2^b} \le 1} \sum_{i=1}^{2^b} \int_{\frac{c_{i-1} + c_i}{2}}^{\frac{c_i + c_{i+1}}{2}} |x - c_i|^2 \cdot f_X(x)\,dx \\
&= d \cdot \mathcal{C}(f_X, b)
\end{align}
$$

Now we must bound the optimal k-means cost $\mathcal{C}(f_X, b)$. For moderate values of $d$, $f_X \to \mathcal{N}(0, 1/d)$. By numerically solving the optimization problem in (4) for values $b = 1, 2, 3, 4$ we get that $\mathcal{C}(f_X, b) \approx \frac{0.36}{d}, \frac{0.117}{d}, \frac{0.03}{d}, \frac{0.009}{d}$, respectively. For larger bit-widths $b > 4$, we can apply the Panter-Dite high-resolution formula for the distortion of a fixed-rate scalar quantizer, yielding the following bound:

$$\mathcal{C}(f_X, b) \le \frac{1}{12} \cdot \left( \int f_X(x)^{1/3}\,dx \right)^3 \cdot \frac{1}{4^b} = \frac{\sqrt{3} \pi }{2 d} \cdot \frac{1}{4^b}$$

This completes the proof.

**Entropy Encoding Codebook Pointers.** CortexQuant's efficiency can be further increased by applying entropy encoding to the indices that point to the closest codebook elements. Specifically, the probability of each codeword index appearing in the quantized vectors can be computed as $p_\ell := \int_{\frac{c_{\ell-1} + c_\ell}{2}}^{\frac{c_\ell + c_{\ell+1}}{2}} f_X(x)\,dx$. Optimally coding the indices, reduces the average bit-width to nearly the entropy of the distribution $\{ p_i \}_{i \in [2^b]}$. This lossless compression does not affect the distortion and provides a bit-width reduction at no cost. The most significant reduction occurs for $b=4$, where the entropy of $\{ p_i \}_{i \in [2^b]}$ is approximately $3.8$. Detailed calculations for optimal prefix codes reveal that the average bit-width can be reduced by $5\%$. However, given the limited gain, we have chosen not to incorporate this technique into CortexQuant to maintain simplicity and speed.

### 3.2 Inner-Product Optimal CortexQuant

For important applications like nearest neighbor search, having an unbiased inner product estimator is essential. However, $\text{CortexQuant}_{\tt mse}$ presented in Section 3.1 does not provide unbiased inner product estimates with query vectors. To illustrate this, consider the case with a bit-width of $b=1$. In this scenario, the optimal codebooks that solve the optimization problem in (4), for sufficiently large $d$, are $\left\{ \pm \sqrt{\frac{2}{\pi d}} \right\}$. This implies that the quantization map for $\text{CortexQuant}_{\tt mse}$ is $Q_{\tt mse}(\mathbf{x}) = \mathtt{sign} \left( \boldsymbol{\Pi} \cdot \mathbf{x} \right)$ for any $\mathbf{x} \in \mathbb{R}^d$, and the dequantization map is $Q_{\tt mse}^{-1}(\mathbf{z}) = \sqrt{\frac{2}{\pi d}} \cdot \boldsymbol{\Pi}^\top \cdot \mathbf{z}$ for any $\mathbf{z} \in \{ -1, +1\}^d$. Therefore, for large enough $d$, according to Lemma 4, we have $\mathbb{E}\left[ \left\langle \mathbf{y}, Q_{\tt mse}^{-1}\left( Q_{\tt mse}(\mathbf{x}) \right) \right\rangle \right] = \frac{2}{\pi} \cdot \langle \mathbf{y}, \mathbf{x} \rangle$, which has a multiplicative bias of $2/\pi$. This bias diminishes with increasing bit-widths $b$, as we empirically demonstrate in Section 5.1.

To address this bias, we propose a solution that combines $\text{CortexQuant}_{\tt mse}$ with an instance of QJL [qjl]. Specifically, let $Q_{\tt mse}$ be the quantization map corresponding to $\text{CortexQuant}_{\tt mse}$ with a bit-width of $b-1$. For any $\mathbf{x} \in \mathbb{S}^{d-1}$ the residual vector, defined as $\mathbf{r} := \mathbf{x} - Q_{\tt mse}^{-1}\left( Q_{\tt mse}(\mathbf{x}) \right)$, has a small L2 norm, i.e., on expectation $\mathbb{E}[\|\mathbf{r}\|] = \sqrt{\mathcal{C}(f_X, b-1)}$ (per (4)). We can then apply the QJL quantization map $Q_{\tt qjl}$ on this residual vector, resulting in an overall bit-width of $b$ and providing the following unbiased inner product estimator:

$$\left\langle \mathbf{y}, Q_{\tt mse}^{-1}\left( Q_{\tt mse}(\mathbf{x}) \right) \right\rangle + \|\mathbf{r}\|_2 \cdot \left\langle \mathbf{y}, Q_{\tt qjl}^{-1}\left( Q_{\tt qjl}(\mathbf{r}) \right) \right\rangle$$

More formally, the quantization map $Q_{\tt prod}: \mathbb{S}^{d-1} \to [2^{b-1}]^d \times \{ -1, 1 \}^d \times \mathbb{R}$ is defined as:

$$Q_{\tt prod}(\mathbf{x}) = \left[ Q_{\tt mse}(\mathbf{x}),\ Q_{\tt qjl}\left( \mathbf{x} - Q_{\tt mse}^{-1}\left( Q_{\tt mse}(\mathbf{x}) \right) \right),\ \left\|\mathbf{x} - Q_{\tt mse}^{-1}\left( Q_{\tt mse}(\mathbf{x}) \right)\right\|_2 \right]$$

A pseudocode for this procedure is given in Algorithm 2.

---

**Algorithm 2:** $\text{CortexQuant}_{\tt prod}$: optimized for inner product

**Input:** dimension $d$ and bit-width $b$

> _// Global Parameters for Setting up $\text{CortexQuant}_{\tt prod}$\_

1. Instantiate a **$\text{CortexQuant}_{\tt mse}$** with bit-width $b-1$ as per Algorithm 1
2. Generate a **random projection matrix** $\mathbf{S} \in \mathbb{R}^{d \times d}$ with i.i.d. entries $\mathbf{S}_{i,j} \sim \mathcal{N}(0, 1)$

---

**Procedure** $\textsc{Quant}_{\tt prod}(\mathbf{x})$

3. $\mathtt{idx} \gets \textsc{Quant}_{\tt mse}(\mathbf{x})$
4. $\mathbf{r} \gets \mathbf{x} - \textsc{DeQuant}_{\tt mse}(\mathtt{idx})$ &emsp; _(residual vector)_
5. $\mathtt{qjl} \gets \mathtt{sign} \left( \mathbf{S} \cdot \mathbf{r} \right)$ &emsp; _(QJL on residual vector)_
6. **output:** $(\mathtt{idx},\ \mathtt{qjl},\ \|\mathbf{r}\|_2)$

---

**Procedure** $\textsc{DeQuant}_{\tt prod}(\mathtt{idx},\ \mathtt{qjl},\ \gamma)$

7. $\tilde{\mathbf{x}}_{\tt mse} \gets \textsc{DeQuant}_{\tt mse}(\mathtt{idx})$
8. $\tilde{\mathbf{x}}_{\tt qjl} \gets \frac{\sqrt{\pi/2}}{d} \cdot \gamma \cdot \mathbf{S}^\top \cdot \mathtt{qjl}$
9. **output:** $\tilde{\mathbf{x}}_{\tt mse} + \tilde{\mathbf{x}}_{\tt qjl}$

---

We prove the main result for $\text{CortexQuant}_{\tt prod}$ in the following theorem.

**Theorem 2** (Performance guarantee: $\text{CortexQuant}_{\tt prod}$). _For any bit-width $b \ge 1$ and any vector $\mathbf{x} \in \mathbb{S}^{d-1}$, the procedure $\textsc{Quant}_{\tt prod}(\mathbf{x})$ in Algorithm 2 outputs an index vector $\mathtt{idx} \in [2^{b-1}]^d$ along with a sign vector $\mathtt{qjl} \in \{ -1, 1 \}^d$ and a positive number $\gamma \ge 0$. When these vectors and the scalar value are passed to the primitive $\textsc{DeQuant}_{\tt prod}(\mathtt{idx}, \mathtt{qjl}, \gamma)$, it produces a reconstructed vector $\tilde{\mathbf{x}} \in \mathbb{R}^d$ that for any vector $\mathbf{y} \in \mathbb{R}^d$ satisfies the following properties:\_

- _Expected inner-product: $\mathbb{E}_{\tilde{\mathbf{x}}}\left[ \left\langle \mathbf{y}, \tilde{\mathbf{x}} \right\rangle \right] = \langle \mathbf{y}, \mathbf{x} \rangle$\_
- _Inner-product distortion defined as $D_{\tt prod} := \mathbb{E}_{\tilde{\mathbf{x}}}\left[\left| \langle \mathbf{y}, \mathbf{x} \rangle - \langle \mathbf{y}, \tilde{\mathbf{x}} \rangle \right|^2 \right]$ is bounded by $D_{\tt prod} \le \frac{\sqrt{3} \pi^2 \cdot \|\mathbf{y}\|_2^2}{d} \cdot \frac{1}{4^{b}}$ for any $b \ge 0$._
- _For small bit-widths, specifically $b = 1, 2, 3, 4$, $D_{\tt prod}$ exhibits finer-grained distortion values: $D_{\tt prod} \approx \frac{\mathbf{1.57}}{d}, \frac{\mathbf{0.56}}{d}, \frac{\mathbf{0.18}}{d}, \frac{\mathbf{0.047}}{d}$, respectively.\_

**Proof.** First we compute the conditional expectation of the inner product estimate $\langle \mathbf{y}, \tilde{\mathbf{x}} \rangle$ conditioned on $\tilde{\mathbf{x}}_{\tt mse}$ as follows:

$$
\begin{align}
\mathbb{E} \left[ \langle \mathbf{y}, \tilde{\mathbf{x}} \rangle | \tilde{\mathbf{x}}_{\tt mse} \right] &= \mathbb{E}_{\tilde{\mathbf{x}}_{\tt qjl}} \left[ \langle \mathbf{y}, \tilde{\mathbf{x}}_{\tt mse} + \tilde{\mathbf{x}}_{\tt qjl} \rangle | \tilde{\mathbf{x}}_{\tt mse} \right]\\
&= \langle \mathbf{y}, \tilde{\mathbf{x}}_{\tt mse} \rangle + \mathbb{E}_{\tilde{\mathbf{x}}_{\tt qjl}} \left[ \langle \mathbf{y}, \tilde{\mathbf{x}}_{\tt qjl} \rangle | \tilde{\mathbf{x}}_{\tt mse} \right] \\
&= \langle \mathbf{y}, \tilde{\mathbf{x}}_{\tt mse} \rangle + \langle \mathbf{y} , \mathbf{r} \rangle\\
&= \langle \mathbf{y} , \mathbf{x} \rangle
\end{align}
$$

where the first equality follows from the definition of $\tilde{\mathbf{x}}$ in step 9 of Algorithm 2. The third equality above follows from Lemma 4 and the last line follows from definition of the residual vector $\mathbf{r} = \mathbf{x} - \tilde{\mathbf{x}}_{\tt mse}$. Now we can compute the unconditional expectation using the law of total expectation: $\mathbb{E}_{\tilde{\mathbf{x}}}\left[ \left\langle \mathbf{y}, \tilde{\mathbf{x}} \right\rangle \right] = \mathbb{E}_{\tilde{\mathbf{x}}_{\tt mse}} \left[\mathbb{E} \left[ \langle \mathbf{y}, \tilde{\mathbf{x}} \rangle | \tilde{\mathbf{x}}_{\tt mse} \right] \right] = \mathbb{E}[\langle \mathbf{y} , \mathbf{x} \rangle] = \langle \mathbf{y} , \mathbf{x} \rangle$, which proves the first claim of the theorem.

We apply the same conditioning on $\tilde{\mathbf{x}}_{\tt mse}$, when computing the distortion, and then compute the resulting conditional distortion:

$$
\begin{align}
&\mathbb{E}\left[\left. \left| \langle \mathbf{y}, \mathbf{x} \rangle - \langle \mathbf{y}, \tilde{\mathbf{x}} \rangle \right|^2 \right| \tilde{\mathbf{x}}_{\tt mse} \right]\\
&= \mathbb{E}_{\tilde{\mathbf{x}}_{\tt qjl}}\left[\left. \left| \langle \mathbf{y}, \mathbf{x} \rangle - \langle \mathbf{y}, \tilde{\mathbf{x}}_{\tt mse} + \tilde{\mathbf{x}}_{\tt qjl} \rangle \right|^2 \right| \tilde{\mathbf{x}}_{\tt mse} \right] \\
&= \mathbb{E}_{\tilde{\mathbf{x}}_{\tt qjl}}\left[\left. \left| \langle \mathbf{y}, \mathbf{r} \rangle - \langle \mathbf{y}, \tilde{\mathbf{x}}_{\tt qjl} \rangle \right|^2 \right| \tilde{\mathbf{x}}_{\tt mse} \right] \\
&= \mathtt{Var} \left( \left. \langle \mathbf{y}, \tilde{\mathbf{x}}_{\tt qjl} \rangle \right| \tilde{\mathbf{x}}_{\tt mse} \right) \\
&\le \frac{\pi}{2d} \cdot \|\mathbf{r}\|_2^2 \|\mathbf{y}\|_2^2
\end{align}
$$

where the second equality follows from the definitions of $\mathbf{r}$ and $\tilde{\mathbf{x}}_{\tt mse}$. The third line follows because $\mathbb{E}[\langle \mathbf{y}, \tilde{\mathbf{x}}_{\tt qjl} \rangle] = \langle \mathbf{y}, \mathbf{r} \rangle$, by Lemma 4. The last line follows from the variance bound of QJL estimator shown in Lemma 4 and using the fact that $\tilde{\mathbf{x}}_{\tt qjl}$ is re-scaled by $\gamma = \|\mathbf{r}\|$.

Now by law of total expectation along with the fact that $\mathbf{r} = \mathbf{x} - \tilde{\mathbf{x}}_{\tt mse}$ we can bound the inner product distortion as follows:

$$
\begin{align}
D_{\tt prod} &= \mathbb{E}_{\tilde{\mathbf{x}}_{\tt mse}} \left[ \mathbb{E}\left[\left. \left| \langle \mathbf{y}, \mathbf{x} \rangle - \langle \mathbf{y}, \tilde{\mathbf{x}} \rangle \right|^2 \right| \tilde{\mathbf{x}}_{\tt mse} \right] \right]\\
&\le \frac{\pi}{2 d} \cdot \|\mathbf{y}\|_2^2 \cdot \mathbb{E}[\|\mathbf{x} - \tilde{\mathbf{x}}_{\tt mse}\|_2^2]\\
&= \frac{\pi}{2 d} \cdot \|\mathbf{y}\|_2^2 \cdot D_{\tt mse}
\end{align}
$$

The theorem follows by invoking the MSE bounds from Theorem 1 with bit-width $b-1$.

### 3.3 Lower Bounds

We show that CortexQuant achieves an optimal distortion rate, up to a small constant factor, for any bit-width by proving lower bounds on the best achievable distortion for any compression algorithm. Our lower bound proof leverages Yao's minimax principle. This principle allows us to relate the lower bound for randomized algorithms with worst-case deterministic input vectors to the lower bound for deterministic algorithms with randomized input vectors. Subsequently, we derive a lower bound on the achievable distortion rate for the latter using Shannon's lower bound (SLB) presented in Section 2.1. Formally, we prove the following theorem.

**Theorem 3** (Lower bound on best achievable compression distortion). _For any randomized quantization algorithm $Q: \mathbb{S}^{d-1} \to \{ 0, 1 \}^{b \cdot d}$ with bit-width $b$ and any reconstruction map $Q^{-1}: \{ 0, 1 \}^{b \cdot d} \to \mathbb{R}^d$, there exist a hard input instance $\mathbf{x} \in \mathbb{S}^{d-1}$ such that:_

$$D_{\tt mse}(Q) := \mathbb{E}\left[\left\| \mathbf{x} - Q^{-1}\left( Q(\mathbf{x}) \right) \right\|_2^2 \right] \ge \frac{1}{4^{b}}$$

_Furthermore, there exists a $\mathbf{y} \in \mathbb{S}^{d-1}$ such that:_

$$D_{\tt prod}(Q) = \mathbb{E}\left[\left| \langle \mathbf{y}, \mathbf{x} \rangle - \langle \mathbf{y}, Q^{-1}\left( Q(\mathbf{x}) \right) \rangle \right|^2 \right] \ge \frac{1}{d} \cdot \frac{1}{4^{b}}$$

**Proof.** By Yao's minimax principle the expected MSE of the optimal randomized compression algorithm for worst-case inputs ($D_{\tt mse}$) is equal to the expected MSE of the optimal deterministic compression algorithm when applied to inputs drawn from a maximally difficult randomized distribution. By definition, the MSE of the latter scenario is lower-bounded by the best achievable MSE for inputs uniformly distributed on the unit hypersphere.

The best achievable MSE for a compression algorithm with bit-width $b$, operating on uniformly distributed inputs from the sphere $\mathbb{S}^{d-1}$, is lower bounded in Lemma 3. Therefore, by invoking Lemma 3 we conclude that $D_{\tt mse} \ge \frac{1}{4^b}$.

Furthermore, from $D_{\tt mse} \ge \frac{1}{4^b}$ and using the definition of $D_{\tt mse}$ we conclude that:

$$
\begin{align}
D_{\tt mse} &= \sum_{j=1}^d \mathbb{E} \left[\left| \mathbf{x}_j - \left[ Q^{-1}\left( Q(\mathbf{x}) \right) \right]_j \right|^2 \right]\\
&= \sum_{j=1}^d \mathbb{E} \left[\left| \langle \mathbf{e}_j, \mathbf{x} \rangle - \langle \mathbf{e}_j, Q^{-1}\left( Q(\mathbf{x}) \right) \rangle \right|^2 \right]\\
&\ge \frac{1}{4^b}
\end{align}
$$

By pigeonhole principle there exist an index $j \in [d]$ such that $\mathbb{E} \left[\left| \langle \mathbf{e}_j, \mathbf{x} \rangle - \langle \mathbf{e}_j, Q^{-1}\left( Q(\mathbf{x}) \right) \rangle \right|^2 \right] \ge \frac{1}{d} \cdot \frac{1}{4^b}$, which completes the proof.

We note that a comparable lower bound for the _worst-case_ distortion in vector quantization can be derived using "sphere packing" arguments. However, Theorem 3 offers a more robust and relevant lower bound for our analysis. This is because it establishes a lower bound on the _expected distortion_, rather than the worst-case error, and aligns seamlessly with our upper bounds presented in Theorem 1 and Theorem 2.

---

## 4. Experiments

All experiments are performed using a single NVIDIA A100 GPU. The experimental section is divided into two parts: one to empirically validate the theoretical results, and another to evaluate the performance of our methods on downstream tasks, specifically KV cache quantization and nearest neighbor vector search.

### 4.1 Empirical Validation

In this section, we verify the theoretical results established in previous sections. We conduct our experiments using the DBpedia Entities dataset, which has been encoded into a 1536-dimensional space using OpenAI3 embeddings. To perform our experiments, we randomly sample 100,000 data points from the dataset, denoted as training set, which serves as our primary dataset. Additionally, we extract 1,000 distinct entries, denoted as query set, to be used as query points.

We evaluate two quantization methods: $\text{CortexQuant}_{\tt prod}$ and $\text{CortexQuant}_{\tt mse}$. The method $\text{CortexQuant}_{\tt mse}$ is designed to be optimized for estimating the mean squared error (MSE) between the quantized and original vectors. In contrast, $\text{CortexQuant}_{\tt prod}$ is unbiased for estimating the inner product between the quantized and original vectors.

Both methods are applied to the task of inner product estimation by quantizing training set and analyzing the distortion in inner product calculations across different bit widths. Increasing the bit width reduces variance in both methods. However, when used for inner product estimation, $\text{CortexQuant}_{\tt mse}$ introduces bias. This bias diminishes as the bit width increases and eventually converges to zero.

The experimental results confirm that $\text{CortexQuant}_{\tt prod}$ remains unbiased for inner product estimation across all bit widths, while $\text{CortexQuant}_{\tt mse}$ gradually improves with increasing bit width.

When quantizing to 2 bits, the variance remains constant regardless of the inner product of the original vector in the $\text{CortexQuant}_{\tt prod}$ approach. However, the bias in the $\text{CortexQuant}_{\tt mse}$ approach is dependent on the average inner product. As the average inner product increases, the bias also increases.

Along with the histograms, the average inner product error and MSE between the original and quantized vectors across different bit ratios are plotted alongside the upper and lower bounds established in our theoretical analysis. Our observations confirm that the results align with the theoretical predictions. Specifically, for inner product estimation, the $\text{CortexQuant}_{\tt prod}$ approach performs better at lower bit ratios. However, as the bit count increases, $\text{CortexQuant}_{\tt mse}$ reduces bias and ultimately achieves superior performance in inner product estimation.

### 4.2 Needle-In-A-Haystack

The "Needle-In-A-Haystack Test" is a benchmark designed to evaluate a model's ability to retrieve specific information embedded within a long document. The test involves placing a unique sentence (the "needle") at an arbitrary location within a much larger text (the "haystack") and assessing whether the model can successfully extract it.

Following the experimental setup of Fu et al. [fu2024data], we conduct evaluations using the `Llama-3.1-8B-Instruct` model. To analyze performance across different input sequence lengths, we vary the document size from _4k to 104k tokens_. The primary metric used for evaluation is the _recall score_, which measures how accurately the model retrieves the hidden sentence.

For comparison, we benchmark our approach against several state-of-the-art memory-efficient methods, including PolarQuant [han2025polarquant], SnapKV [li2024snapkv], PyramidKV [cai2024pyramidkv], and KIVI [liu2024kivi]. Each method is tested under a memory compression ratio of 0.25, meaning that only 25% of the full KV cache is utilized.

The results reveal that quantization methods with theoretical guarantees, such as PolarQuant and CortexQuant, outperform token-level compression techniques like SnapKV and PyramidKV, as well as scalar quantization approaches like KIVI, which lack formal theoretical guarantees. Notably, CortexQuant achieves identical performance to the full-precision model, even at $4\times$ compression, making it a robust solution for long-context processing.

| Method         | Score     |
| -------------- | --------- |
| SnapKV         | 0.858     |
| PyramidKV      | 0.895     |
| KIVI           | 0.981     |
| PolarQuant     | 0.995     |
| Full-Precision | 0.997     |
| **CortexQuant** | **0.997** |

### 4.3 End-to-end Generation on LongBench

We experiment with various KV cache compression algorithms on the LongBench dataset [bai2023longbench], which encompasses a broad range of long-text scenarios, including single- and multi-document question-answering, summarization, few-shot learning, synthetic tasks, and code completion. To ensure a balanced evaluation across different context lengths, we employ **LongBench-E**, a subset designed with a more uniform length distribution.

We compare CortexQuant against the leading baseline methods using both `Llama-3.1-8B-Instruct` and `Ministral-7B-Instruct`. Unlike existing approaches such as **KIVI** and **PolarQuant**, which leave generated tokens unquantized, our method applies quantization even during the streaming generation process.

Our approach outperforms other methods for both models, achieving significantly higher average scores. We evaluate our method using **2.5-bit** and **3.5-bit** quantization during text generation. These non-integer bit precisions result from our strategy of splitting channels into outlier and non-outlier sets, and applying two independent instances of CortexQuant to each, allocating higher bit precision to outliers. For example, in our 2.5-bit setup, 32 outlier channels are quantized at 3 bits, while the remaining 96 channels use 2 bits, leading to an effective bit precision of $(32 \times 3 + 96 \times 2) / 128 = 2.5$. Despite using fewer bits than competing techniques, CortexQuant maintains performance comparable to unquantized models, while compressing quantized vectors by at least a factor of $4.5\times$.

**Table: LongBench-V1 results of various KV cache compression methods on `Llama-3.1-8B-Instruct`.**

| Method                    | KV Size | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code  | Average   |
| ------------------------- | ------- | -------- | ------- | ------------- | -------- | --------- | ----- | --------- |
| **Llama-3.1-8B-Instruct** |         |          |         |               |          |           |       |           |
| Full Cache                | 16      | 45.29    | 45.16   | 26.55         | 68.38    | 59.54     | 46.28 | **50.06** |
| KIVI                      | 3       | 43.38    | 37.99   | 27.16         | 68.38    | 59.50     | 44.68 | 48.50     |
| KIVI                      | 5       | 45.04    | 45.70   | 26.47         | 68.57    | 59.55     | 46.41 | 50.16     |
| PolarQuant                | 3.9     | 45.18    | 44.48   | 26.23         | 68.25    | 60.07     | 45.24 | 49.78     |
| CortexQuant (ours)         | 2.5     | 44.16    | 44.96   | 24.80         | 68.01    | 59.65     | 45.76 | 49.44     |
| CortexQuant (ours)         | 3.5     | 45.01    | 45.31   | 26.00         | 68.63    | 59.95     | 46.17 | **50.06** |
| **Ministral-7B-Instruct** |         |          |         |               |          |           |       |           |
| Full Cache                | 16      | 47.53    | 49.06   | 26.09         | 66.83    | 53.50     | 47.90 | **49.89** |
| CortexQuant (ours)         | 2.5     | 48.38    | 49.22   | 24.91         | 66.69    | 53.17     | 46.83 | 49.62     |

### 4.4 Near Neighbour Search Experiments

In this section, we establish the strength of our proposed method, even in the context of near-neighbor search. We conduct our experiments using the DBpedia [thakur2021beir] Entities dataset, which has been encoded into 1536-dimensional and 3072-dimensional spaces using OpenAI3 embeddings. Additionally, we evaluate performance on a lower-dimensional dataset, utilizing the standard GloVe [pennington2014glove] embeddings. To construct our experimental setup, we randomly sample 100,000 data points from the dataset (training set), and extract 1,000 distinct entries (query set). For the GloVe dataset, we use a pre-existing query set consisting of 10,000 points.

We compare CortexQuant against two baseline quantization approaches: Product Quantization (PQ) and RabitQ [gao2024practical]. We quantize the dataset training set using all three methods and evaluate their performance based on recall ratio at top-k, denoted as 1@k. Specifically, this metric assesses how often the true top inner product result is captured within the top-k approximated results returned by each algorithm.

**Table: Quantization time (in seconds) for different approaches across various dimensions using 4-bit quantization.**

| Approach             | d=200  | d=1536  | d=3072  |
| -------------------- | ------ | ------- | ------- |
| Product Quantization | 37.04  | 239.75  | 494.42  |
| RabitQ               | 597.25 | 2267.59 | 3957.19 |
| CortexQuant           | 0.0007 | 0.0013  | 0.0021  |

**Product Quantization (PQ)** relies on the k-means algorithm to construct codebooks, which require separate storage. As the number of bits increases, the size of the codebook grows exponentially, leading to additional storage overhead. In our experiments, we carefully tuned the parameters to match the bit allocation of other methods. The most efficient implementation, designed for rapid querying, employs AVX2 In-Register Lookup Tables (LUTs). Specifically, it uses LUT16 with (l = 16) codewords. However, we observed substantial quality degradation at this configuration. To achieve a balance between speed and accuracy, we opted for a version of PQ that uses LUT256, which contains 256 codewords. For 2-bit quantization, it groups 4 coordinates per lookup, while for 4-bit quantization, it groups 2 coordinates per lookup. Notably, since we use the same dataset for both training and evaluation, PQ benefits from an inherent advantage in this setup.

**RabitQ.** Unlike PQ, RabitQ lacks a fully vectorized implementation, making it impossible to leverage GPU acceleration. As a result, it runs significantly slower on CPU. Additionally, the method incurs extra computational overheads that we do not explicitly account for in the bit ratio comparisons. While RabitQ claims a certain bit ratio, in practice, it utilizes more bits than reported due to these inefficiencies.

Despite the advantages granted to the baseline methods, CortexQuant consistently outperforms both Product Quantization and RabitQ in terms of recall ratio across all experiments. This demonstrates the robustness and efficiency of our approach, making it a compelling alternative for high-dimensional quantization-based search tasks.
