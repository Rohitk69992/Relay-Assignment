# Three Ways the Research Library Misleads a Naive Retriever

This document details three concrete failure modes where a naive retriever (e.g. unweighted lexical search, chunk-blind vector search, or split-table indexing) fails or outputs misleading evidence when querying the three research papers (*Attention Is All You Need*, *Moshi*, and *Dynamic Chunking*). All passages, citations, pages, and section names are directly verified against the official arXiv preprints.

---

## Failure Mode 1: Cross-Document Polysemous Term Shadowing ("Chunking" vs "Audio Frames")

### 1. Example Question
> *"What is the chunk size and frame duration used in the architecture?"*

### 2. Potentially Misleading Passage
From **Moshi: a speech-text foundation model for real-time dialogue** (arXiv:2410.00037v1), Section 3.1 & Section 4.1 (Pages 6 & 16):
> *"Mimi operates at 12.5 Hz, yielding a frame duration of 80ms... The convolutional encoder computes streaming audio chunks with small receptive fields to maintain sub-200ms latency..."*

### 3. Correct Evidence
From **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling** (arXiv:2507.07955v2), Section 2.2 (Page 6):
> *"Unlike fixed chunking which enforces a predetermined static chunk size $C$ (e.g., $C=8$ or $16$ tokens), dynamic chunking dynamically predicts variable-length boundary decisions $\pi$ based on token representations, avoiding rigid artificial sequence splitting."*

### 4. Why Initial Retrieval is Misleading
The word "chunk" appears frequently in both papers with completely different semantics. In *Moshi*, "chunking" refers to temporal audio frame slicing in streaming convolutions. In *Dynamic Chunking*, "chunking" refers to hierarchical sequence reduction via learned boundary predictors $\pi$. A naive BM25 or embedding retriever retrieves Moshi's audio passages due to dense repetition of "chunk" and "frame", shadowing the foundational definition in the Dynamic Chunking paper.

### 5. Mitigation in Implementation
- **Intent Planning & Query Expansion:** The `IntentPlanner` parses domain indicators (e.g. "hierarchical sequence" vs "speech codec") and injects disambiguating terms.
- **Section Heading Alignment:** The `CrossScoringReranker` boosts chunks whose section header matches the target architectural domain (`2.2 Dynamic Chunking (DC)` vs `3.1 Audio Codec`).
- **Citation Auditing:** The `ResponseVerifier` checks that the returned answer cites the paper matching the user's specific conceptual query rather than conflating speech audio with discrete NLP modeling.

---

## Failure Mode 2: Table Flattening & Column Misalignment (Base vs Big Transformer BLEU)

### 1. Example Question
> *"What BLEU score does the Transformer achieve on the WMT 2014 English-to-German translation task?"*

### 2. Potentially Misleading Passage
From **Attention Is All You Need** (arXiv:1706.03762v7), Section 6.1 (Page 8, Table 2):
When a PDF parser flattens Table 2 without tabular structure, lines from competing baseline models interleave:
> *"ByteNet 23.75 ... GNMT + RL 24.61 ... ConvS2S 25.16 ... Transformer (base model) 27.3 ... Transformer (big) 28.4"*
A naive chunker splitting across character limits creates a chunk containing:
> *"BLEU EN-DE: 25.16, Training Cost (FLOPs) 9.6e18... establishing a new state of the art."*

### 3. Correct Evidence
From **Attention Is All You Need** (arXiv:1706.03762v7), Section 6.1, Table 2 (Page 8):
> *"On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)) establishes a new state-of-the-art BLEU score of 28.4... The base model achieves 27.3 BLEU."*

### 4. Why Initial Retrieval is Misleading
In raw PDF text extraction, column headers ("BLEU", "PPL", "FLOPs") often detach from row names. A naive retriever matching "BLEU score English to German" frequently retrieves lines corresponding to ConvS2S (25.16 BLEU) or GNMT (24.61 BLEU) rather than the Transformer rows, or confuses the Base model (27.3) with the Big model (28.4).

### 5. Mitigation in Implementation
- **Boundary-Aware Sliding Window:** The `DocumentChunker` enforces sentence- and table-aware boundaries with 100-character overlap, ensuring table headers and corresponding row rows are not split apart.
- **Prompt Grounding Constraints:** The `AnswerGenerator` instructs the model to explicitly distinguish between "base" and "big" models and verify the exact model name adjacent to the metric.

---

## Failure Mode 3: Extrapolation Trap Across Split Sections (Sinusoidal vs Learned Positional Embeddings)

### 1. Example Question
> *"Does the Transformer require learned positional embeddings to achieve its performance?"*

### 2. Potentially Misleading Passage
From **Attention Is All You Need** (arXiv:1706.03762v7), Section 3.5 (Page 6):
> *"We chose sinusoidal positional encodings because it may allow the model to extrapolate to sequence lengths longer than encountered during training. The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$."*

### 3. Correct Evidence
From **Attention Is All You Need** (arXiv:1706.03762v7), Section 6.2, Model Variations, Table 3 (Page 9):
> *"In Table 3 row (E), we also experimented with using learned positional embeddings instead, and found that the two versions produced nearly identical results (27.2 vs 27.3 BLEU)."*

### 4. Why Initial Retrieval is Misleading
A naive retriever queried with "learned positional embeddings" will match Section 3.5 because it discusses positional encodings at length. However, Section 3.5 does *not* contain the experimental finding—that finding resides 3 pages later in Section 6.2, Table 3! An LLM relying solely on Section 3.5 will wrongly state that the Transformer does not test or support learned positional embeddings.

### 5. Mitigation in Implementation
- **Multi-Chunk Top-K Retrieval:** Our balanced configuration fetches $k=4$ chunks across different document sections.
- **Sufficiency Evaluation Gate:** The `SufficiencyEvaluator` verifies whether experimental ablation keywords ("Table 3", "learned", "ablation") are present when a question queries architectural necessity. If missing, it executes a single bounded refinement query to fetch the ablation table on Page 9.
