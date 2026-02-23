# ``Linguistics``

NLP and ML primitives for text embeddings, semantic reranking, and language analysis on Apple platforms.

## Overview

The Linguistics package provides a unified interface for converting text into numeric vectors
(embeddings), comparing those vectors for semantic similarity, and reranking candidate documents
against a query. Three embedding backends are available:

| Provider | Dimensions | Offline | GPU |
|---|---|---|---|
| ``NLEmbeddingService`` | 512 | ✓ | — |
| ``FDLEmbeddingService`` | corpus-dependent | ✓ | — |
| ``MLXEmbeddingService`` | 384–1024 | after download | ✓ |

All three conform to the ``EmbeddingProvider`` protocol, so they can be swapped without
changing call-site code. However, ``FDLEmbeddingService`` returns **raw frequency-count
vectors** rather than L2-normalized vectors — see
<doc:fdl-vector-normalization> for the implications and how to normalize when needed.

## Topics

### Embedding Protocol

- ``EmbeddingProvider``
- ``TextEmbedding``
- ``EmbeddingProviderOption``

### Embedding Services

- ``NLEmbeddingService``
- ``FDLEmbeddingService``
- ``MLXEmbeddingService``

### Reranking

- ``Reranker``
- ``EmbeddingReranker``
- ``MLXCrossEncoderReranker``

### Benchmarking & Calibration

- ``EmbeddingBenchmark``
- ``ThresholdCalibrator``

### Text Analysis

- ``POSFilter``

### Errors

- ``NLEmbeddingError``

### Articles

- <doc:fdl-vector-normalization>
