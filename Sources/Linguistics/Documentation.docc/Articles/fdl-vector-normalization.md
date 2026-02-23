# FDL Vector Normalization

Understand why ``FDLEmbeddingService`` returns unnormalized vectors and how to normalize
them when your use case requires it.

## Overview

All embedding services in this package conform to ``EmbeddingProvider`` and can be used
interchangeably at call sites. However, there is one important behavioral difference
between ``FDLEmbeddingService`` and the other two providers:

| Provider | ``EmbeddingProvider/embed(_:)`` return value |
|---|---|
| ``NLEmbeddingService`` | L2-normalized vector (magnitude ≈ 1) |
| ``MLXEmbeddingService`` | L2-normalized vector (magnitude ≈ 1) |
| ``FDLEmbeddingService`` | Raw frequency-count vector (magnitude = √Σcountᵢ²) |

## Why FDL Vectors Are Not Normalized

``FDLEmbeddingService`` builds a fixed vocabulary from a corpus and then embeds each
document as a vector of **occurrence counts** — position *i* holds the number of times
vocabulary token *i* appears in the document after lemmatization and stop-word removal.

These raw counts are intentionally left unnormalized because:

- They can be used directly as input to classifiers or downstream TF-IDF weighting
  without an extra copy.
- The magnitude carries information: a longer document will naturally have higher counts,
  and some pipelines treat that as a feature rather than a nuisance.

## Consequences for `similarity(between:and:)`

The default ``EmbeddingProvider/similarity(between:and:)`` implementation computes the
**dot product** of the two vectors:

```swift
Float(zip(a, b).map(*).reduce(0, +))
```

For L2-normalized vectors this equals cosine similarity and is bounded to `[-1, 1]`.
For raw FDL count vectors it is an unnormalized dot product that grows with document
length. Two long documents that share many words will score far higher than two short
documents expressing the same idea, which is rarely the desired behavior.

> Important: Do not compare FDL similarity scores with scores produced by
> ``NLEmbeddingService`` or ``MLXEmbeddingService`` — the scales are incompatible.

## Normalizing When You Need To

If you need cosine-equivalent similarity from FDL vectors, normalize the result of
``FDLEmbeddingService/embed(_:)`` using the `.normal` property provided by the
`Vector` type in the MatrixStuff package:

```swift
let corpus: [String] = loadDocuments()
let service = FDLEmbeddingService(corpus: corpus)

let rawVector   = try await service.embed(someText)
let unitVector  = rawVector.normal          // L2-normalized copy; magnitude == 1
```

`Vector.normal` divides every element by `Vector.magnitude` (the Euclidean length √Σxᵢ²),
returning a copy whose magnitude is 1. The original vector is unchanged.

```swift
// Cosine similarity between two normalized FDL vectors
let a = try await service.embed(textA)
let b = try await service.embed(textB)
let cosine = Float(zip(a.normal, b.normal).map(*).reduce(0, +))
// cosine is now in [-1, 1], comparable to NLEmbedding / MLX scores
```

> Note: Normalizing after the fact is mathematically equivalent to L2-normalizing inside
> `embed(_:)`, so you can choose whichever approach better matches your pipeline.

## When to Leave Vectors Unnormalized

Raw frequency-count vectors are the right choice when:

- Feeding them into a **linear classifier** or **logistic regression** that expects
  non-normalized bag-of-words features.
- Applying **TF-IDF weighting** — multiply each count by the inverse document frequency
  before normalizing, so that rare terms are upweighted.
- Using a **distance metric** (such as Euclidean distance) rather than cosine similarity,
  where magnitude is meaningful.

## See Also

- ``FDLEmbeddingService``
- ``EmbeddingProvider``
- ``NLEmbeddingService``
- ``MLXEmbeddingService``
