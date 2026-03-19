# Linguistics

A Swift 6.2 SPM library for text embeddings, semantic reranking, and NLP analysis on Apple platforms. It provides a unified `EmbeddingProvider` protocol backed by Apple's NaturalLanguage framework or GPU-accelerated MLX transformer models, tools for loading and embedding research manuscripts and academic program data, and utilities for benchmarking and threshold calibration.

**Platforms:** macOS 14+ · iOS 17+
**Swift:** 6.2 · Strict concurrency enabled

---

## Features

- **Three embedding backends** — offline Apple NLEmbedding (512d), GPU transformer models via MLX (384–1024d), and corpus-based frequency vectors
- **Two reranking strategies** — embedding-based bi-encoder and MLX cross-encoder for two-stage retrieval pipelines
- **Document loaders** — parse Markdown manuscripts and academic-program CSV files into labeled embedding corpora
- **String analysis extensions** — sentiment, readability (ARI), tokenization, stop-word removal, and POS-filtered lemmatization
- **Benchmarking & calibration** — 10 built-in test sets, threshold sweeping, and precision/recall/F1 reporting
- **SwiftUI comparison view** — interactive UI for evaluating and comparing embedding models on custom data

---

## Installation

Add the package in Xcode via **File › Add Package Dependencies**, or in `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/dyerlab/Linguistics", from: "1.0.0")
],
targets: [
    .target(name: "MyTarget", dependencies: ["Linguistics"])
]
```

> MLX models are downloaded to `~/.cache/huggingface/hub/` on first use. Apple NLEmbedding and FDL require no downloads.

---

## Core Concepts

### `EmbeddingProvider`

The central protocol. All backends conform to it:

```swift
public protocol EmbeddingProvider: Sendable {
    func embed(_ text: String) async throws -> Vector
    func embedBatch(_ texts: [String]) async throws -> [Vector]
    func similarity(between: String, and: String) async throws -> Float
    var dimensions: Int { get async throws }
}
```

Vectors are `MatrixStuff.Vector` values. All providers except `FDLEmbeddingService` return L2-normalized vectors — dot product equals cosine similarity.

### `TextEmbedding`

A `Codable`, `Sendable` struct that bundles a vector with its provenance and optional metadata:

```swift
public struct TextEmbedding: Sendable, Codable, Hashable {
    let provider: EmbeddingProviderOption  // which model made this
    let vector: Vector                      // L2-normalized embedding
    let scaling: Double                     // optional weight (default 1.0)
    let metadata: [String: String]          // caller-defined labels
}
```

Create one from any provider:

```swift
let embedding = try await provider.embed("Hello world", as: .nlEmbedding)
```

### `Corpus`

An immutable, `Codable` collection of `TextEmbedding` values from a single source (a paper, a program, a document):

```swift
public struct Corpus: Sendable, Codable, Identifiable, Hashable {
    let id: UUID
    let label: String
    let metadata: [String: String]
    let embeddings: [TextEmbedding]
}
```

---

## Embedding Providers

### NLEmbeddingService — Offline, Instant

Uses Apple's on-device `NLEmbedding` word vectors with average pooling. No downloads, no GPU required. 512-dimensional.

```swift
let service = try NLEmbeddingService()

let vector = try await service.embed("transformer architecture")
let score  = try await service.similarity(between: "cat", and: "kitten")

// Word-level operations
let neighbors = service.neighbors(for: "neural", count: 5)
let distance  = service.distance(from: "happy", to: "joyful")
```

### MLXEmbeddingService — GPU Transformer Models

GPU-accelerated sentence transformers via MLX. Requires Metal (macOS/iOS device). Models are downloaded from HuggingFace Hub on first use.

```swift
// Default: mxbai-embed-large (1024d, ~1.2 GB)
let service = try await MLXEmbeddingService()

// Choose a model
let fast    = try await MLXEmbeddingService(model: .miniLM)         // 384d, ~90 MB
let balanced = try await MLXEmbeddingService(model: .bgeBase)       // 768d, ~400 MB
let quality  = try await MLXEmbeddingService(model: .bgeLarge)      // 1024d, ~1.2 GB
let quantized = try await MLXEmbeddingService(model: .qwen3Embedding) // 4-bit
let matryoshka = try await MLXEmbeddingService(model: .nomicTextV1_5)

// Custom HuggingFace model
let custom = try await MLXEmbeddingService(model: .custom("sentence-transformers/all-mpnet-base-v2"))

// Download progress
let service = try await MLXEmbeddingService(model: .bgeLarge) { progress in
    print("Downloading: \(Int(progress * 100))%")
}
```

### FDLEmbeddingService — Corpus Frequency Vectors

Builds a vocabulary from a corpus at init time. Each embedding is a raw frequency-count vector over the vocabulary. Useful for domain-specific bag-of-words comparisons.

```swift
let documents = ["Introduction to machine learning...", "Neural networks and deep learning..."]
let service = FDLEmbeddingService(corpus: documents)

let vector = try await service.embed("supervised learning methods")

// FDL vectors are NOT L2-normalized — normalize before cosine similarity
let normalized = vector.normal  // MatrixStuff Vector.normal
```

### `EmbeddingProviderOption`

A `Codable` enum for tagging vectors with their provenance. Also a factory:

```swift
let provider = try await EmbeddingProviderOption.bgeBase.makeProvider()
// For FDL, pass the corpus:
let fdl = try await EmbeddingProviderOption.fdlEmbedding.makeProvider(corpus: myDocs)
```

Each case has a `displayName`, `abbreviation`, `color` (SwiftUI), and `requiresDownload` flag.

---

## Reranking

### EmbeddingReranker — Bi-Encoder (Fast)

Wraps any `EmbeddingProvider`. Encodes the query once, then scores all documents via dot product.

```swift
let service = try NLEmbeddingService()
let reranker = EmbeddingReranker(provider: service)

let results = try await reranker.rerank(
    query: "machine learning applications",
    documents: myDocuments,
    topK: 10
)

for result in results {
    print("\(result.score): \(result.item)")
}
```

### MLXCrossEncoderReranker — Cross-Encoder (Accurate)

Jointly encodes query+document pairs. Significantly more accurate than bi-encoders for re-ranking, at the cost of latency.

```swift
let reranker = try await MLXCrossEncoderReranker(model: .bgeRerankerBase)
// Also: .bgeRerankerLarge, .bgeRerankerV2M3 (multilingual), .custom("hub-id")

let results = try await reranker.rerank(
    query: "climate change effects on biodiversity",
    documents: abstractTexts,
    topK: 5
)
```

### Generic Reranking

Rerank any `Sendable` type with a text extractor:

```swift
struct Article: Sendable { let title: String; let body: String }

let ranked = try await reranker.rerank(
    query: "protein folding",
    items: articles,
    topK: 3,
    textExtractor: { "\($0.title) \($0.body)" }
)
```

### Two-Stage Pipeline

The recommended pattern for large corpora:

```swift
// Stage 1: Fast embedding retrieval (top-100)
let embedder = EmbeddingReranker(provider: try await MLXEmbeddingService(model: .miniLM))
let candidates = try await embedder.rerank(query: query, documents: allDocs, topK: 100)

// Stage 2: Accurate cross-encoder reranking (top-10)
let crossEncoder = try await MLXCrossEncoderReranker(model: .bgeRerankerV2M3)
let final = try await crossEncoder.rerank(
    query: query,
    items: candidates,
    topK: 10,
    textExtractor: { $0.item }
)
```

---

## Document Loaders

### ManuscriptLoader — Markdown Research Papers

Parses Markdown files produced by PDF-to-Markdown converters (e.g., `marker`, `nougat`). Classifies sections using a `DocumentProfile`, then embeds each section or paragraph.

```swift
let provider = try NLEmbeddingService()

// Load a single paper
let corpus = try await ManuscriptLoader.load(
    from: paperURL,
    profile: .scientificPaper,   // default — covers IMRaD structure
    granularity: .paragraph,     // .section or .paragraph
    using: provider,
    as: .nlEmbedding
)

print(corpus.label)              // first # heading = paper title
print(corpus.metadata["doi"])    // extracted from first 3 000 chars

// Load an entire directory
let corpora = try await ManuscriptLoader.loadAll(
    from: markdownDirectory,
    granularity: .section,
    using: provider,
    as: .nlEmbedding
)
```

Each `TextEmbedding` in the corpus carries:
- `metadata["part"]` — `ManuscriptParts` section type (Abstract, Introduction, Methods, Results, Discussion, Other)
- `metadata["granularity"]` — `"section"` or `"paragraph"`
- `metadata["text"]` — the source text

#### Custom Document Profiles

```swift
let labReport = DocumentProfile(
    id: "lab-report",
    displayName: "Lab Report",
    rules: [
        SectionRule(pattern: #"^(purpose|objective)$"#,  type: .Introduction),
        SectionRule(pattern: #"^(procedure|protocol)$"#, type: .Methods),
        SectionRule(pattern: #"^(data|observations?)$"#, type: .Results),
    ],
    fallbackType: .Other
)

let corpus = try await ManuscriptLoader.load(
    from: url, profile: labReport, granularity: .section,
    using: provider, as: .nlEmbedding
)
```

### AcademicProgramLoader — Course Catalog CSV

Loads a CSV describing university course catalogs and returns one `Corpus` per academic program. Each course is embedded once per university and reused across programs.

**CSV format** (header row required):

| University | Program | Course | Title | Credits | Bulletin |
|------------|---------|--------|-------|---------|----------|
| VCU | Biology | BIOL 101 | Principles of Biology | 3 | Introduction to cell biology... |

```swift
let provider = try await MLXEmbeddingService(model: .bgeBase)

let programs = try await AcademicProgramLoader.load(
    from: csvURL,
    using: provider,
    as: .bgeBase
)

for program in programs {
    print("\(program.metadata["university"]!) — \(program.label)")
    print("  \(program.embeddings.count) courses")
}
```

Each `TextEmbedding` carries:
- `metadata["course"]` — course code
- `metadata["text"]` — `"\(Title) \(Bulletin)"`
- `scaling` — credit hours (as `Double`)

A bundled sample dataset is included at `Sources/Linguistics/Data/vcu_stem_programs.csv`.

---

## Text Analysis

`String` extensions powered by Apple's `NaturalLanguage` framework.

### Sentiment

```swift
let text = "The results were surprisingly effective and well-received."

text.sentiment              // Double: –1.0 (negative) to 1.0 (positive)
text.sentimentScore         // averaged across paragraphs
text.sentenceLevelSentiment // [Double] — one per sentence
text.sentimentString        // emoji: "😊", "😐", "😞"
```

### Readability

```swift
text.ARI        // Automated Readability Index (grade level)
text.words      // word count
text.sentences  // sentence count
text.paragraphs // paragraph count
```

### Tokenization

```swift
let tokens = text.wordTokens(language: .english, minTokenLength: 3, lowercased: true)
let clean   = text.tokensWithoutStopwords()

// Access or extend the built-in stop-word list
var stops = String.englishStopwords
stops.insert("however")
let filtered = text.tokensWithoutStopwords(stopwords: stops)
```

### Lemmatization & POS Filtering

```swift
// Content lemmas: nouns, verbs, and adjectives — stop words removed by default
let lemmas = text.contentLemmas()
// e.g. ["result", "surprising", "effective", "receive"]
```

---

## Benchmarking & Calibration

### EmbeddingBenchmark

Measures a provider's ability to distinguish semantically similar from dissimilar pairs. The key metric is `discriminationGap` (avgHighSimilarity − avgLowSimilarity).

```swift
let benchmark = EmbeddingBenchmark()
let provider  = try NLEmbeddingService()

// Use a built-in test set
let result = try await benchmark.runWithReport(
    provider: provider,
    name: "NLEmbedding",
    pairs: EmbeddingBenchmark.scientificPairs
)

print(result.discriminationGap)  // higher is better
print(result.accuracy)
print(result.summary)

// runSafe skips pairs where NLEmbedding lacks vocabulary
let safeResult = await benchmark.runSafe(
    provider: provider,
    name: "NLEmbedding",
    pairs: EmbeddingBenchmark.allTestSets.flatMap(\.pairs)
)
```

**Built-in test sets:** `generalPairs`, `shortPhrasePairs`, `technicalPairs`, `questionPairs`, `scientificPairs`, `singleWordPairs`, `longPassagePairs`, `paraphrasePairs`, `retrievalPairs`, `conversationalPairs`.

### ThresholdCalibrator

Sweeps similarity thresholds over labeled examples to find the best operating point for your use case.

```swift
let calibrator = ThresholdCalibrator.semanticSearch  // preset sweep range
// Also: .duplicateDetection, .contentDiscovery

let examples: [LabeledPair] = [
    LabeledPair("neural network", "deep learning", isMatch: true),
    LabeledPair("neural network", "organic chemistry", isMatch: false),
]

let result = try await calibrator.calibrate(provider: provider, examples: examples)

print("Best F1 threshold: \(result.bestF1Threshold)")
print("Best precision threshold: \(result.bestPrecisionThreshold)")
print(result.report)  // full threshold sweep table

// Also calibrate a reranker
let rerankerResult = try await calibrator.calibrate(reranker: crossEncoder, examples: examples)
```

---

## SwiftUI Comparison View

An interactive view for evaluating embedding providers on custom text pairs. Drop it into any SwiftUI app to explore model selection, discrimination charts, and threshold recommendations.

```swift
import SwiftUI
import Linguistics

struct ContentView: View {
    var body: some View {
        EmbeddingComparisonView()
    }
}
```

Features:
- Enter custom text pairs or load from built-in test sets
- Run NLEmbedding-only (instant) or full GPU comparison
- View per-provider similarity distributions as charts
- Get automatic provider and threshold recommendations

---

## Architecture Notes

- **Swift 6 strict concurrency** throughout. `MLXEmbeddingService` and `MLXCrossEncoderReranker` are `actor`s to isolate Metal/MLX GPU state. All public types are `Sendable`.
- **`FDLEmbeddingService` is not L2-normalized** — call `vector.normal` (MatrixStuff) before computing cosine similarity or comparing with other providers.
- **Model files are never bundled** — MLX models are downloaded from HuggingFace Hub on first use.
- **Tests use Swift Testing** (`@Test`, `#expect`) not XCTest. MLX tests require Metal and must be run from Xcode — `swift test` from the command line runs NLEmbedding tests only.
- **`EmbeddingComparisonViewModel`** is `@Observable` + `@MainActor` — do not add `@Published` properties.

---

## License

MIT
