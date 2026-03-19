# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

`Linguistics` is a Swift 6.2 SPM library (macOS 14+ / iOS 17+) providing NLP and ML primitives: text embeddings, semantic reranking, text analysis (sentiment, readability, tokenization, lemmatization), document/manuscript loading, academic program corpus construction, and benchmarking utilities. It ships a SwiftUI comparison view for evaluating embedding models on custom data.

## Build & Test Commands

```bash
# Build
swift build

# Run tests (NLEmbedding tests only — MLX tests require Xcode/Metal)
swift test

# Run a single test by name
swift test --filter nlEmbeddingServiceInitialization

# Run all tests including MLX (must launch from Xcode — Metal shader compilation required)
# Use Product > Test in Xcode, or xcodebuild test
```

**Important**: `MLXEmbeddingService` and `MLXCrossEncoderReranker` tests fail when run via `swift test` from the command line because MLX requires Metal shader compilation (only available through Xcode). Tests annotated with `.timeLimit(.minutes(...))` all require GPU.

## Architecture

### Core Abstraction Layer

**`EmbeddingProvider`** (`Embeddings/EmbeddingProvider.swift`) — protocol with `async throws` requirements:
- `embed(_ text: String) async throws -> Vector` (MatrixStuff `Vector`, not `[Float]`)
- `embedBatch(_ texts: [String]) async throws -> [Vector]`
- `similarity(between:and:) async throws -> Float`
- `dimensions: Int { get async throws }`

All returned vectors are L2-normalized (except `FDLEmbeddingService` — see below). Default implementations of `embedBatch` and `similarity` are provided; concrete types override for efficiency. A convenience overload `embed(_:as:) async throws -> TextEmbedding` is provided via extension for provenance-tagged results.

**`Reranker`** (`Reranking/Reranker.swift`) — protocol for two-stage retrieval:
- `score(query:document:) async throws -> Float`
- `scoreBatch(query:documents:) async throws -> [Float]`
- `rerank(query:documents:topK:)` and generic `rerank(query:items:topK:textExtractor:)`

Default implementations handle sorting and topK slicing; concrete types only need to implement `score`.

### Embedding Implementations

| Class | Concurrency | Model | Dimensions | Downloads |
|-------|-------------|-------|------------|-----------|
| `NLEmbeddingService` | `@unchecked Sendable` class | Apple NLEmbedding (word vectors, avg pooling) | 512 | None |
| `MLXEmbeddingService` | `actor` | Transformer (mxbaiEmbedLarge default) | 384–1024 | HuggingFace Hub → `~/.cache/huggingface/hub/` |
| `FDLEmbeddingService` | `@unchecked Sendable` class | Corpus frequency-count (bag-of-words) | vocab size | None |

`MLXEmbeddingService.Model` enum exposes `.miniLM` (384d), `.bgeBase` (768d), `.bgeLarge` (1024d), `.mxbaiEmbedLarge` (1024d), `.qwen3Embedding`, `.nomicTextV1_5`, plus `.custom(String)` and `.directory(URL)`.

`NLEmbeddingService` additionally exposes word-level operations: `embedWord(_:)`, `contains(_:)`, `neighbors(for:count:)`, `distance(from:to:)`.

**MLX embedding approach** (`MLXEmbeddingService.embed`): tokenizes with the HuggingFace tokenizer, runs a forward pass, then applies mean pooling (not the model's `pooledOutput`) with L2 normalization — this matches sentence-transformer convention.

**FDL embedding approach** (`FDLEmbeddingService`): vocabulary built once from a corpus at `init(corpus:)` — texts are joined, lemmatized, stop-word filtered, and deduplicated into a sorted token list. `embed(_:)` returns a raw frequency-count `Vector` (one element per vocab token). **Not L2-normalized** — call `vector.normal` (MatrixStuff) before computing cosine similarity or comparing with other providers.

### Reranking Implementations

- **`EmbeddingReranker`** — lightweight struct that wraps any `EmbeddingProvider`; encodes the query once and dot-products against all document embeddings
- **`MLXCrossEncoderReranker`** — `actor`; concatenates query+document tokens, runs a forward pass, applies sigmoid to get 0–1 relevance scores; supports `.bgeRerankerBase`, `.bgeRerankerLarge`, `.bgeRerankerV2M3`, `.custom`

Typical pipeline: fast embedding retrieval (top-100) → cross-encoder reranker (top-10).

### Text Analysis (`Extensions/String.swift`)

`String` extensions using `NaturalLanguage`:
- **Sentiment**: `.sentiment` (paragraph), `.sentenceLevelSentiment`, `.paragraphLevelSentiment`, `.sentimentScore` (averaged), `.sentimentString` (emoji)
- **Readability**: `.ARI` (Automated Readability Index), `.words`, `.sentences`, `.paragraphs`
- **Tokenization**: `wordTokens(language:minTokenLength:lowercased:)`, `tokensWithoutStopwords(...)`, `englishStopwords` (static)
- **Lemmatization + POS filtering**: `linguisticTokens(...)` (internal), `contentLemmas(...)` (public convenience — nouns, verbs, adjectives)

`POSFilter` enum (`Types/POSFilter.swift`) maps package-level categories to `NLTag` sets.

### Benchmarking & Calibration

- **`EmbeddingBenchmark`** — runs labeled `TextPair` sets and returns `BenchmarkResult` with `discriminationGap` (key metric), `avgHighSimilarity`, `avgLowSimilarity`, `accuracy`. Includes 10 built-in test sets (`generalPairs`, `technicalPairs`, `scientificPairs`, etc.). Use `runSafe` for `NLEmbeddingService` to skip OOV failures.
- **`ThresholdCalibrator`** — sweeps similarity thresholds over `LabeledPair` examples and returns `CalibrationResult` with `bestF1Threshold`, `bestPrecisionThreshold`, `bestRecallThreshold`. Presets: `.duplicateDetection`, `.semanticSearch`, `.contentDiscovery`.

### SwiftUI Views (`Views/`)

- **`EmbeddingComparisonView`** + **`EmbeddingComparisonViewModel`** — full interactive UI for entering custom text pairs, running NLEmbedding or full GPU comparison, viewing discrimination charts, and generating provider/reranker/threshold recommendations.
- **`EmbeddingCharts.swift`** / **`EmbeddingComparisonPreviews.swift`** — chart sub-views used by the comparison view.

### Document Loading (`Loaders/`)

These types support loading structured text corpora from Markdown manuscripts and academic-program CSV files.

**Document structure types:**
- **`ManuscriptParts`** (`Loaders/ManuscriptParts.swift`) — `Codable, CaseIterable, Sendable` enum: `.Title`, `.Abstract`, `.Introduction`, `.Methods`, `.Results`, `.Discussion`, `.Other`.
- **`SectionRule`** (`Loaders/SectionRule.swift`) — `Sendable` struct pairing a regex `pattern: String` with a `type: ManuscriptParts`. Used inside `DocumentProfile`.
- **`DocumentProfile`** (`Loaders/DocumentProfile.swift`) — `Identifiable, Hashable, Sendable` struct holding `id`, `displayName`, `rules: [SectionRule]`, and `fallbackType: ManuscriptParts`. `classify(_ line:)` strips leading numbering then matches rules in order; unmatched lines return `nil`. Static preset `.scientificPaper` covers standard IMRaD structure.
- **`EmbeddingGranularity`** (`Types/EmbeddingGranularity.swift`) — `Codable, CaseIterable, Sendable` enum: `.section` (one embedding per classified section) or `.paragraph` (one per `NLTokenizer` paragraph unit within each section).

**Loaders (caseless enums used as namespaces):**

- **`ManuscriptLoader`** — loads Markdown files produced by PDF-to-Markdown converters (`marker`, `nougat`, etc.). Lines before the first heading are skipped. Level-1 headings not matched by the profile become `.Title`; all other headings use the profile or fall back to `fallbackType`. Each `TextEmbedding` carries metadata keys `"part"` (`ManuscriptParts.rawValue`), `"granularity"` (`EmbeddingGranularity.rawValue`), and `"text"`. The `Corpus.label` is the first level-1 heading; `Corpus.metadata` includes `"filename"` and `"doi"` (if found in the first 3 000 chars).
  - `load(from:profile:granularity:using:as:) async throws -> Corpus`
  - `loadAll(from:profile:granularity:using:as:) async throws -> [Corpus]` — processes all `.md` files in a directory in ascending filename order.

- **`AcademicProgramLoader`** — loads a CSV with columns `University, Program, Course, Title, Credits, Bulletin`. Embeds `"\(Title) \(Bulletin)"` once per unique course per university (first-occurrence wins); `Credits` maps to `TextEmbedding.scaling` (defaults to `1.0`). Returns one `Corpus` per `(University, Program)` pair in file order; `Corpus.label` = program name, `Corpus.metadata["university"]` = university name.
  - `load(from:using:as:) async throws -> [Corpus]`
  - Includes a bundled sample dataset at `Sources/Linguistics/Data/vcu_stem_programs.csv`.

### Models & Types

- **`Corpus`** (`Models/Corpus.swift`) — `Sendable, Codable, Identifiable, Hashable` struct grouping a `[TextEmbedding]` array under a stable `UUID` id, a `label: String`, and a `metadata: [String: String]` bag. Equality and hashing are identity-based (UUID only). Designed for cross-package JSON serialization or SwiftData wrapping.
- **`TextEmbedding`** (`Models/TextEmbedding.swift`) — `Sendable, Codable, Hashable` struct bundling an `EmbeddingProviderOption` tag with a `Vector`, a `scaling: Double` (default `1.0`), and a `metadata: [String: String]` bag. Used for cross-package provenance (Objectives, Places, etc.). Created via `provider.embed(_:as:)` convenience (sets metadata and scaling to defaults).
- **`EmbeddingProviderOption`** (`Types/EmbeddingProviderOption.swift`) — `Codable, Sendable, Hashable` enum with cases for all providers: `.fdlEmbedding`, `.nlEmbedding`, `.miniLM`, `.bgeBase`, `.bgeLarge`, `.mxbaiEmbedLarge`, `.qwen3Embedding`, `.nomicTextV1_5`, `.custom(String)`. Has `displayName`, `abbreviation`, `color` (SwiftUI), and `requiresDownload` properties. `makeProvider(corpus:)` factory creates the corresponding `EmbeddingProvider`; `corpus:` is only required for `.fdlEmbedding`.

### Algorithms

`JaccardDistance.swift` — internal `jaccardSimilarity(a:b:) -> CGFloat` (not public; used by qwotr).

## Key Conventions

- Swift 6 strict concurrency throughout: `EmbeddingProvider` is `Sendable`; `MLXEmbeddingService` and `MLXCrossEncoderReranker` are `actor`s to isolate Metal/MLX state.
- Tests use Swift Testing (`@Test`, `#expect`, `#require`), not XCTest.
- `RankedResult<T>` is generic and `Sendable`; `T` must be `Sendable`.
- Model files are never bundled — always downloaded to HuggingFace cache on first use.
- `EmbeddingComparisonViewModel` is `@Observable` + `@MainActor`; do not add `@Published` properties.
