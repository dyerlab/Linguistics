//
//  EmbeddingProviderOption.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/20/26.
//

/// A portable, `Codable` tag identifying which embedding provider and model produced
/// a given vector.
///
/// `EmbeddingProviderOption` is the provenance label attached to every
/// ``TextEmbedding``. It travels with the vector through JSON serialization,
/// SQLite persistence (via ``CorpusStore``), and cross-package transmission so
/// downstream consumers always know which model space a vector lives in.
///
/// ## Choosing a provider
///
/// | Case | Model | Dimensions | Requires download |
/// |------|-------|-----------|-------------------|
/// | `.fdlEmbedding` | FDL frequency-count | vocab size (variable) | No |
/// | `.nlEmbedding` | Apple NLEmbedding | ~300 (macOS 26) | No |
/// | `.miniLM` | MiniLM sentence transformer | 384 | Yes (~90 MB) |
/// | `.bgeBase` | BGE Base | 768 | Yes (~400 MB) |
/// | `.bgeLarge` | BGE Large | 1 024 | Yes (~1.2 GB) |
/// | `.mxbaiEmbedLarge` | mxbai-embed-large | 1 024 | Yes (~1.2 GB) |
/// | `.qwen3Embedding` | Qwen3 (4-bit quantized) | varies | Yes |
/// | `.nomicTextV1_5` | Nomic Embed Text v1.5 | varies | Yes |
/// | `.custom(String)` | Any HuggingFace Hub model ID | varies | Yes |

import SwiftUI

public enum EmbeddingProviderOption: Sendable, Codable, Hashable {

    /// FDL (Frequency-Dependent Lexical) bag-of-words embedding.
    ///
    /// Vectors are raw token-frequency counts over a fixed vocabulary; they are
    /// **not** L2-normalized. Call `.normal` (MatrixStuff) before computing cosine
    /// similarity. Requires a corpus at construction time via ``makeProvider(corpus:)``.
    case fdlEmbedding

    /// Apple's on-device `NLEmbedding` word-vector model.
    ///
    /// Embeds text by averaging per-word vectors. Instant startup, zero download,
    /// fully offline. Produces ~300 dimensions on macOS 26 (was 512 on earlier OS versions).
    case nlEmbedding

    /// MiniLM sentence transformer (384 dimensions, ~90 MB download).
    ///
    /// A lightweight model that balances quality and inference speed.
    /// Good first choice when GPU memory or download size is constrained.
    case miniLM

    /// BGE Base sentence transformer (768 dimensions, ~400 MB download).
    ///
    /// Mid-range model optimised for retrieval and semantic search tasks.
    case bgeBase

    /// BGE Large sentence transformer (1 024 dimensions, ~1.2 GB download).
    ///
    /// High-capacity retrieval model. Prefer over `.bgeBase` when recall on
    /// long-tail queries matters more than inference speed.
    case bgeLarge

    /// mxbai-embed-large sentence transformer (1 024 dimensions, ~1.2 GB download).
    ///
    /// High-quality general-purpose model. Typically the best choice for nuanced
    /// semantic similarity tasks such as academic manuscript analysis.
    case mxbaiEmbedLarge

    /// Qwen3 embedding model in 4-bit quantization.
    ///
    /// Provides a compact footprint with competitive quality for multilingual text.
    case qwen3Embedding

    /// Nomic Embed Text v1.5 — a Matryoshka representation learning model.
    ///
    /// Supports dimension truncation: the first *d* dimensions of a full vector
    /// still form a meaningful embedding, enabling storage/speed trade-offs.
    case nomicTextV1_5

    /// Any model available on HuggingFace Hub, identified by its repository ID.
    ///
    /// - Parameter id: A HuggingFace Hub model identifier such as
    ///   `"sentence-transformers/all-mpnet-base-v2"`.
    case custom(String)

    // MARK: - Descriptive properties

    /// A human-readable description including model name, dimensions, and download size.
    ///
    /// Intended for display in picker and list interfaces.
    public var displayName: String {
        switch self {
        case .fdlEmbedding:     return "FDL (VariableD, frequency dependent, offline)"
        case .nlEmbedding:      return "Natural Language (512d, offline)"
        case .miniLM:           return "MiniLM (384d, ~90 MB)"
        case .bgeBase:          return "BGE Base (768d, ~400 MB)"
        case .bgeLarge:         return "BGE Large (1024d, ~1.2 GB)"
        case .mxbaiEmbedLarge:  return "mxbai-embed-large (1024d, ~1.2 GB)"
        case .qwen3Embedding:   return "Qwen3 Embedding (4-bit quantized)"
        case .nomicTextV1_5:    return "Nomic Embed Text v1.5"
        case .custom(let id):   return id
        }
    }

    /// A short lowercase label used as the `provider` key in SQLite and chart axes.
    ///
    /// The abbreviation is stable across releases; do not change existing values
    /// or existing SQLite databases will lose provider identity.
    public var abbreviation: String {
        switch self {
        case .fdlEmbedding:         return "fdl"
        case .nlEmbedding:          return "nl"
        case .miniLM:               return "miniLM"
        case .bgeBase:              return "bge"
        case .bgeLarge:             return "bgeLG"
        case .mxbaiEmbedLarge:      return "mxbai"
        case .qwen3Embedding:       return "qwen"
        case .nomicTextV1_5:        return "nomic"
        case .custom(let string):   return "\(string)"
        }
    }

    /// A SwiftUI `Color` used to distinguish this provider in charts and comparison views.
    ///
    /// Colors are chosen to be perceptually distinct across the standard provider set.
    /// Custom providers are rendered in gray.
    public var color: Color {
        switch self {
        case .fdlEmbedding:     return .red
        case .nlEmbedding:      return .orange
        case .miniLM:           return .yellow
        case .bgeBase:          return .green
        case .bgeLarge:         return .mint
        case .mxbaiEmbedLarge:  return .teal
        case .qwen3Embedding:   return .blue
        case .nomicTextV1_5:    return .indigo
        case .custom:           return .gray
        }
    }

    /// Whether this provider downloads model weights from HuggingFace Hub on first use.
    ///
    /// `false` only for `.nlEmbedding` (uses Apple's on-device model) and
    /// `.fdlEmbedding` (built from a corpus at construction time, no external files).
    public var requiresDownload: Bool { self != .nlEmbedding && self != .fdlEmbedding }

    // MARK: - Factory

    /// Creates and returns the corresponding ``EmbeddingProvider``.
    ///
    /// For MLX-backed providers this downloads and caches model weights on first
    /// call (~seconds to minutes depending on model size and network speed).
    /// Subsequent calls reuse the cached weights.
    ///
    /// - Parameter corpus: Required **only** for ``fdlEmbedding``; pass the full set
    ///   of documents whose vocabulary should define the embedding space. Ignored by
    ///   all other cases — you may safely pass `nil` (the default).
    /// - Throws: ``NLEmbeddingError`` if `corpus` is `nil` for `.fdlEmbedding`, or
    ///   any error thrown by the underlying ``MLXEmbeddingService`` initializer.
    public func makeProvider(corpus: [String]? = nil) async throws -> any EmbeddingProvider {
        switch self {
        case .fdlEmbedding:
            guard let corpus else {
                throw NLEmbeddingError.encodingFailed("FDLEmbeddingService requires a corpus — pass corpus: to makeProvider().")
            }
            return FDLEmbeddingService(corpus: corpus)
        case .nlEmbedding:      return try NLEmbeddingService()
        case .miniLM:           return try await MLXEmbeddingService(model: .miniLM)
        case .bgeBase:          return try await MLXEmbeddingService(model: .bgeBase)
        case .bgeLarge:         return try await MLXEmbeddingService(model: .bgeLarge)
        case .mxbaiEmbedLarge:  return try await MLXEmbeddingService(model: .mxbaiEmbedLarge)
        case .qwen3Embedding:   return try await MLXEmbeddingService(model: .qwen3Embedding)
        case .nomicTextV1_5:    return try await MLXEmbeddingService(model: .nomicTextV1_5)
        case .custom(let id):   return try await MLXEmbeddingService(model: .custom(id))
        }
    }
}
