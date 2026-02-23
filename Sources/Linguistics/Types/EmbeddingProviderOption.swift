//
//  EmbeddingProviderOption.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/20/26.
//

/// Identifies which embedding provider and model produced a given vector.
///
/// `EmbeddingProviderOption` is a portable, `Codable` tag used to record
/// provenance when storing or transmitting `TextEmbedding` values across
/// domain packages (Objectives, Places, etc.).
public enum EmbeddingProviderOption: Sendable, Codable, Hashable {
    case nlEmbedding        // NLEmbeddingService — 512d, instant, offline
    case miniLM             // MLXEmbeddingService(.miniLM) — 384d, ~90 MB
    case bgeBase            // MLXEmbeddingService(.bgeBase) — 768d, ~400 MB
    case bgeLarge           // MLXEmbeddingService(.bgeLarge) — 1024d, ~1.2 GB
    case mxbaiEmbedLarge    // MLXEmbeddingService(.mxbaiEmbedLarge) — 1024d, ~1.2 GB
    case qwen3Embedding     // MLXEmbeddingService(.qwen3Embedding) — 4-bit quantized
    case nomicTextV1_5      // MLXEmbeddingService(.nomicTextV1_5) — Matryoshka
    case custom(String)     // Any HuggingFace Hub ID

    public var displayName: String {
        switch self {
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

    public var requiresDownload: Bool { self != .nlEmbedding }

    /// Creates and returns the corresponding ``EmbeddingProvider``.
    public func makeProvider() async throws -> any EmbeddingProvider {
        switch self {
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
