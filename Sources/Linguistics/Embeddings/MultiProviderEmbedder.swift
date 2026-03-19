//
//  MultiProviderEmbedder.swift
//  Linguistics
//
//  Created by Rodney Dyer on 3/19/26.
//

import Foundation

/// Runs text through multiple ``EmbeddingProvider``s and returns one result per provider.
///
/// `MultiProviderEmbedder` is the recommended way to embed the same text or corpus
/// using several models in a research pipeline. It handles concurrent dispatch,
/// error isolation, and mapping results back to their provider tags.
///
/// ## Typical usage
///
/// ```swift
/// // Build from provider options (downloads MLX models as needed)
/// let embedder = try await MultiProviderEmbedder(options: [.nlEmbedding, .miniLM, .bgeBase])
///
/// // Single text → one TextEmbedding per provider
/// let results = try await embedder.embed("Photosynthesis converts light to chemical energy.")
/// // results[.nlEmbedding]?.vector.count == 512
/// // results[.miniLM]?.vector.count == 384
///
/// // Full corpus → one Corpus per provider, preserving section metadata
/// let corpora = try await embedder.embed(corpus: sourceCorpus)
/// ```
///
/// ## Concurrency
///
/// Individual text embedding calls dispatch all providers as concurrent tasks via
/// `TaskGroup`. Corpus-level embedding calls batch texts per provider sequentially
/// to avoid memory pressure from multiple large models running simultaneously.
public struct MultiProviderEmbedder: Sendable {

    /// The active provider instances, keyed by their ``EmbeddingProviderOption`` tag.
    public let providers: [EmbeddingProviderOption: any EmbeddingProvider]

    // MARK: - Initializers

    /// Creates an embedder from pre-instantiated provider instances.
    ///
    /// - Parameter providers: A mapping from option tag to provider instance.
    public init(providers: [EmbeddingProviderOption: any EmbeddingProvider]) {
        self.providers = providers
    }

    /// Creates an embedder by instantiating each requested option.
    ///
    /// Providers are initialized sequentially — MLX loads Metal shader kernels on
    /// first use and concurrent initialization can cause resource contention.
    ///
    /// - Parameters:
    ///   - options: The providers to activate.
    ///   - corpus: Texts used to build the vocabulary for `.fdlEmbedding`.
    ///             Pass `nil` (or omit) when `.fdlEmbedding` is not included.
    public init(options: [EmbeddingProviderOption], corpus: [String]? = nil) async throws {
        var result: [EmbeddingProviderOption: any EmbeddingProvider] = [:]
        for option in options {
            result[option] = try await option.makeProvider(corpus: corpus)
        }
        self.providers = result
    }

    // MARK: - Single text

    /// Embeds `text` using all active providers concurrently.
    ///
    /// Providers execute as concurrent `Task`s. If any provider throws, the error
    /// propagates and the entire call fails.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: One ``TextEmbedding`` per active provider.
    public func embed(_ text: String) async throws -> [EmbeddingProviderOption: TextEmbedding] {
        try await withThrowingTaskGroup(
            of: (EmbeddingProviderOption, TextEmbedding).self
        ) { group in
            for (option, provider) in providers {
                group.addTask {
                    let embedding = try await provider.embed(text, as: option)
                    return (option, embedding)
                }
            }
            var results: [EmbeddingProviderOption: TextEmbedding] = [:]
            for try await (option, embedding) in group {
                results[option] = embedding
            }
            return results
        }
    }

    /// Embeds each text in `texts` using all active providers.
    ///
    /// Returns one dictionary per input text, in the same order as `texts`.
    ///
    /// - Parameter texts: The texts to embed.
    /// - Returns: Array of per-text result dictionaries, one entry per provider.
    public func embedBatch(_ texts: [String]) async throws -> [[EmbeddingProviderOption: TextEmbedding]] {
        var results: [[EmbeddingProviderOption: TextEmbedding]] = []
        for text in texts {
            results.append(try await embed(text))
        }
        return results
    }

    // MARK: - Corpus

    /// Re-embeds the source texts in `corpus` using all active providers.
    ///
    /// Source text is extracted from `TextEmbedding.metadata["text"]`. Section
    /// metadata (`"part"`, `"granularity"`) and `scaling` values are preserved in
    /// each output ``Corpus``. The source corpus UUID is stored in each output
    /// corpus's `metadata["source_corpus_id"]` for traceability.
    ///
    /// - Parameter corpus: The corpus whose texts should be re-embedded.
    /// - Returns: One ``Corpus`` per active provider (new `UUID`s, same `label`).
    public func embed(corpus: Corpus) async throws -> [EmbeddingProviderOption: Corpus] {
        let texts        = corpus.embeddings.map { $0.metadata["text"] ?? "" }
        let parts        = corpus.embeddings.map { $0.metadata["part"] }
        let granularites = corpus.embeddings.map { $0.metadata["granularity"] }
        let scalings     = corpus.embeddings.map { $0.scaling }

        // Embed all texts through all providers
        let batchResults = try await embedBatch(texts)

        // Reassemble one Corpus per provider
        var output: [EmbeddingProviderOption: Corpus] = [:]
        for (option, _) in providers {
            var newEmbeddings: [TextEmbedding] = []

            for (idx, perText) in batchResults.enumerated() {
                guard let embedding = perText[option] else { continue }

                var meta = embedding.metadata
                if let part = parts[idx]        { meta["part"] = part }
                if let gran = granularites[idx]  { meta["granularity"] = gran }
                if !texts[idx].isEmpty           { meta["text"] = texts[idx] }

                newEmbeddings.append(TextEmbedding(
                    provider: option,
                    vector: embedding.vector,
                    metadata: meta,
                    scaling: scalings[idx]
                ))
            }

            var meta = corpus.metadata
            meta["provider"]         = option.abbreviation
            meta["source_corpus_id"] = corpus.id.uuidString

            output[option] = Corpus(
                label: corpus.label,
                metadata: meta,
                embeddings: newEmbeddings
            )
        }
        return output
    }
}
