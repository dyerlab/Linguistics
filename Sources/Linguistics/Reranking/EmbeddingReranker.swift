//
//  EmbeddingReranker.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/5/26.
//

import Foundation
import MatrixStuff

// MARK: - EmbeddingReranker

/// A reranker that uses embedding similarity for scoring.
///
/// `EmbeddingReranker` provides lightweight reranking using any ``EmbeddingProvider``.
/// While not as powerful as true cross-encoder rerankers, it offers several benefits:
///
/// - Fresh encoding ensures query and documents use identical model state
/// - Can use a different (better) model than initial retrieval
/// - Provides consistent scoring across all candidates
/// - No additional model downloads required
///
/// ## Usage
///
/// ```swift
/// // Use a high-quality model for reranking after fast retrieval
/// let retriever = try await MLXEmbeddingService(model: .miniLM)  // Fast
/// let reranker = EmbeddingReranker(
///     provider: try await MLXEmbeddingService(model: .bgeLarge)  // Accurate
/// )
///
/// // Initial retrieval
/// let candidates = try await search(query: query, using: retriever)
///
/// // Rerank with higher-quality model
/// let reranked = try await reranker.rerank(
///     query: query,
///     documents: candidates,
///     topK: 10
/// )
/// ```
///
/// ## When to Use
///
/// Use `EmbeddingReranker` when:
/// - You want a simple reranking solution without additional models
/// - Initial retrieval uses a fast/small model, reranking uses a better one
/// - Cross-encoder latency is too high for your use case
///
/// For maximum accuracy, consider ``MLXCrossEncoderReranker`` instead.
///
/// ## Topics
///
/// ### Creating a Reranker
/// - ``init(provider:)``
///
/// ### Protocol Conformance
/// - ``Reranker``
public struct EmbeddingReranker: Reranker, Sendable {

    /// The embedding provider used for scoring.
    private let provider: any EmbeddingProvider

    /// Creates a new embedding-based reranker.
    ///
    /// - Parameter provider: The embedding model to use for scoring
    ///
    /// ## Example
    ///
    /// ```swift
    /// let provider = try await MLXEmbeddingService(model: .bgeLarge)
    /// let reranker = EmbeddingReranker(provider: provider)
    /// ```
    public init(provider: any EmbeddingProvider) {
        self.provider = provider
    }

    /// Scores a document's relevance to a query using embedding similarity.
    ///
    /// Returns the cosine similarity between query and document embeddings.
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - document: The document to score
    /// - Returns: Cosine similarity score (-1 to 1)
    /// - Throws: An error if embedding fails
    public func score(query: String, document: String) async throws -> Float {
        try await provider.similarity(between: query, and: document)
    }

    /// Scores multiple documents against a query.
    ///
    /// Optimized to encode the query once and compare against all documents.
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - documents: Documents to score
    /// - Returns: Array of similarity scores
    /// - Throws: An error if embedding fails
    public func scoreBatch(query: String, documents: [String]) async throws -> [Float] {
        // Encode query once, then compare against all documents
        let queryEmbedding = try await provider.embed(query)
        let documentEmbeddings = try await provider.embedBatch(documents)

        return documentEmbeddings.map { docEmb in
            cosineSimilarity(queryEmbedding, docEmb)
        }
    }

    /// Computes cosine similarity between two vectors.
    private func cosineSimilarity(_ a: Vector, _ b: Vector) -> Float {
        guard a.count == b.count else { return 0 }
        // Assuming normalized vectors
        return Float(zip(a, b).map(*).reduce(0, +))
    }
}
