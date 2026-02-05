//
//  Reranker.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/5/26.
//

import Foundation

// MARK: - RankedResult

/// A scored item from reranking.
///
/// Contains the original item along with its relevance score and position
/// before reranking. Use this to understand how items moved in the ranking.
///
/// ## Example
///
/// ```swift
/// let results = try await reranker.rerank(query: "fix bug", documents: docs)
/// for result in results {
///     print("Score: \(result.score), was #\(result.originalIndex)")
/// }
/// ```
public struct RankedResult<T: Sendable>: Sendable {

    /// The original item that was ranked.
    public let item: T

    /// Relevance score (higher = more relevant).
    ///
    /// The scale depends on the reranker implementation:
    /// - `EmbeddingReranker`: Cosine similarity (-1 to 1)
    /// - `MLXCrossEncoderReranker`: Sigmoid output (0 to 1)
    public let score: Float

    /// The item's index in the original input array.
    ///
    /// Use this to track how items moved during reranking.
    public let originalIndex: Int

    /// Creates a new ranked result.
    ///
    /// - Parameters:
    ///   - item: The ranked item
    ///   - score: Relevance score
    ///   - originalIndex: Position in original input
    public init(item: T, score: Float, originalIndex: Int) {
        self.item = item
        self.score = score
        self.originalIndex = originalIndex
    }
}

// MARK: - Reranker Protocol

/// Protocol for cross-encoder reranking models.
///
/// Rerankers process query-document pairs together (cross-attention) for more
/// accurate relevance scoring than bi-encoder embeddings. Use as a second stage
/// after initial embedding-based retrieval.
///
/// ## Overview
///
/// The typical retrieval pipeline uses reranking to improve precision:
///
/// ```
/// Query → Embedding Search (fast, top-100) → Reranker (accurate, top-10) → Results
/// ```
///
/// ## Usage
///
/// ```swift
/// // Stage 1: Fast embedding retrieval
/// let candidates = try await embeddingSearch(query: "how to fix null pointer", topK: 100)
///
/// // Stage 2: Accurate reranking
/// let reranked = try await reranker.rerank(
///     query: "how to fix null pointer",
///     documents: candidates,
///     topK: 10
/// )
///
/// // Use top results
/// for result in reranked {
///     print("[\(result.score)] \(result.item)")
/// }
/// ```
///
/// ## Implementations
///
/// - ``EmbeddingReranker``: Uses embedding similarity (lightweight)
/// - ``MLXCrossEncoderReranker``: True cross-encoder (highest accuracy)
///
/// ## Topics
///
/// ### Scoring
/// - ``score(query:document:)``
/// - ``scoreBatch(query:documents:)``
///
/// ### Reranking
/// - ``rerank(query:documents:topK:)``
/// - ``rerank(query:items:topK:textExtractor:)``
public protocol Reranker: Sendable {

    /// Scores the relevance of a single document to a query.
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - document: The document to score
    /// - Returns: Relevance score (higher = more relevant)
    /// - Throws: An error if scoring fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let score = try await reranker.score(
    ///     query: "python list comprehension",
    ///     document: "List comprehensions provide a concise way to create lists..."
    /// )
    /// ```
    func score(query: String, document: String) async throws -> Float

    /// Scores multiple documents against a query.
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - documents: Documents to score
    /// - Returns: Array of scores in same order as input documents
    /// - Throws: An error if scoring fails
    func scoreBatch(query: String, documents: [String]) async throws -> [Float]

    /// Reranks documents by relevance to query.
    ///
    /// Returns documents sorted by relevance score (highest first).
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - documents: Documents to rerank
    ///   - topK: Number of top results to return (nil = all)
    /// - Returns: Documents sorted by relevance
    /// - Throws: An error if reranking fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let results = try await reranker.rerank(
    ///     query: "machine learning tutorial",
    ///     documents: searchResults,
    ///     topK: 5
    /// )
    /// ```
    func rerank(query: String, documents: [String], topK: Int?) async throws -> [RankedResult<String>]

    /// Reranks arbitrary items using a text extractor.
    ///
    /// Use this when your items are custom types (not just strings).
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - items: Items to rerank
    ///   - topK: Number of top results to return
    ///   - textExtractor: Function to extract searchable text from each item
    /// - Returns: Items sorted by relevance
    /// - Throws: An error if reranking fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// struct Article { let title: String; let body: String }
    ///
    /// let results = try await reranker.rerank(
    ///     query: "climate change",
    ///     items: articles,
    ///     topK: 10,
    ///     textExtractor: { "\($0.title) \($0.body)" }
    /// )
    /// ```
    func rerank<T: Sendable>(
        query: String,
        items: [T],
        topK: Int?,
        textExtractor: @Sendable (T) -> String
    ) async throws -> [RankedResult<T>]
}

// MARK: - Default Implementations

public extension Reranker {

    /// Default batch scoring that processes documents sequentially.
    func scoreBatch(query: String, documents: [String]) async throws -> [Float] {
        var scores: [Float] = []
        for document in documents {
            let score = try await score(query: query, document: document)
            scores.append(score)
        }
        return scores
    }

    /// Default reranking implementation.
    func rerank(query: String, documents: [String], topK: Int? = nil) async throws -> [RankedResult<String>] {
        let scores = try await scoreBatch(query: query, documents: documents)

        var results = zip(documents, scores).enumerated().map { index, pair in
            RankedResult(item: pair.0, score: pair.1, originalIndex: index)
        }

        results.sort { $0.score > $1.score }

        if let topK = topK, topK < results.count {
            return Array(results.prefix(topK))
        }
        return results
    }

    /// Default reranking for custom item types.
    func rerank<T: Sendable>(
        query: String,
        items: [T],
        topK: Int? = nil,
        textExtractor: @Sendable (T) -> String
    ) async throws -> [RankedResult<T>] {
        let documents = items.map(textExtractor)
        let scores = try await scoreBatch(query: query, documents: documents)

        var results = zip(items, scores).enumerated().map { index, pair in
            RankedResult(item: pair.0, score: pair.1, originalIndex: index)
        }

        results.sort { $0.score > $1.score }

        if let topK = topK, topK < results.count {
            return Array(results.prefix(topK))
        }
        return results
    }
}
