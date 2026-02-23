//
//  EmbeddingProvider.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/4/26.
//

import Foundation
import MatrixStuff

// MARK: - EmbeddingProvider Protocol

/// A protocol defining the interface for text embedding services.
///
/// `EmbeddingProvider` provides a common interface for different embedding backends,
/// enabling consistent usage across Apple's NLEmbedding, MLX transformer models,
/// or custom implementations.
///
/// ## Overview
///
/// Text embeddings convert text into dense vector representations that capture semantic meaning.
/// Similar texts produce similar vectors, enabling:
/// - Semantic search and retrieval
/// - Text clustering and classification
/// - Duplicate detection
/// - Recommendation systems
///
/// ## Conforming to EmbeddingProvider
///
/// To create a custom embedding provider, implement the required methods:
///
/// ```swift
/// struct MyEmbeddingProvider: EmbeddingProvider {
///     var dimensions: Int {
///         get async throws { 768 }
///     }
///
///     func embed(_ text: String) async throws -> Vector {
///         // Your embedding logic here
///     }
/// }
/// ```
///
/// ## Topics
///
/// ### Creating Embeddings
/// - ``embed(_:)``
/// - ``embedBatch(_:)``
///
/// ### Comparing Texts
/// - ``similarity(between:and:)``
///
/// ### Properties
/// - ``dimensions``
public protocol EmbeddingProvider: Sendable {

    /// The dimensionality of embedding vectors produced by this provider.
    ///
    /// Different models produce vectors of different sizes:
    /// - NLEmbedding: 512 dimensions
    /// - MiniLM: 384 dimensions
    /// - BGE-Large: 1024 dimensions
    ///
    /// This property may require loading the model on first access.
    var dimensions: Int { get async throws }

    /// Generates an embedding vector for the given text.
    ///
    /// The returned vector is typically L2-normalized, meaning its magnitude equals 1.
    /// This allows cosine similarity to be computed as a simple dot product.
    ///
    /// - Parameter text: The text to embed (word, sentence, or paragraph)
    /// - Returns: A normalized float vector of length ``dimensions``
    /// - Throws: An error if the text cannot be encoded
    ///
    /// ## Example
    ///
    /// ```swift
    /// let embedding = try await provider.embed("Hello, world!")
    /// print("Vector length: \(embedding.count)")
    /// ```
    func embed(_ text: String) async throws -> Vector

    /// Generates embedding vectors for multiple texts.
    ///
    /// For providers that support batching, this may be more efficient than
    /// calling ``embed(_:)`` repeatedly.
    ///
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Array of embedding vectors in the same order as input
    /// - Throws: An error if any text cannot be encoded
    ///
    /// ## Example
    ///
    /// ```swift
    /// let texts = ["First document", "Second document", "Third document"]
    /// let embeddings = try await provider.embedBatch(texts)
    /// // embeddings.count == 3
    /// ```
    func embedBatch(_ texts: [String]) async throws -> [Vector]

    /// Calculates the cosine similarity between two texts.
    ///
    /// Cosine similarity measures the angle between two vectors, returning a value
    /// between -1 (opposite) and 1 (identical). For normalized vectors, this equals
    /// the dot product.
    ///
    /// - Parameters:
    ///   - text1: First text to compare
    ///   - text2: Second text to compare
    /// - Returns: Similarity score between -1 and 1 (higher means more similar)
    /// - Throws: An error if either text cannot be encoded
    ///
    /// ## Example
    ///
    /// ```swift
    /// let similarity = try await provider.similarity(
    ///     between: "The cat sat on the mat",
    ///     and: "A feline rested on the rug"
    /// )
    /// // similarity ≈ 0.7 (high similarity)
    /// ```
    func similarity(between text1: String, and text2: String) async throws -> Float
}

// MARK: - Default Implementations

public extension EmbeddingProvider {

    /// Default batch implementation that processes texts sequentially.
    ///
    /// Override this method in your implementation if your backend supports
    /// more efficient batch processing.
    func embedBatch(_ texts: [String]) async throws -> [Vector] {
        var results: [Vector] = []
        for text in texts {
            let embedding = try await embed(text)
            results.append(embedding)
        }
        return results
    }

    /// Default similarity implementation using cosine similarity of embeddings.
    ///
    /// This computes embeddings for both texts and returns their dot product
    /// (equivalent to cosine similarity for normalized vectors).
    func similarity(between text1: String, and text2: String) async throws -> Float {
        let emb1 = try await embed(text1)
        let emb2 = try await embed(text2)
        return cosineSimilarity(emb1, emb2)
    }

    /// Computes cosine similarity between two vectors.
    ///
    /// Assumes vectors are already normalized (returns dot product).
    private func cosineSimilarity(_ a: Vector, _ b: Vector) -> Float {
        guard a.count == b.count else { return 0 }
        return Float(zip(a, b).map(*).reduce(0, +))
    }
}
