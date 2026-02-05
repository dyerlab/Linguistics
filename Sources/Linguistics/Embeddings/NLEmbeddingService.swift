//
//  NLEmbeddingService.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/4/26.
//

import Foundation
import NaturalLanguage

// MARK: - NLEmbeddingService

/// Text embedding service using Apple's NaturalLanguage framework.
///
/// `NLEmbeddingService` provides word and sentence embeddings using Apple's pre-trained
/// models. It works offline with no downloads required and is available on all Apple platforms.
///
/// ## Overview
///
/// This service produces 512-dimensional embeddings using Apple's built-in word vectors.
/// For sentence embeddings, it averages word vectors with L2 normalization.
///
/// ## Usage
///
/// ```swift
/// let service = try NLEmbeddingService(language: .english)
///
/// // Get embedding for text
/// let embedding = try await service.embed("Hello, world!")
///
/// // Compare two texts
/// let similarity = try await service.similarity(
///     between: "The cat sat on the mat",
///     and: "A feline rested on the rug"
/// )
/// ```
///
/// ## Word-Level Operations
///
/// Unlike transformer-based models, `NLEmbeddingService` provides direct access to
/// word-level operations:
///
/// ```swift
/// // Check if word is in vocabulary
/// if service.contains("hello") {
///     let wordVector = service.embedWord("hello")
/// }
///
/// // Find similar words
/// let neighbors = service.neighbors(for: "happy", count: 5)
/// // [("glad", 0.12), ("pleased", 0.15), ...]
/// ```
///
/// ## Performance Characteristics
///
/// | Metric | Value |
/// |--------|-------|
/// | Dimensions | 512 |
/// | Startup Time | Instant |
/// | Memory | Low |
/// | Downloads | None |
/// | Offline | Yes |
///
/// ## Topics
///
/// ### Creating a Service
/// - ``init(language:)``
///
/// ### Word Operations
/// - ``embedWord(_:)``
/// - ``contains(_:)``
/// - ``neighbors(for:count:)``
/// - ``distance(from:to:)``
///
/// ### Protocol Conformance
/// - ``EmbeddingProvider``
@available(iOS 14, macOS 11, *)
public final class NLEmbeddingService: @unchecked Sendable {

    // MARK: - Properties

    /// The underlying NLEmbedding word vectors.
    private let wordEmbedding: NLEmbedding

    /// The language used for embeddings.
    public let language: NLLanguage

    /// The dimensionality of embedding vectors (512 for NLEmbedding).
    private let _dimensions: Int

    // MARK: - Initialization

    /// Creates a new NLEmbedding service for the specified language.
    ///
    /// - Parameter language: The language for embeddings (default: `.english`)
    /// - Throws: ``NLEmbeddingError/modelNotAvailable(_:)`` if the embedding model
    ///   is not available for the specified language
    ///
    /// ## Supported Languages
    ///
    /// NLEmbedding supports many languages. Check availability with:
    /// ```swift
    /// if NLEmbedding.wordEmbedding(for: .spanish) != nil {
    ///     let service = try NLEmbeddingService(language: .spanish)
    /// }
    /// ```
    public init(language: NLLanguage = .english) throws {
        guard let embedding = NLEmbedding.wordEmbedding(for: language) else {
            throw NLEmbeddingError.modelNotAvailable(language)
        }
        self.wordEmbedding = embedding
        self.language = language
        self._dimensions = embedding.dimension
    }

    // MARK: - Word Embeddings

    /// Gets the embedding vector for a single word.
    ///
    /// Returns the raw word vector from Apple's pre-trained embeddings.
    /// Words are lowercased before lookup.
    ///
    /// - Parameter word: The word to embed
    /// - Returns: Float vector of length 512, or `nil` if word not in vocabulary
    ///
    /// ## Example
    ///
    /// ```swift
    /// if let vector = service.embedWord("computer") {
    ///     print("Embedding dimensions: \(vector.count)")  // 512
    /// }
    /// ```
    public func embedWord(_ word: String) -> [Float]? {
        guard let vector = wordEmbedding.vector(for: word.lowercased()) else {
            return nil
        }
        return vector.map { Float($0) }
    }

    /// Checks if a word exists in the embedding vocabulary.
    ///
    /// - Parameter word: The word to check (case-insensitive)
    /// - Returns: `true` if the word has an embedding vector
    ///
    /// ## Example
    ///
    /// ```swift
    /// service.contains("hello")  // true
    /// service.contains("asdfgh") // false
    /// ```
    public func contains(_ word: String) -> Bool {
        wordEmbedding.contains(word.lowercased())
    }

    /// Finds the nearest neighbor words to a given word.
    ///
    /// Uses the embedding space to find semantically similar words.
    ///
    /// - Parameters:
    ///   - word: The query word (case-insensitive)
    ///   - count: Maximum number of neighbors to return (default: 10)
    /// - Returns: Array of (word, distance) tuples, sorted by distance (smaller = more similar)
    ///
    /// ## Example
    ///
    /// ```swift
    /// let neighbors = service.neighbors(for: "happy", count: 5)
    /// for (word, distance) in neighbors {
    ///     print("\(word): \(distance)")
    /// }
    /// // glad: 0.123
    /// // pleased: 0.156
    /// // ...
    /// ```
    public func neighbors(for word: String, count: Int = 10) -> [(String, Double)] {
        wordEmbedding.neighbors(for: word.lowercased(), maximumCount: count)
    }

    /// Calculates the embedding distance between two words.
    ///
    /// Returns the distance in embedding space (smaller = more similar).
    /// This is different from similarity (larger = more similar).
    ///
    /// - Parameters:
    ///   - word1: First word (case-insensitive)
    ///   - word2: Second word (case-insensitive)
    /// - Returns: Distance value, or `nil` if either word is not in vocabulary
    ///
    /// ## Example
    ///
    /// ```swift
    /// if let distance = service.distance(from: "cat", to: "dog") {
    ///     print("Distance: \(distance)")  // Small value (similar concepts)
    /// }
    /// ```
    public func distance(from word1: String, to word2: String) -> Double? {
        let d = wordEmbedding.distance(between: word1.lowercased(), and: word2.lowercased())
        guard d.isFinite else {
            return nil
        }
        return d
    }

    // MARK: - Private Helpers

    /// Embeds text by averaging word vectors.
    private func embedText(_ text: String) -> [Float]? {
        let tokens = tokenize(text)
        guard !tokens.isEmpty else { return nil }

        var vectors: [[Double]] = []

        for token in tokens {
            if let vector = wordEmbedding.vector(for: token) {
                vectors.append(vector)
            }
        }

        guard !vectors.isEmpty else { return nil }

        // Average pooling
        let dims = vectors[0].count
        var averaged = [Double](repeating: 0, count: dims)

        for vector in vectors {
            for i in 0..<dims {
                averaged[i] += vector[i]
            }
        }

        let count = Double(vectors.count)
        averaged = averaged.map { $0 / count }

        // Normalize
        return normalize(averaged).map { Float($0) }
    }

    /// Tokenizes text into words suitable for embedding lookup.
    private func tokenize(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        tokenizer.setLanguage(language)

        var tokens: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let token = String(text[range]).lowercased()
            if token.count >= 2, token.allSatisfy({ $0.isLetter }) {
                tokens.append(token)
            }
            return true
        }
        return tokens
    }

    /// L2 normalizes a vector.
    private func normalize(_ vector: [Double]) -> [Double] {
        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard magnitude > 0 else { return vector }
        return vector.map { $0 / magnitude }
    }
}

// MARK: - EmbeddingProvider Conformance

@available(iOS 14, macOS 11, *)
extension NLEmbeddingService: EmbeddingProvider {

    /// The dimensionality of embedding vectors (512 for NLEmbedding).
    public var dimensions: Int {
        get async throws {
            _dimensions
        }
    }

    /// Generates an embedding vector for the given text.
    ///
    /// For multi-word text, this averages the word vectors and normalizes the result.
    /// Words not in the vocabulary are skipped.
    ///
    /// - Parameter text: The text to embed
    /// - Returns: Normalized float vector of length 512
    /// - Throws: ``NLEmbeddingError/encodingFailed(_:)`` if no valid words found
    public func embed(_ text: String) async throws -> [Float] {
        guard let embedding = embedText(text) else {
            throw NLEmbeddingError.encodingFailed(text)
        }
        return embedding
    }

    /// Generates embedding vectors for multiple texts.
    ///
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Array of embedding vectors
    /// - Throws: ``NLEmbeddingError/encodingFailed(_:)`` if any text has no valid words
    public func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        try texts.map { text in
            guard let embedding = embedText(text) else {
                throw NLEmbeddingError.encodingFailed(text)
            }
            return embedding
        }
    }

    /// Calculates cosine similarity between two texts.
    ///
    /// - Parameters:
    ///   - text1: First text
    ///   - text2: Second text
    /// - Returns: Similarity score between -1 and 1
    /// - Throws: ``NLEmbeddingError/encodingFailed(_:)`` if either text has no valid words
    public func similarity(between text1: String, and text2: String) async throws -> Float {
        let emb1 = try await embed(text1)
        let emb2 = try await embed(text2)
        // Vectors are already normalized, so dot product = cosine similarity
        return zip(emb1, emb2).map(*).reduce(0, +)
    }
}

// MARK: - NLEmbeddingError

/// Errors specific to NLEmbedding-based services.
///
/// These errors indicate issues with model availability or text encoding.
public enum NLEmbeddingError: LocalizedError {

    /// The embedding model is not available for the specified language.
    ///
    /// This typically means the language is not supported or the model
    /// hasn't been downloaded on the device.
    case modelNotAvailable(NLLanguage)

    /// Failed to encode the text because no valid words were found in vocabulary.
    ///
    /// This occurs when the input text contains only out-of-vocabulary words,
    /// numbers, or symbols.
    case encodingFailed(String)

    /// A localized description of the error.
    public var errorDescription: String? {
        switch self {
        case .modelNotAvailable(let language):
            return "Embedding model not available for language: \(language.rawValue)"
        case .encodingFailed(let text):
            let preview = text.prefix(50)
            return "Failed to encode text: '\(preview)...' - no valid words found in vocabulary"
        }
    }
}
