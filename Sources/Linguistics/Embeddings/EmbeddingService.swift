//
//  EmbeddingService.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/4/26.
//

import Foundation
import NaturalLanguage

/// Text embedding service using Apple's NaturalLanguage framework.
///
/// Provides word and sentence embeddings using Apple's pre-trained models.
/// For sentence embeddings, uses averaged word vectors with TF-IDF-style weighting.
@available(iOS 14, macOS 11, *)
public class EmbeddingService {

    private let wordEmbedding: NLEmbedding
    private let language: NLLanguage

    /// The dimensionality of the embedding vectors
    public var dimensions: Int {
        wordEmbedding.dimension
    }

    /// Initialize with a specific language
    /// - Parameter language: The language for embeddings (default: .english)
    /// - Throws: If embedding model is not available for the language
    public init(language: NLLanguage = .english) throws {
        guard let embedding = NLEmbedding.wordEmbedding(for: language) else {
            throw EmbeddingError.modelNotAvailable(language)
        }
        self.wordEmbedding = embedding
        self.language = language
    }

    // MARK: - Word Embeddings

    /// Get embedding vector for a single word
    /// - Parameter word: The word to embed
    /// - Returns: Float vector or nil if word not in vocabulary
    public func embedWord(_ word: String) -> [Float]? {
        guard let vector = wordEmbedding.vector(for: word.lowercased()) else {
            return nil
        }
        return vector.map { Float($0) }
    }

    /// Check if a word is in the embedding vocabulary
    public func contains(_ word: String) -> Bool {
        wordEmbedding.contains(word.lowercased())
    }

    /// Find nearest neighbors to a word
    /// - Parameters:
    ///   - word: The query word
    ///   - count: Number of neighbors to return
    /// - Returns: Array of (word, distance) tuples
    public func neighbors(for word: String, count: Int = 10) -> [(String, Double)] {
        wordEmbedding.neighbors(for: word.lowercased(), maximumCount: count)
    }

    // MARK: - Sentence Embeddings

    /// Get embedding vector for a sentence/text by averaging word vectors
    /// - Parameter text: The text to embed
    /// - Returns: Normalized float vector, or nil if no words found in vocabulary
    public func embed(_ text: String) -> [Float]? {
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

    /// Get embeddings for multiple texts
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Array of embedding vectors (nil entries for texts with no valid words)
    public func embedBatch(_ texts: [String]) -> [[Float]?] {
        texts.map { embed($0) }
    }

    // MARK: - Similarity

    /// Calculate cosine similarity between two texts
    /// - Returns: Similarity score between -1 and 1, or nil if either text has no embedding
    public func similarity(between text1: String, and text2: String) -> Float? {
        guard let emb1 = embed(text1), let emb2 = embed(text2) else {
            return nil
        }
        return cosineSimilarity(emb1, emb2)
    }

    /// Calculate distance between two words using the embedding model
    public func distance(from word1: String, to word2: String) -> Double? {
        let d = wordEmbedding.distance(between: word1.lowercased(), and: word2.lowercased())
        guard d.isFinite else {
            return nil
        }
        return d
    }

    // MARK: - Private Helpers

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

    private func normalize(_ vector: [Double]) -> [Double] {
        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard magnitude > 0 else { return vector }
        return vector.map { $0 / magnitude }
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        // Vectors are already normalized, so dot product = cosine similarity
        zip(a, b).map(*).reduce(0, +)
    }
}

// MARK: - Errors

public enum EmbeddingError: LocalizedError {
    case modelNotAvailable(NLLanguage)

    public var errorDescription: String? {
        switch self {
        case .modelNotAvailable(let language):
            return "Embedding model not available for language: \(language.rawValue)"
        }
    }
}
