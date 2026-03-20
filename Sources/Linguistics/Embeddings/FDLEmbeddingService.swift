//
//  FDLEmbeddingService.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/23/26.
//

import Foundation
import MatrixStuff
import NaturalLanguage


/// Frequency Dependent Linguistic Embedding Service
///
/// A textual embedding based on stemmed word frequencies across a fixed corpus vocabulary.
///
/// The vocabulary is built once from the supplied corpus at initialisation time: all texts
/// are joined, lowercased, stop-word filtered, and lemmatised to produce a sorted token
/// list.  Each subsequent call to ``embed(_:)`` returns a raw frequency-count vector whose
/// length equals the vocabulary size.
///
/// > Important: Unlike ``NLEmbeddingService`` and ``MLXEmbeddingService``, this service
/// > returns **raw frequency-count vectors** — the vectors are *not* L2-normalized.
/// > The default ``EmbeddingProvider/similarity(between:and:)`` therefore computes an
/// > unnormalized dot product rather than cosine similarity, and scores are not bounded
/// > to `[-1, 1]`. Normalize with `vector.normal` from MatrixStuff before comparing
/// > across providers or when cosine similarity is required.
/// > See <doc:fdl-vector-normalization> for details.
///
/// ## Usage
///
/// ```swift
/// let corpus: [String] = ...   // all documents that define the vocabulary
/// let service = FDLEmbeddingService(corpus: corpus)
///
/// // Via EmbeddingProvider protocol
/// let vector = try await service.embed("The quick brown fox")
///
/// // Normalize if cosine similarity is needed
/// let unitVector = vector.normal
///
/// // With provenance tag (TextEmbedding convenience)
/// let embedding = try await service.embed("The quick brown fox", as: .fdlEmbedding)
/// ```
@available(iOS 14, macOS 11, *)
public final class FDLEmbeddingService: @unchecked Sendable {

    // Sorted vocabulary derived from the corpus.
    private let tokens: [String]

    // Reverse-lookup dictionary: token → index in `tokens` (O(1) per lookup).
    private let tokenIndex: [String: Int]

    /// Creates a new service whose vocabulary is derived from `corpus`.
    ///
    /// Each document in `corpus` is lemmatised, stop-word filtered, and lowercased
    /// independently.  A token is retained in the vocabulary only if it appears in at
    /// least `minimumDocumentFrequency` × `corpus.count` documents (document frequency,
    /// not raw term frequency).  This trims the long tail of highly idiosyncratic terms
    /// that would add noise without contributing signal.
    ///
    /// The vocabulary is immutable after initialisation.
    ///
    /// - Parameters:
    ///   - corpus: The documents that define the embedding vocabulary.
    ///   - minimumDocumentFrequency: Fraction of documents that must contain a token
    ///     for it to be included. Defaults to `0.05` (5 %). Must be in `0…1`.
    public init(corpus: [String], minimumDocumentFrequency: Double = 0.05) {
        let threshold = max(1, Int((minimumDocumentFrequency * Double(corpus.count)).rounded(.up)))

        var tokenDocCount: [String: Int] = [:]
        for text in corpus {
            let docTokens = Set(text.linguisticTokens(keepNumerics: false))
            for token in docTokens {
                tokenDocCount[token, default: 0] += 1
            }
        }

        let sorted = tokenDocCount
            .filter { $0.value >= threshold }
            .map { $0.key }
            .sorted()

        self.tokens = sorted
        self.tokenIndex = Dictionary(
            uniqueKeysWithValues: sorted.enumerated().map { ($0.element, $0.offset) }
        )
    }

    // MARK: - Private helpers

    private func embedText(_ text: String) async throws -> Vector {
        guard !tokens.isEmpty else {
            throw NLEmbeddingError.encodingFailed("FDLEmbeddingService has an empty vocabulary.")
        }

        var vec = Vector(repeating: 0.0, count: tokens.count)
        var missedTokens = 0

        for token in text.linguisticTokens(keepNumerics: false) {
            if let index = tokenIndex[token] {
                vec[index] += 1.0
            } else {
                missedTokens += 1
            }
        }

        if missedTokens > 0 {
            print("WARNING: \(missedTokens) token(s) in input were not in the FDLEmbeddingService vocabulary.")
        }

        return vec
    }
}

// MARK: - EmbeddingProvider Conformance

@available(iOS 14, macOS 11, *)
extension FDLEmbeddingService: EmbeddingProvider {

    /// The dimensionality of embedding vectors, equal to the corpus vocabulary size.
    public var dimensions: Int {
        get async throws { tokens.count }
    }

    /// Generates a frequency-count embedding vector for the given text.
    ///
    /// Each element of the returned vector is the occurrence count of the
    /// corresponding vocabulary token in `text`.  Tokens absent from the
    /// vocabulary are silently skipped (and tallied in a stdout warning).
    ///
    /// > Warning: The returned vector is **not** L2-normalized. Its magnitude equals
    /// > `√Σcountᵢ²` and grows with document length. Use `vector.normal` from
    /// > MatrixStuff to obtain a unit vector before computing cosine similarity or
    /// > comparing scores with ``NLEmbeddingService`` or ``MLXEmbeddingService``.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: Raw frequency-count vector of length `dimensions`.
    /// - Throws: ``NLEmbeddingError/encodingFailed(_:)`` if the vocabulary is empty.
    public func embed(_ text: String) async throws -> Vector {
        try await embedText(text)
    }
}
