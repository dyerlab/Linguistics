//
//  TextEmbedding.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/23/26.
//

import Foundation
import MatrixStuff

/// A portable embedding value bundling a provider tag with its vector.
///
/// `TextEmbedding` is designed to cross package boundaries (Objectives, Places, etc.)
/// so downstream consumers know which model produced a stored vector without
/// having a direct dependency on the Linguistics internals.
///
/// ## Example
///
/// ```swift
/// let provider = try NLEmbeddingService()
/// let embedding = try await provider.embed("Hello", as: .nlEmbedding)
/// // embedding.provider == .nlEmbedding
/// // embedding.vector.count == 512
/// ```
public struct TextEmbedding: Sendable, Codable, Hashable {

    /// Identifies the model that produced ``vector``.
    public let provider: EmbeddingProviderOption

    /// The L2-normalized embedding vector.
    public let vector: Vector

    public init(provider: EmbeddingProviderOption, vector: Vector) {
        self.provider = provider
        self.vector = vector
    }
}

// MARK: - EmbeddingProvider convenience

public extension EmbeddingProvider {

    /// Embeds text and wraps the result in a ``TextEmbedding`` with the given provider tag.
    ///
    /// - Parameters:
    ///   - text: The text to embed
    ///   - option: The ``EmbeddingProviderOption`` tag to attach for provenance
    /// - Returns: A ``TextEmbedding`` with the normalized vector and provider label
    func embed(_ text: String, as option: EmbeddingProviderOption) async throws -> TextEmbedding {
        TextEmbedding(provider: option, vector: try await embed(text))
    }
}
