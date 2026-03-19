//
//  Corpus.swift
//  Linguistics
//
//  Created by Rodney Dyer on 3/10/26.
//

import Foundation

/// A labeled, immutable collection of ``TextEmbedding`` values from a single source.
///
/// `Corpus` groups embeddings that share a common owner — a research manuscript,
/// an academic program, a document, etc. — identified by a stable UUID and a
/// human-readable label.
///
/// The embeddings themselves carry ``TextEmbedding/metadata`` to describe their
/// origin within the corpus (section name, granularity, course code, etc.).
///
/// ## Example
///
/// ```swift
/// let intro  = TextEmbedding(provider: .nlEmbedding, vector: v1, metadata: ["section": "Introduction"])
/// let methods = TextEmbedding(provider: .nlEmbedding, vector: v2, metadata: ["section": "Methods"])
///
/// let paper = Corpus(label: "Effects of Climate on Species Distribution", embeddings: [intro, methods])
/// ```
///
/// ## Persistence
///
/// `Corpus` is `Codable` for JSON serialization in CLI pipelines or database storage.
/// A SwiftData app can wrap it in a `@Model` class or store it as a JSON attribute.
public struct Corpus: Sendable, Codable, Identifiable {

    /// Stable identity for persistence and cross-system references.
    public let id: UUID

    /// Human-readable name for the corpus (e.g., paper title, program name).
    public let label: String

    /// Caller-defined descriptive metadata.
    ///
    /// Keys and values are unconstrained strings, allowing any labeling scheme:
    ///
    /// ```swift
    /// // Research manuscript section
    /// ["Coauthors": "Me & You", "DOI": "abc-112"]
    ///
    /// // Academic program course
    /// ["university": "VCU", "CIP": "34-308"]
    /// ```
    public let metadata: [String: String]
    
    /// The embeddings owned by this corpus, in insertion order.
    public let embeddings: [TextEmbedding]

    public init(id: UUID = UUID(), label: String, metadata: [String:String] = [:], embeddings: [TextEmbedding]) {
        self.id = id
        self.label = label
        self.embeddings = embeddings
        self.metadata = metadata
    }
}

// MARK: - Hashable & Equatable

extension Corpus: Hashable {

    /// Identity-based hashing — consistent with `Identifiable` and SwiftData conventions.
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    public static func == (lhs: Corpus, rhs: Corpus) -> Bool {
        lhs.id == rhs.id
    }
}
