//
//  EmbeddingGranularity.swift
//  Linguistics
//
//  Created by Rodney Dyer on 3/10/26.
//

/// The level at which text is chunked before embedding.
public enum EmbeddingGranularity: String, Codable, CaseIterable, Sendable {

    /// One embedding per classified document section (e.g., the full Introduction).
    case section

    /// One embedding per paragraph within each section.
    case paragraph
}
