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

    /// One embedding for the full section (sequence_index 0) plus one per paragraph
    /// within that section (sequence_index 1…N), all in a single pass.
    ///
    /// This is the preferred granularity for hierarchical analysis — e.g., studying
    /// how semantic variance changes across ordered paragraphs within an Introduction
    /// or Methods section.
    case sectionAndParagraphs
}
