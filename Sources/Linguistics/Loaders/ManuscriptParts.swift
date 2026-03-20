//
//  ManuscriptParts.swift
//  Linguistics
//
//  Created by rodney on 3/5/26.
//

import Foundation

/// The logical sections of a structured academic manuscript.
///
/// `ManuscriptParts` follows the IMRaD convention used by most peer-reviewed journals
/// and is the classification vocabulary for ``DocumentProfile`` and ``ManuscriptLoader``.
/// Each case maps to the `part` metadata key stored on every ``TextEmbedding`` and
/// written to the `part` column in ``CorpusStore``.
///
/// ## Mapping
///
/// | Case | Typical headings |
/// |------|-----------------|
/// | `.Title` | First level-1 heading not matched by any profile rule |
/// | `.Abstract` | "Abstract" |
/// | `.Introduction` | "Introduction", "Background" |
/// | `.Methods` | "Methods", "Materials and Methods", "Methodology" |
/// | `.Results` | "Results", "Findings", "Observations" |
/// | `.Discussion` | "Discussion", "Interpretation" |
/// | `.Other` | Conclusions, References, Acknowledgements, Keywords, and anything unmatched |
///
/// The classification is performed by ``DocumentProfile/classify(_:)`` using the
/// ``SectionRule`` array defined in the profile. Headings that match no rule receive
/// the profile's ``DocumentProfile/fallbackType``, which is `.Other` for
/// ``DocumentProfile/scientificPaper``.
public enum ManuscriptParts: String, Codable, CaseIterable, Sendable {

    /// The paper's title, derived from the first level-1 Markdown heading.
    case Title = "Title"

    /// The abstract — a brief structured summary appearing before the introduction.
    case Abstract = "Abstract"

    /// The introduction, providing motivation, background, and research questions.
    case Introduction = "Introduction"

    /// The methods section, describing study design, data collection, and analysis.
    case Methods = "Methods"

    /// The results section, reporting findings without interpretation.
    case Results = "Results"

    /// The discussion section, interpreting results in the context of prior work.
    case Discussion = "Discussion"

    /// A catch-all for sections that don't fit the primary IMRaD structure:
    /// conclusions, references, acknowledgements, keywords, and any unclassified headings.
    case Other = "Other"
}
