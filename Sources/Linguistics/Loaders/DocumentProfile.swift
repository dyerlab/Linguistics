//
//  DocumentProfile.swift
//  Linguistics
//
//  Created by rodney on 3/5/26.
//

import Foundation

/// A document classification schema that maps section headings to ``ManuscriptParts`` types.
///
/// A `DocumentProfile` holds an ordered list of ``SectionRule`` values. When
/// ``ManuscriptLoader`` encounters a heading, it calls ``classify(_:)`` to walk the
/// rules in order and return the first match. Headings that match no rule are
/// assigned ``fallbackType``.
///
/// The built-in ``scientificPaper`` preset covers standard IMRaD structure and is
/// sufficient for most peer-reviewed journal articles. Custom profiles can be
/// constructed for other document conventions (e.g., grant proposals, technical reports).
///
/// `DocumentProfile` is `Identifiable` and `Hashable` by ``id``, so it can be used
/// directly in SwiftUI `List` and `Picker` views.
///
/// ## Example — custom profile
///
/// ```swift
/// let grantProfile = DocumentProfile(
///     id: "grant-proposal",
///     displayName: "NIH Grant Proposal",
///     rules: [
///         SectionRule(pattern: #"^specific\s+aims$"#,        type: .Introduction),
///         SectionRule(pattern: #"^(background|significance)$"#, type: .Introduction),
///         SectionRule(pattern: #"^(approach|methods?)$"#,    type: .Methods),
///     ],
///     fallbackType: .Other
/// )
/// ```
public struct DocumentProfile: Identifiable, Hashable, Sendable {

    /// A unique, stable identifier for this profile.
    ///
    /// For profiles tied to a specific manuscript use its DOI; for generic schemas
    /// use a short descriptive slug (e.g. `"scientific-paper"`).
    public let id: String

    /// A human-readable name shown in picker and list interfaces.
    ///
    /// For manuscript-specific profiles prefer the paper's title; for generic
    /// schemas use a short descriptive name (e.g. `"Scientific Paper"`).
    public let displayName: String

    /// The ordered list of classification rules applied to each heading.
    ///
    /// Rules are evaluated in declaration order. The first pattern that matches
    /// wins; remaining rules are not tested. Place more specific patterns before
    /// more general ones to avoid shadowing.
    public let rules: [SectionRule]

    /// The section type assigned to headings that match no rule.
    ///
    /// ``DocumentProfile/scientificPaper`` uses `.Other` as its fallback,
    /// which captures conclusions, references, acknowledgements, and any
    /// non-standard headings.
    public let fallbackType: ManuscriptParts

    /// Creates a `DocumentProfile` with the given identity, display name, rules, and fallback.
    ///
    /// - Parameters:
    ///   - id: A stable unique identifier (DOI or slug).
    ///   - displayName: A human-readable label for UI display.
    ///   - rules: Classification rules evaluated in order against each heading.
    ///   - fallbackType: Section type assigned when no rule matches.
    public init(id: String, displayName: String, rules: [SectionRule], fallbackType: ManuscriptParts) {
        self.id = id
        self.displayName = displayName
        self.rules = rules
        self.fallbackType = fallbackType
    }

    /// Identity-based equality — two profiles are equal when their ``id``s match.
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id
    }

    /// Identity-based hashing — consistent with ``==(_:_:)``.
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    /// Classifies a heading line against this profile's rules.
    ///
    /// Before pattern matching, leading outline numbering is stripped
    /// (e.g. `"1. Introduction"` → `"Introduction"`, `"2.1 Methods"` → `"Methods"`).
    /// Rules are then tested in order with case-insensitive regex matching.
    ///
    /// - Parameter line: The raw heading text (without the leading `#` characters).
    /// - Returns: The matched ``ManuscriptParts``, or `nil` if no rule matches.
    ///   Pass `nil` results to ``fallbackType`` or treat as a title heading — see
    ///   ``ManuscriptLoader`` for the exact logic.
    public func classify(_ line: String) -> ManuscriptParts? {
        let stripped = line
            .replacingOccurrences(of: #"^\d+[\.\d]*\s*"#, with: "",
                                  options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
        for rule in rules {
            if stripped.range(of: rule.pattern,
                              options: [.regularExpression, .caseInsensitive]) != nil {
                return rule.type
            }
        }
        return nil
    }
}

extension DocumentProfile {

    /// A profile covering standard IMRaD scientific paper structure.
    ///
    /// Recognises the following headings (case-insensitive, stripping leading
    /// outline numbering):
    ///
    /// | Pattern | Assigned type |
    /// |---------|--------------|
    /// | Abstract | `.Abstract` |
    /// | Introduction, Background | `.Introduction` |
    /// | Methods, Materials and Methods, Methodology, Experimental, Study Design, Data Collection | `.Methods` |
    /// | Results, Findings, Observations | `.Results` |
    /// | Discussion, Interpretation | `.Discussion` |
    /// | Conclusions, Summary, Closing Remarks | `.Other` |
    /// | References, Bibliography, Works Cited, Literature Cited | `.Other` |
    /// | Acknowledgements, Funding | `.Other` |
    /// | Keywords | `.Other` |
    ///
    /// Any heading not matched by the rules is assigned `.Other` via `fallbackType`.
    public static let scientificPaper = DocumentProfile(
        id: "scientific-paper",
        displayName: "Scientific Paper",
        rules: [
            SectionRule(pattern: #"^abstract$"#,                        type: .Abstract),

            SectionRule(pattern: #"^(introduction|background)$"#,       type: .Introduction),
            SectionRule(pattern: #"^(methods?|materials?\s+and\s+methods?|methodology|experimental|study\s+design|data\s+collection)$"#,
                                                                         type: .Methods),
            SectionRule(pattern: #"^(results?|findings?|observations?)$"#, type: .Results),
            SectionRule(pattern: #"^(discussion|interpretation)$"#,     type: .Discussion),

            SectionRule(pattern: #"^(conclusions?|summary|closing\s+remarks?)$"#, type: .Other),
            SectionRule(pattern: #"^(references?|bibliography|works?\s+cited|literature\s+cited)$"#, type: .Other),
            SectionRule(pattern: #"^(acknowledgements?|acknowledgments?|funding)$"#, type: .Other),
            SectionRule(pattern: #"^(keywords?|key\s+words?)$"#,        type: .Other),
        ],
        fallbackType: .Other
    )

    static let allProfiles: [DocumentProfile] = [scientificPaper]
}
