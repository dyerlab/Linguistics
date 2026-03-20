//
//  SectionRule.swift
//  Linguistics
//
//  Created by rodney on 3/10/26.
//

import Foundation

/// A rule that maps a heading-text pattern to a ``ManuscriptParts`` section type.
///
/// `SectionRule` pairs a case-insensitive regular expression with the manuscript
/// section it identifies. Rules are evaluated in declaration order by
/// ``DocumentProfile/classify(_:)``, which strips leading outline numbering
/// (e.g. `"1."`, `"2.1 "`) before testing each pattern.
///
/// ## Example
///
/// ```swift
/// let rule = SectionRule(pattern: #"^(introduction|background)$"#, type: .Introduction)
/// ```
///
/// Patterns are matched with `.regularExpression` and `.caseInsensitive` options,
/// so `"INTRODUCTION"`, `"Introduction"`, and `"introduction"` all match the
/// example above.
public struct SectionRule: Sendable {

    /// A regular expression pattern matched against the stripped heading text.
    ///
    /// Patterns are applied with `.caseInsensitive` and `.regularExpression` options.
    /// Anchoring with `^` and `$` is recommended to avoid partial matches (e.g.,
    /// `"^methods?$"` matches `"Method"` and `"Methods"` but not `"Supplementary Methods"`).
    public let pattern: String

    /// The manuscript section type assigned when this rule's pattern matches.
    public let type: ManuscriptParts

    /// Creates a `SectionRule` with the given regex pattern and target section type.
    ///
    /// - Parameters:
    ///   - pattern: A case-insensitive regular expression matched against stripped heading text.
    ///   - type: The ``ManuscriptParts`` value assigned when the pattern matches.
    public init(pattern: String, type: ManuscriptParts) {
        self.pattern = pattern
        self.type = type
    }
}
