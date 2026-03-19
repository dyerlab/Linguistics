//
//  DocumentProfile.swift
//  TrendSpotting
//
//  Created by rodney on 3/5/26.
//

import Foundation


/// Definition of a document based on partitions.
public struct DocumentProfile: Identifiable, Hashable, Sendable {

    // if possible, use DOI
    public let id: String

    // if possible use Title of manucript
    public let displayName: String

    // Rules for this type of document
    public let rules: [SectionRule]

    // Grab-bag unallocated parts
    public let fallbackType: ManuscriptParts

    public init(id: String, displayName: String, rules: [SectionRule], fallbackType: ManuscriptParts) {
        self.id = id
        self.displayName = displayName
        self.rules = rules
        self.fallbackType = fallbackType
    }

    // hashable
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    /// Match a candidate header line against profile rules.
    /// Strips leading "1." / "2.1 " numbering before matching.
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
