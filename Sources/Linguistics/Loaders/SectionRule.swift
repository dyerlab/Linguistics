//
//  File.swift
//  Linguistics
//
//  Created by rodney on 3/10/26.
//

import Foundation

// Data Structure for defining components in a document.

public struct SectionRule: Sendable {
    public let pattern: String
    public let type: ManuscriptParts

    public init(pattern: String, type: ManuscriptParts) {
        self.pattern = pattern
        self.type = type
    }
}
