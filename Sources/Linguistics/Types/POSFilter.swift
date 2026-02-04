//
//  POSFilter.swift
//  qwotr
//
//  Created by Rodney Dyer on 1/4/26.
//

import Foundation
import NaturalLanguage

/// A lightweight set of POS categories for filtering tokens.
public enum POSFilter: Hashable, Sendable {
    case noun
    case verb
    case adjective
    case adverb
    case pronoun
    case determiner
    case preposition
    case conjunction
    case numeral
    case particle
    case interjection
    case classifier
    case other

    nonisolated var nlTags: Set<NLTag> {
        switch self {
        case .noun: return [.noun]
        case .verb: return [.verb]
        case .adjective: return [.adjective]
        case .adverb: return [.adverb]
        case .pronoun: return [.pronoun]
        case .determiner: return [.determiner]
        case .preposition: return [.preposition]
        case .conjunction: return [.conjunction]
        case .numeral: return [.number]
        case .particle: return [.particle]
        case .interjection: return [.interjection]
        case .classifier: return [.classifier]
        case .other:
            return []
        }
    }
}
