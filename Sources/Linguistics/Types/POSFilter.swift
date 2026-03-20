//
//  POSFilter.swift
//  Linguistics
//
//  Created by Rodney Dyer on 1/4/26.
//

import Foundation
import NaturalLanguage

/// A part-of-speech category used to filter tokens during linguistic preprocessing.
///
/// `POSFilter` provides a package-level vocabulary that maps to `NLTag` sets from
/// Apple's `NaturalLanguage` framework. Pass a subset of cases to
/// `String.linguisticTokens(allowedPOS:)` or `String.contentLemmas()` to retain
/// only tokens of the desired grammatical class.
///
/// ## Common usage
///
/// ```swift
/// // Keep only nouns, verbs, and adjectives (content words)
/// let tokens = text.linguisticTokens(allowedPOS: [.noun, .verb, .adjective])
///
/// // Convenience wrapper for the above
/// let lemmas = text.contentLemmas()
/// ```
///
/// `POSFilter` is `Hashable` so it can be used in sets and as dictionary keys.
public enum POSFilter: Hashable, Sendable {

    /// Common nouns and proper nouns (`NLTag.noun`).
    case noun

    /// Action words and state verbs (`NLTag.verb`).
    case verb

    /// Descriptive modifiers (`NLTag.adjective`).
    case adjective

    /// Words that modify verbs, adjectives, or other adverbs (`NLTag.adverb`).
    case adverb

    /// Personal, possessive, demonstrative, and interrogative pronouns (`NLTag.pronoun`).
    case pronoun

    /// Articles and other determiners such as "the", "a", "this" (`NLTag.determiner`).
    case determiner

    /// Words expressing spatial or temporal relations (`NLTag.preposition`).
    case preposition

    /// Coordinating and subordinating conjunctions (`NLTag.conjunction`).
    case conjunction

    /// Cardinal and ordinal numbers expressed as words (`NLTag.number`).
    case numeral

    /// Grammatical particles such as infinitive markers (`NLTag.particle`).
    case particle

    /// Exclamations and other non-compositional expressions (`NLTag.interjection`).
    case interjection

    /// Numeral classifiers used in some East Asian languages (`NLTag.classifier`).
    case classifier

    /// Any part of speech not covered by the cases above.
    ///
    /// When used in an `allowedPOS` filter this case matches nothing — its
    /// ``nlTags`` set is empty — so including `.other` has no effect on the
    /// token output.
    case other

    /// The set of `NLTag` values that correspond to this filter case.
    ///
    /// Used internally by `String.linguisticTokens(allowedPOS:)` to build the
    /// set of acceptable `NaturalLanguage` tags before enumerating tokens.
    nonisolated var nlTags: Set<NLTag> {
        switch self {
        case .noun:         return [.noun]
        case .verb:         return [.verb]
        case .adjective:    return [.adjective]
        case .adverb:       return [.adverb]
        case .pronoun:      return [.pronoun]
        case .determiner:   return [.determiner]
        case .preposition:  return [.preposition]
        case .conjunction:  return [.conjunction]
        case .numeral:      return [.number]
        case .particle:     return [.particle]
        case .interjection: return [.interjection]
        case .classifier:   return [.classifier]
        case .other:        return []
        }
    }
}
