// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
//                      _                 _       _
//                   __| |_   _  ___ _ __| | __ _| |__
//                  / _` | | | |/ _ \ '__| |/ _` | '_ \
//                 | (_| | |_| |  __/ |  | | (_| | |_) |
//                  \__,_|\__, |\___|_|  |_|\__,_|_.__/
//                        |_ _/
//
//         Making Population Genetic Software That Doesn't Suck
//
//  Copyright (c) 2021-2026 Administravia LLC.  All Rights Reserved.
//
//  Created by Rodney Dyer on 1/16/26.
//

import Foundation
import NaturalLanguage

/// `String` extensions providing NLP utilities: sentiment analysis, readability
/// metrics, tokenization, stopword removal, and lemmatization.
///
/// All methods use Apple's `NaturalLanguage` framework and require no additional
/// model downloads.
public extension String {

    // MARK: - Sentiment

    /// The mean sentence-level sentiment score across the entire string.
    ///
    /// Scores range from `-1.0` (strongly negative) to `+1.0` (strongly positive),
    /// with `0.0` representing neutral text. The value is the arithmetic mean of
    /// all sentence-level scores returned by ``sentenceLevelSentiment``.
    ///
    /// Returns `0.0` for empty or unanalysable strings.
    var sentimentScore: Double {
        let scores = self.sentenceLevelSentiment
        if scores.isEmpty { return 0 }
        else { return scores.reduce(0.0, +) / Double(scores.count) }
    }

    /// The overall sentiment expressed as a single emoji on a 21-point scale.
    ///
    /// Maps ``sentimentScore`` to one of 21 emojis, from strongly negative
    /// (`🤬` at −1.0) through neutral (`😶` at 0.0) to strongly positive
    /// (`😍` at +1.0). Useful for quick visual summaries in UI contexts.
    var sentimentString: String {
        let sentimentEmojis = [
            "🤬", "😡", "😠", "😤", "😫", "😣", "😔", "🙁", "😕", "😑",
            "😶",
            "😌", "🙂", "😊", "😋", "😄", "😆", "😏", "😎", "🤩", "😍"
        ]
        let score = self.sentimentScore
        let emojiIndex = Int(score * 10) + 10
        return sentimentEmojis[emojiIndex]
    }

    /// The paragraph-level sentiment score for the entire string treated as one paragraph.
    ///
    /// Uses `NLTagger` with the `.sentimentScore` scheme at `.paragraph` granularity.
    /// Range: `−1.0` (negative) to `+1.0` (positive). Returns `0.0` when the
    /// framework cannot produce a score.
    ///
    /// For multi-paragraph text prefer ``paragraphLevelSentiment`` or
    /// ``sentimentScore`` (sentence-averaged) for finer granularity.
    var sentiment: Double {
        let input = self
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = input
        let (sentiment, _) = tagger.tag(at: input.startIndex, unit: .paragraph, scheme: .sentimentScore)
        return Double(sentiment?.rawValue ?? "0") ?? 0
    }

    /// An array of sentiment scores, one per sentence in the string.
    ///
    /// Sentences are identified by `NLTokenizer` at `.sentence` granularity.
    /// Each score is computed by calling ``sentiment`` on the sentence substring.
    /// Range per element: `−1.0` to `+1.0`.
    var sentenceLevelSentiment: [Double] {
        var ret = [Double]()
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = self
        tokenizer.enumerateTokens(in: self.startIndex ..< self.endIndex) { tokenRange, _ in
            ret.append(String(self[tokenRange]).sentiment)
            return true
        }
        return ret
    }

    /// An array of sentiment scores, one per paragraph in the string.
    ///
    /// Paragraphs are identified by `NLTokenizer` at `.paragraph` granularity.
    /// Each score is computed by calling ``sentiment`` on the paragraph substring.
    /// Range per element: `−1.0` to `+1.0`.
    var paragraphLevelSentiment: [Double] {
        var ret = [Double]()
        let tokenizer = NLTokenizer(unit: .paragraph)
        tokenizer.string = self
        tokenizer.enumerateTokens(in: self.startIndex ..< self.endIndex) { tokenRange, _ in
            ret.append(String(self[tokenRange]).sentiment)
            return true
        }
        return ret
    }

    // MARK: - Readability

    /// The number of words in the string, as counted by `NLTokenizer` at word granularity.
    var words: Int {
        var ret: Int = 0
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = self
        tokenizer.enumerateTokens(in: self.startIndex ..< self.endIndex) { _, _ in
            ret += 1
            return true
        }
        return ret
    }

    /// The number of sentences in the string, as counted by `NLTokenizer` at sentence granularity.
    var sentences: Int {
        var ret: Int = 0
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = self
        tokenizer.enumerateTokens(in: self.startIndex ..< self.endIndex) { _, _ in
            ret += 1
            return true
        }
        return ret
    }

    /// The number of paragraphs in the string, as counted by `NLTokenizer` at paragraph granularity.
    var paragraphs: Int {
        var ret: Int = 0
        let tokenizer = NLTokenizer(unit: .paragraph)
        tokenizer.string = self
        tokenizer.enumerateTokens(in: self.startIndex ..< self.endIndex) { _, _ in
            ret += 1
            return true
        }
        return ret
    }

    /// The Automated Readability Index (ARI) for the string.
    ///
    /// ARI estimates the US school grade level required to comprehend the text.
    /// Typical values:
    ///
    /// | Score | Reading level |
    /// |-------|--------------|
    /// | 1–6   | Elementary school |
    /// | 7–12  | Middle / high school |
    /// | 13+   | College or above |
    ///
    /// Formula: `4.71 × (characters/words) + 0.5 × (words/sentences) − 21.43`
    ///
    /// Reference: [Automated readability index — Wikipedia](https://en.wikipedia.org/wiki/Automated_readability_index)
    ///
    /// Returns `0.0` for empty strings or strings with no sentences.
    var ARI: Double {
        var characters: Double = 0.0
        var wordCount: Double = 0.0
        let sentenceCount = Double(self.sentences)
        guard sentenceCount > 0 else { return 0.0 }

        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = self
        tokenizer.enumerateTokens(in: self.startIndex ..< self.endIndex) { tokenRange, _ in
            wordCount += 1.0
            characters += Double(String(self[tokenRange]).count)
            return true
        }

        return wordCount == 0 ? 0.0
            : (4.71 * (characters / wordCount)) + (0.5 * (wordCount / sentenceCount)) - 21.43
    }

    // MARK: - Tokenization + Stopwords

    /// Tokenizes the string into words using `NLTokenizer`.
    ///
    /// Tokens are stripped of surrounding punctuation and optionally lowercased.
    /// Only tokens with at least `minTokenLength` characters are returned.
    /// No stopword removal or lemmatization is applied — use
    /// ``tokensWithoutStopwords(language:stopwords:minTokenLength:lowercased:)`` or
    /// ``contentLemmas(language:removeStopwords:stopwords:)`` for those pipelines.
    ///
    /// - Parameters:
    ///   - language: Optional language hint passed to `NLTokenizer`. When `nil`,
    ///     the framework attempts automatic language detection.
    ///   - minTokenLength: Minimum character length (post-cleanup) to retain a token.
    ///     Defaults to `2`.
    ///   - lowercased: When `true`, all tokens are lowercased before being returned.
    ///     Defaults to `true`.
    /// - Returns: An array of word tokens in document order.
    nonisolated func wordTokens(language: NLLanguage? = nil,
                                minTokenLength: Int = 2,
                                lowercased: Bool = true) -> [String] {
        let input = self
        guard !input.isEmpty else { return [] }

        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = input
        if let language { tokenizer.setLanguage(language) }

        var tokens: [String] = []
        tokens.reserveCapacity(max(8, input.count / 6))

        tokenizer.enumerateTokens(in: input.startIndex ..< input.endIndex) { tokenRange, _ in
            var tok = String(input[tokenRange])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if lowercased { tok = tok.lowercased() }
            if tok.count < minTokenLength { return true }

            // Keep only alphanumerics inside the token (strip embedded punctuation)
            tok = tok.filter { $0.isLetter || $0.isNumber || $0 == "'" }
            if tok.count < minTokenLength { return true }

            tokens.append(tok)
            return true
        }

        return tokens
    }

    /// A curated set of common English stopwords.
    ///
    /// Covers articles, prepositions, conjunctions, pronouns, common auxiliary
    /// verbs, and other high-frequency function words that carry little semantic
    /// content. Pass a custom set to any tokenization method to extend or override
    /// these defaults.
    nonisolated static var englishStopwords: Set<String> {
        [
            "a","an","the","and","or","but","if","then","else","when","while",
            "to","of","in","on","for","with","as","at","by","from","into","onto","over","under",
            "is","are","was","were","be","been","being","am",
            "it","this","that","these","those","i","you","we","they","he","she","them","us","our","your","his","her","their",
            "not","no","yes","do","does","did","doing","done",
            "so","than","too","very","just","only","also","about",
            "can","could","should","would","may","might","must","will",
            "there","here","what","which","who","whom","whose","why","how",
            "up","down","out","off","again","further","once"
        ]
    }

    /// Filters a token list by removing stopwords and optionally pure numeric tokens.
    ///
    /// - Parameters:
    ///   - tokens: The pre-tokenized word list to filter.
    ///   - stopwords: The set of stopwords to remove. Defaults to ``englishStopwords``.
    ///   - keepNumerics: When `false`, tokens consisting entirely of digits are also
    ///     removed. Defaults to `true`.
    /// - Returns: The filtered token list, preserving original order.
    nonisolated func removingStopwords(from tokens: [String],
                                       stopwords: Set<String> = Self.englishStopwords,
                                       keepNumerics: Bool = true) -> [String] {
        guard !tokens.isEmpty else { return [] }
        return tokens.filter { tok in
            if stopwords.contains(tok) { return false }
            if !keepNumerics && tok.allSatisfy({ $0.isNumber }) { return false }
            return true
        }
    }

    /// Tokenizes the string and removes stopwords in a single call.
    ///
    /// Equivalent to calling ``wordTokens(language:minTokenLength:lowercased:)``
    /// followed by ``removingStopwords(from:stopwords:keepNumerics:)``. No
    /// lemmatization is applied — use ``contentLemmas(language:removeStopwords:stopwords:)``
    /// when lemmatized tokens are needed.
    ///
    /// - Parameters:
    ///   - language: Optional language hint passed to `NLTokenizer`.
    ///   - stopwords: Stopword set to remove. Defaults to ``englishStopwords``.
    ///   - minTokenLength: Minimum character length to retain a token. Defaults to `2`.
    ///   - lowercased: Whether to lowercase tokens. Defaults to `true`.
    /// - Returns: Stopword-filtered word tokens in document order.
    nonisolated func tokensWithoutStopwords(language: NLLanguage? = nil,
                                            stopwords: Set<String> = Self.englishStopwords,
                                            minTokenLength: Int = 2,
                                            lowercased: Bool = true) -> [String] {
        let toks = wordTokens(language: language, minTokenLength: minTokenLength, lowercased: lowercased)
        return removingStopwords(from: toks, stopwords: stopwords)
    }

    // MARK: - Lemmatization + POS Filtering

    /// Tokenizes, optionally lemmatizes, and POS-filters the string.
    ///
    /// This is the full linguistic preprocessing pipeline:
    /// 1. **Tokenization** — word-unit via `NLTagger`
    /// 2. **POS filtering** — retain only tokens whose `NLTag` matches `allowedPOS`
    /// 3. **Lemmatization** — replace surface forms with lemmas (e.g. `"running"` → `"run"`)
    /// 4. **Stopword removal** — discard tokens in `stopwords`
    ///
    /// Use ``contentLemmas(language:removeStopwords:stopwords:)`` for the most common
    /// configuration (nouns + verbs + adjectives, lemmatized, stopwords removed).
    ///
    /// - Parameters:
    ///   - language: Optional language hint. When `nil`, `NaturalLanguage` attempts
    ///     automatic detection.
    ///   - minTokenLength: Minimum character length (post-cleanup) to retain. Defaults to `2`.
    ///   - lowercased: Whether to lowercase output tokens. Defaults to `true`.
    ///   - lemmatize: When `true`, replaces surface tokens with their base lemma via
    ///     `NLTagger`. Recommended for similarity and search pipelines. Defaults to `true`.
    ///   - allowedPOS: If non-`nil`, only tokens whose `NLTag` appears in the union of
    ///     ``POSFilter/nlTags`` for each case are retained. Pass `nil` to keep all POS classes.
    ///   - removeStopwords: Whether to remove stopwords after lemma selection. Defaults to `true`.
    ///   - stopwords: The stopword set to apply when `removeStopwords` is `true`.
    ///     Defaults to ``englishStopwords``.
    ///   - keepNumerics: Whether to retain purely numeric tokens. Defaults to `true`.
    /// - Returns: Normalized tokens in document order.
    internal nonisolated func linguisticTokens(language: NLLanguage? = nil,
                                               minTokenLength: Int = 2,
                                               lowercased: Bool = true,
                                               lemmatize: Bool = true,
                                               allowedPOS: [POSFilter]? = nil,
                                               removeStopwords: Bool = true,
                                               stopwords: Set<String> = Self.englishStopwords,
                                               keepNumerics: Bool = true) -> [String] {
        let input = self
        guard !input.isEmpty else { return [] }

        let tagger = NLTagger(tagSchemes: [.lemma, .lexicalClass])
        tagger.string = input
        if let language {
            tagger.setLanguage(language, range: input.startIndex..<input.endIndex)
        }

        let allowedNLTags: Set<NLTag>?
        if let filters = allowedPOS {
            var tags = Set<NLTag>()
            for filter in filters { tags.formUnion(filter.nlTags) }
            allowedNLTags = tags
        } else {
            allowedNLTags = nil
        }

        var out: [String] = []
        out.reserveCapacity(max(8, input.count / 6))

        let options: NLTagger.Options = [.omitWhitespace, .omitPunctuation]

        tagger.enumerateTags(in: input.startIndex..<input.endIndex,
                             unit: .word,
                             scheme: .lexicalClass,
                             options: options) { tag, tokenRange in
            if let allowedNLTags {
                guard let tag, allowedNLTags.contains(tag) else { return true }
            }

            let surface = String(input[tokenRange])
            let lemmaResult: (NLTag?, Range<String.Index>) = tagger.tag(
                at: tokenRange.lowerBound, unit: .word, scheme: .lemma)
            var tok = (lemmatize ? (lemmaResult.0?.rawValue ?? surface) : surface)
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if lowercased { tok = tok.lowercased() }
            tok = tok.filter { $0.isLetter || $0.isNumber || $0 == "'" }

            if tok.count < minTokenLength { return true }
            if !keepNumerics && tok.allSatisfy({ $0.isNumber }) { return true }
            if removeStopwords, stopwords.contains(tok) { return true }

            out.append(tok)
            return true
        }

        return out
    }

    /// Lemmatized content-word tokens, filtered to nouns, verbs, and adjectives.
    ///
    /// A convenience wrapper around ``linguisticTokens(language:minTokenLength:lowercased:lemmatize:allowedPOS:removeStopwords:stopwords:keepNumerics:)``
    /// with settings tuned for semantic similarity and topic-modelling pipelines:
    /// - POS filter: `.noun`, `.verb`, `.adjective`
    /// - Lemmatization: enabled
    /// - Stopword removal: controlled by `removeStopwords`
    /// - Minimum token length: 2
    ///
    /// These tokens are the recommended input for building an ``FDLEmbeddingService``
    /// corpus vocabulary.
    ///
    /// - Parameters:
    ///   - language: Optional language hint.
    ///   - removeStopwords: Whether to remove stopwords. Defaults to `true`.
    ///   - stopwords: Stopword set to apply. Defaults to ``englishStopwords``.
    /// - Returns: Lemmatized content-word tokens in document order.
    nonisolated func contentLemmas(language: NLLanguage? = nil,
                                   removeStopwords: Bool = true,
                                   stopwords: Set<String> = Self.englishStopwords) -> [String] {
        linguisticTokens(language: language,
                         minTokenLength: 2,
                         lowercased: true,
                         lemmatize: true,
                         allowedPOS: [.noun, .verb, .adjective],
                         removeStopwords: removeStopwords,
                         stopwords: stopwords)
    }
}
