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



/// Linguistic extensions
public extension String {
    
    /// The overall sentiment score
    var sentimentScore: Double {
        let scores = self.sentenceLevelSentiment
        if scores.isEmpty { return 0 }
        else { return scores.reduce(0.0,+) / Double( scores.count ) }
    }
    
    /// Translate sentiment into emojis
    var sentimentString: String {
        let sentimentEmojis = [
            "🤬", "😡", "😠", "😤", "😫", "😣", "😔", "🙁", "😕", "😑",
            "😶",
            "😌", "🙂", "😊", "😋", "😄", "😆", "😏", "😎", "🤩", "😍"
        ]
        
        let score = self.sentimentScore
        let emojiIndex = Int( score * 10 ) + 10
        return sentimentEmojis[ emojiIndex ]
    }
  
    
    // A paragraph level sentiment score.
    var sentiment: Double {
        let input = self

        // feed it into the NaturalLanguage framework
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = input

        // ask for the results
        let (sentiment, _) = tagger.tag(at: input.startIndex, unit: .paragraph, scheme: .sentimentScore)

        // read the sentiment back and print it
        let score = Double(sentiment?.rawValue ?? "0") ?? 0
        
        return score
        
    }
    
    /// A sentence level sentiment score
    var sentenceLevelSentiment: [Double] {
        var ret = [Double]()
        let input = self
        
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = input

        tokenizer.enumerateTokens(in: input.startIndex ..< input.endIndex ) { tokenRange, _ in
            let sentence = String(input[tokenRange])
            ret.append( sentence.sentiment )
            return true
        }
        return ret
    }
    
    /// A paragraph level sentiment scores
    var paragraphLevelSentiment: [Double] {
        var ret = [Double]()
        let input = self
        
        let tokenizer = NLTokenizer(unit: .paragraph)
        tokenizer.string = input

        tokenizer.enumerateTokens(in: input.startIndex ..< input.endIndex ) { tokenRange, _ in
            let sentence = String(input[tokenRange])
            ret.append( sentence.sentiment )
            return true
        }
        return ret
    }
 
    
    /// Word Count
    var words: Int {
        var ret: Int = 0
        let input = self
        let tokenizer = NLTokenizer( unit: .word)
        tokenizer.string = input
        
        tokenizer.enumerateTokens(in: input.startIndex ..< input.endIndex ) { tokenRange, _ in
            ret += 1
            return true
        }
        return ret
    }
    
    /// Sentence count
    var sentences: Int {
        var ret: Int = 0
        let input = self
        let tokenizer = NLTokenizer( unit: .sentence)
        tokenizer.string = input
        
        tokenizer.enumerateTokens(in: input.startIndex ..< input.endIndex ) { tokenRange, _ in
            ret += 1
            return true
        }
        return ret
    }
    
    /// Paragraph count
    var paragraphs: Int {
        var ret: Int = 0
        let input = self
        let tokenizer = NLTokenizer( unit: .paragraph)
        tokenizer.string = input
        
        tokenizer.enumerateTokens(in: input.startIndex ..< input.endIndex ) { tokenRange, _ in
            ret += 1
            return true
        }
        return ret
    }
    
    /// Automated Readability Index
    ///
    /// A quick index on the legibility of the text
    ///  The algorithm was taken from [wikipedia](https://en.wikipedia.org/wiki/Automated_readability_index)
    var ARI: Double {
        var characters: Double = 0.0
        var words: Double = 0.0
        let sentences = Double( self.sentences )
        let input = self
        
        if sentences == 0 { return 0.0 }
        
        let tokenizer = NLTokenizer( unit: .word )
        tokenizer.string = input
        
        tokenizer.enumerateTokens(in: input.startIndex ..< input.endIndex ) { tokenRange, _ in
            words += 1.0
            characters += Double(String(input[tokenRange]).count)
            return true
        }
        
        return (words == 0 ) ? 0.0 : (4.71 * ( characters/words)) + ( 0.5 * (words/sentences))  - 21.43
    }
    
    
    
    // MARK: - Tokenization + Stopwords

    /// Basic word tokenization using NaturalLanguage.
    /// - Parameters:
    ///   - language: Optional language hint. If nil, `.undetermined` is used.
    ///   - minTokenLength: Minimum number of characters to keep.
    ///   - lowercased: Whether to lowercase tokens.
    /// - Returns: Array of word tokens (no stopword removal).
    nonisolated func wordTokens(language: NLLanguage? = nil,
                                minTokenLength: Int = 2,
                                lowercased: Bool = true) -> [String] {
        let input = self
        guard !input.isEmpty else { return [] }

        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = input
        if let language {
            tokenizer.setLanguage(language)
        }

        var tokens: [String] = []
        tokens.reserveCapacity(max(8, input.count / 6))

        tokenizer.enumerateTokens(in: input.startIndex ..< input.endIndex) { tokenRange, _ in
            var tok = String(input[tokenRange])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if lowercased { tok = tok.lowercased() }
            if tok.count < minTokenLength { return true }

            // Keep only alphanumerics inside the token (e.g., strip punctuation)
            tok = tok.filter { $0.isLetter || $0.isNumber || $0 == "'" }
            if tok.count < minTokenLength { return true }

            tokens.append(tok)
            return true
        }

        return tokens
    }

    /// Returns a set of common English stopwords. Keep this small-ish and editable.
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

    /// Remove stopwords from a token list.
    /// - Parameters:
    ///   - stopwords: Stopword set to remove.
    ///   - keepNumerics: Whether to keep purely numeric tokens.
    /// - Returns: Filtered token list.
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

    /// Convenience: tokenize words and optionally remove stopwords.
    nonisolated func tokensWithoutStopwords(language: NLLanguage? = nil,
                                            stopwords: Set<String> = Self.englishStopwords,
                                            minTokenLength: Int = 2,
                                            lowercased: Bool = true) -> [String] {
        let toks = wordTokens(language: language, minTokenLength: minTokenLength, lowercased: lowercased)
        return removingStopwords(from: toks, stopwords: stopwords)
    }


    // MARK: - Lemmatization + POS Filtering

    /// Tokenize + optionally lemmatize and POS-filter.
    ///
    /// This is designed as a self-contained linguistic preprocessing step:
    /// - Tokenization: word unit
    /// - Lemmatization: `.lemma` via `NLTagger` (fallback to surface form)
    /// - POS filtering: `.lexicalClass`
    /// - Optional stopword removal
    ///
    /// - Parameters:
    ///   - language: Optional language hint; if nil, NaturalLanguage attempts to infer.
    ///   - minTokenLength: Minimum number of characters after cleanup to keep.
    ///   - lowercased: Whether to lowercase output tokens.
    ///   - lemmatize: Whether to return lemmas (recommended for similarity/search pipelines).
    ///   - allowedPOS: If non-nil, only tokens with lexical classes in this set are kept.
    ///                Example: `[.noun, .verb, .adjective]`.
    ///   - removeStopwords: Whether to remove stopwords (applied after lemma/surface selection).
    ///   - stopwords: Stopword set to remove when `removeStopwords` is true.
    ///   - keepNumerics: Whether to keep purely numeric tokens.
    /// - Returns: Normalized tokens.
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

        // We use a tagger so we can get lemma + lexical class aligned.
        let tagger = NLTagger(tagSchemes: [.lemma, .lexicalClass])
        tagger.string = input
        if let language {
            tagger.setLanguage(language, range: input.startIndex..<input.endIndex)
        }

        // Precompute allowed NLTags for POS filtering.
        let allowedNLTags: Set<NLTag>?
        if let filters = allowedPOS {
            var tags = Set<NLTag>()
            for filter in filters {
                tags.formUnion(filter.nlTags)
            }
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
            // POS filter, if requested.
            if let allowedNLTags {
                guard let tag, allowedNLTags.contains(tag) else { return true }
            }

            // Choose lemma or surface form.
            let surface = String(input[tokenRange])
            let lemmaResult: (NLTag?, Range<String.Index>) = tagger.tag(at: tokenRange.lowerBound, unit: .word, scheme: .lemma)
            let lemmaTag = lemmaResult.0
            var tok = (lemmatize ? (lemmaTag?.rawValue ?? surface) : surface)
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if lowercased { tok = tok.lowercased() }

            // Keep only letters/numbers/apostrophes; drop other punctuation.
            tok = tok.filter { $0.isLetter || $0.isNumber || $0 == "'" }

            if tok.count < minTokenLength { return true }

            if !keepNumerics && tok.allSatisfy({ $0.isNumber }) { return true }

            if removeStopwords, stopwords.contains(tok) { return true }

            out.append(tok)
            return true
        }

        return out
    }

    /// Convenience: lemmatized tokens filtered to content words (nouns/verbs/adjectives).
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

