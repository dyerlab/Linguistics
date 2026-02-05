//
//  EmbeddingBenchmark.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/5/26.
//

import Foundation

// MARK: - TextPair

/// A labeled text pair for benchmarking embedding similarity discrimination.
///
/// Use `TextPair` to create test cases for evaluating how well an embedding
/// model distinguishes between semantically similar and unrelated texts.
///
/// ## Overview
///
/// Each pair contains two texts and a label indicating whether they should
/// be considered similar (high similarity score expected) or dissimilar
/// (low similarity score expected).
///
/// ## Example
///
/// ```swift
/// // Create pairs for your domain
/// let pairs = [
///     TextPair("How do I reset my password?",
///              "I forgot my login credentials",
///              shouldBeSimilar: true,
///              label: "Password help"),
///     TextPair("How do I reset my password?",
///              "What's the weather today?",
///              shouldBeSimilar: false,
///              label: "Password vs weather"),
/// ]
///
/// // Run benchmark
/// let result = try await benchmark.run(
///     provider: embeddingService,
///     name: "MyService",
///     pairs: pairs
/// )
/// ```
///
/// ## Topics
///
/// ### Creating Pairs
/// - ``init(_:_:shouldBeSimilar:label:)``
///
/// ### Properties
/// - ``text1``
/// - ``text2``
/// - ``shouldBeSimilar``
/// - ``label``
public struct TextPair: Sendable {

    /// The first text in the pair (e.g., a query).
    public let text1: String

    /// The second text in the pair (e.g., a document).
    public let text2: String

    /// Whether these texts should be considered semantically similar.
    ///
    /// - `true`: Texts are related; expect high similarity score
    /// - `false`: Texts are unrelated; expect low similarity score
    public let shouldBeSimilar: Bool

    /// Optional descriptive label for this pair.
    ///
    /// Labels appear in benchmark reports and help identify
    /// which pairs are performing well or poorly.
    public let label: String?

    /// Creates a new text pair for benchmarking.
    ///
    /// - Parameters:
    ///   - text1: First text (e.g., query)
    ///   - text2: Second text (e.g., document)
    ///   - shouldBeSimilar: Whether texts should match (true) or not (false)
    ///   - label: Optional descriptive label for reporting
    public init(_ text1: String, _ text2: String, shouldBeSimilar: Bool, label: String? = nil) {
        self.text1 = text1
        self.text2 = text2
        self.shouldBeSimilar = shouldBeSimilar
        self.label = label
    }
}

// MARK: - BenchmarkResult

/// Results from running a benchmark on an embedding provider.
///
/// Contains aggregate statistics and individual scores for analyzing
/// how well an embedding model discriminates between similar and
/// dissimilar text pairs.
///
/// ## Overview
///
/// The key metric is ``discriminationGap`` - the difference between
/// average similarity for matching pairs and non-matching pairs.
/// Higher gaps indicate better discrimination ability.
///
/// ## Example
///
/// ```swift
/// let result = try await benchmark.run(
///     provider: service,
///     name: "MiniLM",
///     pairs: testPairs
/// )
///
/// print(result.summary)
/// // Benchmark: MiniLM
/// //   Similar pairs avg:    0.785
/// //   Dissimilar pairs avg: 0.234
/// //   Discrimination gap:   0.551
/// //   Accuracy @ 0.50: 95.0%
///
/// // Inspect individual scores
/// for (pair, similarity) in result.scores {
///     print("\(pair.label ?? "Pair"): \(similarity)")
/// }
/// ```
///
/// ## Interpreting Results
///
/// | Gap | Interpretation |
/// |-----|----------------|
/// | > 0.4 | Excellent - model distinguishes content well |
/// | 0.25-0.4 | Good - suitable for most applications |
/// | 0.15-0.25 | Moderate - consider a reranker |
/// | < 0.15 | Poor - model struggles with this content type |
///
/// ## Topics
///
/// ### Metrics
/// - ``discriminationGap``
/// - ``avgHighSimilarity``
/// - ``avgLowSimilarity``
/// - ``accuracy``
///
/// ### Detailed Data
/// - ``scores``
/// - ``summary``
public struct BenchmarkResult: Sendable {

    /// Name or description of the provider tested.
    public let providerName: String

    /// Individual similarity scores for each text pair.
    ///
    /// Use this to identify specific pairs where the model
    /// performs well or poorly.
    public let scores: [(pair: TextPair, similarity: Float)]

    /// Average similarity score for pairs that should be similar.
    ///
    /// Higher values indicate the model correctly identifies
    /// semantic similarity. Typical good values are 0.6-0.9.
    public let avgHighSimilarity: Float

    /// Average similarity score for pairs that should be dissimilar.
    ///
    /// Lower values indicate the model correctly identifies
    /// unrelated content. Typical good values are 0.1-0.4.
    public let avgLowSimilarity: Float

    /// The discrimination gap: ``avgHighSimilarity`` - ``avgLowSimilarity``.
    ///
    /// This is the key metric for evaluating embedding quality.
    /// Higher gaps mean better discrimination between similar
    /// and dissimilar content.
    ///
    /// - > 0.4: Excellent discrimination
    /// - 0.25-0.4: Good discrimination
    /// - 0.15-0.25: Moderate (consider reranking)
    /// - < 0.15: Poor (may need different model)
    public let discriminationGap: Float

    /// Classification accuracy at the specified threshold.
    ///
    /// Percentage of pairs correctly classified where:
    /// - Similar pairs have similarity >= threshold
    /// - Dissimilar pairs have similarity < threshold
    public let accuracy: Float

    /// The similarity threshold used for accuracy calculation.
    public let threshold: Float

    /// Formatted summary string for display.
    ///
    /// Includes all key metrics in a human-readable format.
    public var summary: String {
        var lines = [String]()
        lines.append("Benchmark: \(providerName)")
        lines.append("  Similar pairs avg:    \(String(format: "%.3f", avgHighSimilarity))")
        lines.append("  Dissimilar pairs avg: \(String(format: "%.3f", avgLowSimilarity))")
        lines.append("  Discrimination gap:   \(String(format: "%.3f", discriminationGap))")
        lines.append("  Accuracy @ \(String(format: "%.2f", threshold)): \(String(format: "%.1f%%", accuracy * 100))")
        return lines.joined(separator: "\n")
    }
}

// MARK: - EmbeddingBenchmark

/// Utility for benchmarking embedding providers on text similarity tasks.
///
/// `EmbeddingBenchmark` helps you evaluate which embedding model works best
/// for your specific content type by measuring discrimination between
/// semantically similar and dissimilar text pairs.
///
/// ## Overview
///
/// Different embedding models excel at different content types:
/// - **NLEmbedding**: Good for single words and short phrases
/// - **MiniLM**: Excellent for sentence similarity and paraphrases
/// - **BGE/mxbai**: Best for technical content and retrieval
///
/// Use this benchmark to find the optimal model for your domain.
///
/// ## Quick Start
///
/// ```swift
/// let benchmark = EmbeddingBenchmark()
///
/// // Test with built-in pairs
/// let result = try await benchmark.run(
///     provider: myEmbeddingService,
///     name: "My Service",
///     pairs: EmbeddingBenchmark.generalPairs
/// )
/// print(result.summary)
/// ```
///
/// ## Custom Domain Testing
///
/// For best results, create pairs from your actual content:
///
/// ```swift
/// let customPairs = [
///     TextPair("DNA sequence alignment", "genome mapping tools",
///              shouldBeSimilar: true, label: "Genetics"),
///     TextPair("DNA sequence alignment", "chocolate cake recipe",
///              shouldBeSimilar: false, label: "Genetics vs cooking"),
/// ]
///
/// let result = try await benchmark.run(
///     provider: service,
///     name: "BGE-Large",
///     pairs: customPairs
/// )
/// ```
///
/// ## Comparing Multiple Models
///
/// ```swift
/// let benchmark = EmbeddingBenchmark()
/// let pairs = EmbeddingBenchmark.technicalPairs
///
/// // Test each model
/// let nlResult = try await benchmark.run(
///     provider: nlService,
///     name: "NLEmbedding",
///     pairs: pairs
/// )
///
/// let miniLMResult = try await benchmark.run(
///     provider: miniLMService,
///     name: "MiniLM",
///     pairs: pairs
/// )
///
/// // Compare discrimination gaps
/// print("NLEmbedding gap: \(nlResult.discriminationGap)")
/// print("MiniLM gap: \(miniLMResult.discriminationGap)")
/// ```
///
/// ## Built-in Test Sets
///
/// | Test Set | Best For |
/// |----------|----------|
/// | ``generalPairs`` | General-purpose evaluation |
/// | ``shortPhrasePairs`` | Word-level understanding |
/// | ``technicalPairs`` | Programming/technical content |
/// | ``questionPairs`` | FAQ and search applications |
/// | ``scientificPairs`` | Academic/scientific content |
/// | ``paraphrasePairs`` | Sentence similarity |
/// | ``retrievalPairs`` | Document retrieval |
/// | ``conversationalPairs`` | Informal/chat content |
///
/// ## Topics
///
/// ### Running Benchmarks
/// - ``run(provider:name:pairs:threshold:)``
/// - ``runWithReport(provider:name:pairs:threshold:)``
/// - ``runSafe(provider:name:pairs:threshold:)``
///
/// ### Built-in Test Sets
/// - ``generalPairs``
/// - ``shortPhrasePairs``
/// - ``technicalPairs``
/// - ``questionPairs``
/// - ``scientificPairs``
/// - ``singleWordPairs``
/// - ``longPassagePairs``
/// - ``paraphrasePairs``
/// - ``retrievalPairs``
/// - ``conversationalPairs``
/// - ``allTestSets``
public struct EmbeddingBenchmark: Sendable {

    /// Creates a new embedding benchmark instance.
    public init() {}

    /// Runs a benchmark on an embedding provider with the given text pairs.
    ///
    /// Computes similarity scores for all pairs and calculates aggregate
    /// metrics including discrimination gap and accuracy.
    ///
    /// - Parameters:
    ///   - provider: The embedding provider to benchmark
    ///   - name: Display name for results (appears in reports)
    ///   - pairs: Array of text pairs with expected similarity labels
    ///   - threshold: Similarity threshold for accuracy calculation (default: 0.5)
    /// - Returns: ``BenchmarkResult`` containing all metrics and individual scores
    /// - Throws: An error if any similarity computation fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try await benchmark.run(
    ///     provider: embeddingService,
    ///     name: "MiniLM-384d",
    ///     pairs: EmbeddingBenchmark.technicalPairs,
    ///     threshold: 0.55
    /// )
    /// ```
    public func run(
        provider: any EmbeddingProvider,
        name: String,
        pairs: [TextPair],
        threshold: Float = 0.5
    ) async throws -> BenchmarkResult {
        var scores: [(pair: TextPair, similarity: Float)] = []

        for pair in pairs {
            let similarity = try await provider.similarity(between: pair.text1, and: pair.text2)
            scores.append((pair, similarity))
        }

        let highPairs = scores.filter { $0.pair.shouldBeSimilar }
        let lowPairs = scores.filter { !$0.pair.shouldBeSimilar }

        let avgHigh = highPairs.isEmpty ? 0 : highPairs.map(\.similarity).reduce(0, +) / Float(highPairs.count)
        let avgLow = lowPairs.isEmpty ? 0 : lowPairs.map(\.similarity).reduce(0, +) / Float(lowPairs.count)

        // Calculate accuracy: similar pairs should be >= threshold, dissimilar < threshold
        let correctHigh = highPairs.filter { $0.similarity >= threshold }.count
        let correctLow = lowPairs.filter { $0.similarity < threshold }.count
        let accuracy = Float(correctHigh + correctLow) / Float(scores.count)

        return BenchmarkResult(
            providerName: name,
            scores: scores,
            avgHighSimilarity: avgHigh,
            avgLowSimilarity: avgLow,
            discriminationGap: avgHigh - avgLow,
            accuracy: accuracy,
            threshold: threshold
        )
    }

    /// Runs a benchmark and prints detailed results to console.
    ///
    /// Same as ``run(provider:name:pairs:threshold:)`` but also prints
    /// a formatted report including individual pair scores.
    ///
    /// - Parameters:
    ///   - provider: The embedding provider to benchmark
    ///   - name: Display name for results
    ///   - pairs: Array of text pairs with expected similarity labels
    ///   - threshold: Similarity threshold for accuracy calculation (default: 0.5)
    /// - Returns: ``BenchmarkResult`` containing all metrics
    /// - Throws: An error if any similarity computation fails
    ///
    /// ## Output Format
    ///
    /// ```
    /// Benchmark: MiniLM
    ///   Similar pairs avg:    0.785
    ///   Dissimilar pairs avg: 0.234
    ///   Discrimination gap:   0.551
    ///   Accuracy @ 0.50: 95.0%
    ///
    ///   Individual scores:
    ///     ✓ [HIGH] 0.823: Password help
    ///     ✓ [LOW] 0.187: Password vs weather
    ///     ✗ [HIGH] 0.432: Ambiguous pair
    /// ```
    public func runWithReport(
        provider: any EmbeddingProvider,
        name: String,
        pairs: [TextPair],
        threshold: Float = 0.5
    ) async throws -> BenchmarkResult {
        let result = try await run(provider: provider, name: name, pairs: pairs, threshold: threshold)

        print(result.summary)
        print("\n  Individual scores:")
        for (pair, similarity) in result.scores {
            let expected = pair.shouldBeSimilar ? "HIGH" : "LOW"
            let match = (pair.shouldBeSimilar == (similarity >= threshold)) ? "✓" : "✗"
            let label = pair.label ?? "\(pair.text1.prefix(20))..."
            print("    \(match) [\(expected)] \(String(format: "%.3f", similarity)): \(label)")
        }

        return result
    }

    /// Runs a benchmark, skipping pairs that fail encoding.
    ///
    /// Use this for providers with limited vocabulary (like ``NLEmbeddingService``)
    /// that may fail on certain words or phrases. Failed pairs are silently
    /// skipped rather than throwing an error.
    ///
    /// - Parameters:
    ///   - provider: The embedding provider to benchmark
    ///   - name: Display name for results
    ///   - pairs: Array of text pairs with expected similarity labels
    ///   - threshold: Similarity threshold for accuracy calculation (default: 0.5)
    /// - Returns: ``BenchmarkResult`` (may have fewer pairs than input if some failed)
    ///
    /// ## Example
    ///
    /// ```swift
    /// // NLEmbedding may not have all words in vocabulary
    /// let result = await benchmark.runSafe(
    ///     provider: nlEmbeddingService,
    ///     name: "NLEmbedding",
    ///     pairs: EmbeddingBenchmark.conversationalPairs
    /// )
    ///
    /// // Check if pairs were skipped
    /// if result.providerName.contains("skipped") {
    ///     print("Some pairs were skipped due to vocabulary limitations")
    /// }
    /// ```
    public func runSafe(
        provider: any EmbeddingProvider,
        name: String,
        pairs: [TextPair],
        threshold: Float = 0.5
    ) async -> BenchmarkResult {
        var scores: [(pair: TextPair, similarity: Float)] = []
        var skipped = 0

        for pair in pairs {
            do {
                let similarity = try await provider.similarity(between: pair.text1, and: pair.text2)
                scores.append((pair, similarity))
            } catch {
                skipped += 1
            }
        }

        let highPairs = scores.filter { $0.pair.shouldBeSimilar }
        let lowPairs = scores.filter { !$0.pair.shouldBeSimilar }

        let avgHigh = highPairs.isEmpty ? 0 : highPairs.map(\.similarity).reduce(0, +) / Float(highPairs.count)
        let avgLow = lowPairs.isEmpty ? 0 : lowPairs.map(\.similarity).reduce(0, +) / Float(lowPairs.count)

        let correctHigh = highPairs.filter { $0.similarity >= threshold }.count
        let correctLow = lowPairs.filter { $0.similarity < threshold }.count
        let accuracy = scores.isEmpty ? 0 : Float(correctHigh + correctLow) / Float(scores.count)

        return BenchmarkResult(
            providerName: skipped > 0 ? "\(name) (\(skipped) skipped)" : name,
            scores: scores,
            avgHighSimilarity: avgHigh,
            avgLowSimilarity: avgLow,
            discriminationGap: avgHigh - avgLow,
            accuracy: accuracy,
            threshold: threshold
        )
    }
}

// MARK: - Built-in Test Sets

extension EmbeddingBenchmark {

    /// General-purpose test pairs covering common scenarios.
    ///
    /// A balanced set of pairs testing everyday topics like weather,
    /// technology, pets, and shopping. Good for initial model evaluation.
    public static let generalPairs: [TextPair] = [
        // Similar pairs
        TextPair("A dog is running in the park", "A puppy plays outside in the grass",
                 shouldBeSimilar: true, label: "dogs/outdoor activity"),
        TextPair("How do I reset my password?", "I forgot my login credentials",
                 shouldBeSimilar: true, label: "account access"),
        TextPair("The weather today is sunny and warm", "It's a beautiful clear day outside",
                 shouldBeSimilar: true, label: "weather/pleasant"),
        TextPair("Machine learning models require training data", "Neural networks learn from examples",
                 shouldBeSimilar: true, label: "ML concepts"),

        // Dissimilar pairs
        TextPair("A dog is running in the park", "The stock market crashed yesterday",
                 shouldBeSimilar: false, label: "dogs vs finance"),
        TextPair("How do I reset my password?", "The recipe calls for two cups of flour",
                 shouldBeSimilar: false, label: "tech vs cooking"),
        TextPair("The weather today is sunny", "Quantum entanglement enables teleportation",
                 shouldBeSimilar: false, label: "weather vs physics"),
        TextPair("I need to buy groceries", "The ancient Romans built aqueducts",
                 shouldBeSimilar: false, label: "shopping vs history"),
    ]

    /// Short phrase pairs (1-4 words) testing word-level understanding.
    ///
    /// Tests basic synonym recognition and conceptual similarity.
    /// ``NLEmbeddingService`` (word vectors) may perform well here.
    public static let shortPhrasePairs: [TextPair] = [
        // Similar
        TextPair("happy", "joyful", shouldBeSimilar: true, label: "happy/joyful"),
        TextPair("quick brown fox", "fast red wolf", shouldBeSimilar: true, label: "quick animal"),
        TextPair("hot coffee", "warm tea", shouldBeSimilar: true, label: "warm drinks"),
        TextPair("big house", "large home", shouldBeSimilar: true, label: "big dwelling"),

        // Dissimilar
        TextPair("happy", "refrigerator", shouldBeSimilar: false, label: "emotion vs appliance"),
        TextPair("quick brown fox", "economic policy", shouldBeSimilar: false, label: "animal vs policy"),
        TextPair("hot coffee", "cold logic", shouldBeSimilar: false, label: "drink vs abstract"),
        TextPair("big house", "small idea", shouldBeSimilar: false, label: "physical vs abstract"),
    ]

    /// Technical and programming-related pairs.
    ///
    /// Tests understanding of software development concepts like
    /// error messages, git operations, and database queries.
    /// Transformer models typically excel on technical content.
    public static let technicalPairs: [TextPair] = [
        // Similar
        TextPair("null pointer exception", "accessing nil reference",
                 shouldBeSimilar: true, label: "null errors"),
        TextPair("sort the array in ascending order", "arrange elements from smallest to largest",
                 shouldBeSimilar: true, label: "sorting"),
        TextPair("connect to the database", "establish database connection",
                 shouldBeSimilar: true, label: "db connection"),
        TextPair("git push origin main", "upload commits to remote repository",
                 shouldBeSimilar: true, label: "git push"),

        // Dissimilar
        TextPair("null pointer exception", "beautiful sunset painting",
                 shouldBeSimilar: false, label: "error vs art"),
        TextPair("sort the array", "bake a chocolate cake",
                 shouldBeSimilar: false, label: "code vs cooking"),
        TextPair("connect to database", "pet the fluffy cat",
                 shouldBeSimilar: false, label: "tech vs pets"),
        TextPair("git push origin main", "plant tomatoes in spring",
                 shouldBeSimilar: false, label: "git vs gardening"),
    ]

    /// Question-answering pairs for FAQ and search use cases.
    ///
    /// Tests recognition of paraphrased questions asking the same thing.
    /// Useful for evaluating FAQ matching and customer support applications.
    public static let questionPairs: [TextPair] = [
        // Similar questions (paraphrases)
        TextPair("What is the capital of France?", "Which city is France's capital?",
                 shouldBeSimilar: true, label: "France capital"),
        TextPair("How do I create an account?", "What are the steps to sign up?",
                 shouldBeSimilar: true, label: "account creation"),
        TextPair("When does the store open?", "What are your opening hours?",
                 shouldBeSimilar: true, label: "store hours"),
        TextPair("Can I get a refund?", "What is your return policy?",
                 shouldBeSimilar: true, label: "refunds"),

        // Dissimilar questions
        TextPair("What is the capital of France?", "How do I bake bread?",
                 shouldBeSimilar: false, label: "geography vs cooking"),
        TextPair("How do I create an account?", "What causes earthquakes?",
                 shouldBeSimilar: false, label: "tech vs geology"),
        TextPair("When does the store open?", "Who invented the telephone?",
                 shouldBeSimilar: false, label: "hours vs history"),
        TextPair("Can I get a refund?", "How far is the moon?",
                 shouldBeSimilar: false, label: "policy vs astronomy"),
    ]

    /// Scientific and academic pairs.
    ///
    /// Tests understanding of biology, neuroscience, and climate science
    /// concepts. Large transformer models (BGE-Large, mxbai) typically
    /// perform best on specialized scientific content.
    public static let scientificPairs: [TextPair] = [
        // Similar
        TextPair("DNA replication occurs during cell division",
                 "Genetic material is copied when cells divide",
                 shouldBeSimilar: true, label: "DNA replication"),
        TextPair("Photosynthesis converts sunlight to energy",
                 "Plants use light to produce glucose",
                 shouldBeSimilar: true, label: "photosynthesis"),
        TextPair("Climate change affects global temperatures",
                 "Global warming impacts weather patterns",
                 shouldBeSimilar: true, label: "climate"),
        TextPair("Neurons transmit electrical signals",
                 "Nerve cells communicate through impulses",
                 shouldBeSimilar: true, label: "neuroscience"),

        // Dissimilar
        TextPair("DNA replication occurs during cell division",
                 "The Renaissance began in Italy",
                 shouldBeSimilar: false, label: "biology vs history"),
        TextPair("Photosynthesis converts sunlight to energy",
                 "Shakespeare wrote many plays",
                 shouldBeSimilar: false, label: "biology vs literature"),
        TextPair("Climate change affects temperatures",
                 "The guitar has six strings",
                 shouldBeSimilar: false, label: "climate vs music"),
        TextPair("Neurons transmit signals",
                 "Pizza originated in Naples",
                 shouldBeSimilar: false, label: "neuro vs food"),
    ]

    /// Single word pairs testing vocabulary-based similarity.
    ///
    /// Tests synonym recognition and antonym differentiation.
    /// ``NLEmbeddingService`` (word vectors) may excel here since
    /// it's specifically trained on word-level relationships.
    public static let singleWordPairs: [TextPair] = [
        // Similar (synonyms/related)
        TextPair("happy", "joyful", shouldBeSimilar: true, label: "happy/joyful"),
        TextPair("fast", "quick", shouldBeSimilar: true, label: "fast/quick"),
        TextPair("big", "large", shouldBeSimilar: true, label: "big/large"),
        TextPair("smart", "intelligent", shouldBeSimilar: true, label: "smart/intelligent"),
        TextPair("angry", "furious", shouldBeSimilar: true, label: "angry/furious"),
        TextPair("cold", "freezing", shouldBeSimilar: true, label: "cold/freezing"),

        // Dissimilar (unrelated)
        TextPair("happy", "table", shouldBeSimilar: false, label: "emotion vs furniture"),
        TextPair("fast", "purple", shouldBeSimilar: false, label: "speed vs color"),
        TextPair("dog", "algorithm", shouldBeSimilar: false, label: "animal vs concept"),
        TextPair("ocean", "keyboard", shouldBeSimilar: false, label: "nature vs tech"),
        TextPair("music", "concrete", shouldBeSimilar: false, label: "art vs material"),
        TextPair("dream", "invoice", shouldBeSimilar: false, label: "abstract vs business"),
    ]

    /// Long passage pairs testing contextual understanding.
    ///
    /// Multi-sentence passages covering machine learning, climate,
    /// nutrition, and history. Transformer models excel at capturing
    /// meaning across long contexts where word vectors struggle.
    public static let longPassagePairs: [TextPair] = [
        // Similar (same topic, different wording)
        TextPair(
            "The process of machine learning involves training algorithms on large datasets to recognize patterns and make predictions. These models improve over time as they are exposed to more data, adjusting their internal parameters to minimize errors.",
            "Artificial intelligence systems learn by processing vast amounts of information, identifying underlying structures in the data. Through iterative optimization, these algorithms refine their predictions and become more accurate with experience.",
            shouldBeSimilar: true, label: "ML explanation"),
        TextPair(
            "Climate change poses significant challenges to agricultural systems worldwide. Rising temperatures, shifting precipitation patterns, and increased frequency of extreme weather events threaten crop yields and food security for billions of people.",
            "Global warming is disrupting farming practices across the planet. Changes in weather patterns, including droughts and floods, along with temperature increases, are making it harder to grow food and maintain stable agricultural production.",
            shouldBeSimilar: true, label: "climate/agriculture"),

        // Dissimilar (completely different topics)
        TextPair(
            "The process of machine learning involves training algorithms on large datasets to recognize patterns and make predictions. These models improve over time as they are exposed to more data.",
            "The ancient Egyptians built the pyramids using massive limestone blocks quarried from nearby sites. Workers transported these stones using sledges and ramps, a process that took decades to complete.",
            shouldBeSimilar: false, label: "ML vs ancient history"),
        TextPair(
            "Proper nutrition is essential for maintaining good health. A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals in appropriate proportions to support bodily functions and energy needs.",
            "The stock market experienced significant volatility last quarter due to rising interest rates and geopolitical tensions. Investors shifted toward defensive sectors as growth stocks underperformed.",
            shouldBeSimilar: false, label: "nutrition vs finance"),
    ]

    /// Semantic paraphrase pairs testing understanding beyond surface words.
    ///
    /// Includes true paraphrases (same meaning, different words) and
    /// lexically similar but semantically different pairs (word ambiguity).
    /// Sentence transformers like MiniLM are optimized for this task.
    public static let paraphrasePairs: [TextPair] = [
        // True paraphrases (same meaning, different words)
        TextPair("The cat sat on the mat", "A feline was resting on the rug",
                 shouldBeSimilar: true, label: "cat on mat"),
        TextPair("She sold her car yesterday", "Her vehicle was sold the previous day",
                 shouldBeSimilar: true, label: "sold car"),
        TextPair("The movie was really boring", "I found the film quite dull",
                 shouldBeSimilar: true, label: "boring movie"),
        TextPair("He runs five miles every morning", "Each day he jogs 5 miles at dawn",
                 shouldBeSimilar: true, label: "morning run"),
        TextPair("The restaurant serves excellent Italian food", "They have great pasta and pizza there",
                 shouldBeSimilar: true, label: "Italian restaurant"),
        TextPair("I can't find my keys anywhere", "My keys seem to have disappeared",
                 shouldBeSimilar: true, label: "lost keys"),

        // Not paraphrases (similar words but different meaning)
        TextPair("The bank was closed", "The river bank was muddy",
                 shouldBeSimilar: false, label: "bank ambiguity"),
        TextPair("I saw her duck", "The duck swam in the pond",
                 shouldBeSimilar: false, label: "duck ambiguity"),
        TextPair("The bar was set high", "The bar served cocktails",
                 shouldBeSimilar: false, label: "bar ambiguity"),
        TextPair("He's a real star", "The star exploded in a supernova",
                 shouldBeSimilar: false, label: "star ambiguity"),
        TextPair("Time flies like an arrow", "Fruit flies like a banana",
                 shouldBeSimilar: false, label: "classic ambiguity"),
        TextPair("The chicken is ready to eat", "The chicken is ready to eat the corn",
                 shouldBeSimilar: false, label: "syntax ambiguity"),
    ]

    /// Query-document retrieval pairs testing asymmetric search.
    ///
    /// Short queries matched against longer document passages.
    /// BGE and other retrieval-focused models are specifically
    /// optimized for this asymmetric query-document matching.
    public static let retrievalPairs: [TextPair] = [
        // Relevant query-document pairs
        TextPair("how to make pasta",
                 "Boil water in a large pot, add salt, then cook pasta for 8-10 minutes until al dente. Drain and serve with your favorite sauce.",
                 shouldBeSimilar: true, label: "pasta recipe"),
        TextPair("symptoms of flu",
                 "Common influenza symptoms include fever, body aches, fatigue, cough, and sore throat. Symptoms typically appear 1-4 days after exposure.",
                 shouldBeSimilar: true, label: "flu symptoms"),
        TextPair("python list comprehension",
                 "List comprehensions provide a concise way to create lists in Python. The syntax is [expression for item in iterable if condition].",
                 shouldBeSimilar: true, label: "python docs"),
        TextPair("best practices for code review",
                 "Effective code reviews focus on logic errors, security issues, and maintainability. Keep reviews small, provide constructive feedback, and automate style checks.",
                 shouldBeSimilar: true, label: "code review"),

        // Irrelevant query-document pairs
        TextPair("how to make pasta",
                 "The history of automotive engineering began in the late 19th century with the invention of the internal combustion engine.",
                 shouldBeSimilar: false, label: "pasta vs cars"),
        TextPair("symptoms of flu",
                 "Abstract expressionism emerged in New York City in the 1940s as artists sought to convey emotion through non-representational forms.",
                 shouldBeSimilar: false, label: "flu vs art"),
        TextPair("python list comprehension",
                 "The migration patterns of monarch butterflies span thousands of miles across North America each year.",
                 shouldBeSimilar: false, label: "python vs butterflies"),
        TextPair("best practices for code review",
                 "Traditional Japanese tea ceremonies involve precise rituals and movements that have been practiced for centuries.",
                 shouldBeSimilar: false, label: "code vs tea"),
    ]

    /// Conversational and informal pairs testing colloquial understanding.
    ///
    /// Chat-style messages and casual language. Uses full words
    /// (not abbreviations like "brb") to ensure compatibility with
    /// word-vector models like ``NLEmbeddingService``.
    public static let conversationalPairs: [TextPair] = [
        // Similar (same intent)
        TextPair("gonna grab some coffee, want anything?", "I'm getting coffee, need something?",
                 shouldBeSimilar: true, label: "coffee offer"),
        TextPair("that's so hilarious I'm laughing", "haha that is so funny",
                 shouldBeSimilar: true, label: "laughing"),
        TextPair("be right back in a minute", "I will return shortly",
                 shouldBeSimilar: true, label: "returning soon"),
        TextPair("I don't know, maybe later", "not sure, perhaps another time",
                 shouldBeSimilar: true, label: "uncertain/later"),
        TextPair("this is really amazing stuff", "this is incredible work",
                 shouldBeSimilar: true, label: "approval"),
        TextPair("honestly that was really tough", "to be frank that was difficult",
                 shouldBeSimilar: true, label: "difficulty admission"),

        // Dissimilar
        TextPair("gonna grab some coffee", "the server crashed again",
                 shouldBeSimilar: false, label: "coffee vs tech"),
        TextPair("that's hilarious I'm laughing", "the quarterly report is due",
                 shouldBeSimilar: false, label: "casual vs formal"),
        TextPair("be right back soon", "the ancient ruins date back millennia",
                 shouldBeSimilar: false, label: "chat vs history"),
        TextPair("this is really amazing", "the fire department responded quickly",
                 shouldBeSimilar: false, label: "praise vs emergency"),
        TextPair("I don't know maybe later", "mitochondria are the powerhouse of the cell",
                 shouldBeSimilar: false, label: "casual vs science"),
        TextPair("honestly that was tough", "the diamond is extremely hard",
                 shouldBeSimilar: false, label: "difficulty vs materials"),
    ]

    /// All test sets with metadata about which models may perform best.
    ///
    /// Use this to run comprehensive benchmarks across all content types:
    ///
    /// ```swift
    /// for (name, pairs, favoredModel) in EmbeddingBenchmark.allTestSets {
    ///     let result = try await benchmark.run(
    ///         provider: service,
    ///         name: "\(name) - MyService",
    ///         pairs: pairs
    ///     )
    ///     print("\(name): gap=\(result.discriminationGap), favored=\(favoredModel)")
    /// }
    /// ```
    public static let allTestSets: [(name: String, pairs: [TextPair], favoredModel: String)] = [
        ("Single Words", singleWordPairs, "NLEmbedding (word vectors)"),
        ("Short Phrases", shortPhrasePairs, "General (all models)"),
        ("Paraphrases", paraphrasePairs, "MiniLM (sentence similarity)"),
        ("General", generalPairs, "General (all models)"),
        ("Technical", technicalPairs, "Transformer models"),
        ("Questions", questionPairs, "Retrieval models (BGE)"),
        ("Scientific", scientificPairs, "Large transformers"),
        ("Long Passages", longPassagePairs, "Large transformers (BGE, mxbai)"),
        ("Retrieval", retrievalPairs, "BGE, mxbai (retrieval-focused)"),
        ("Conversational", conversationalPairs, "Sentence transformers"),
    ]
}
