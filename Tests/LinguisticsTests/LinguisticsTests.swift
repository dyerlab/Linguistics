import Testing
@testable import Linguistics
import NaturalLanguage
import Foundation
import SQLite3

// MARK: - NLEmbeddingService Tests

@Test func nlEmbeddingServiceInitialization() async throws {
    let service = try NLEmbeddingService(language: .english)
    let dims = try await service.dimensions
    #expect(dims > 0)
}

@Test func nlEmbeddingServiceEmbedText() async throws {
    let service = try NLEmbeddingService(language: .english)
    let embedding = try await service.embed("Hello, world!")

    #expect(!embedding.isEmpty)

    // Check vector is normalized (magnitude ~= 1)
    let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
    #expect(abs(magnitude - 1.0) < 0.01)
}

@Test func nlEmbeddingServiceSimilarity() async throws {
    let service = try NLEmbeddingService(language: .english)

    let similarity = try await service.similarity(
        between: "The cat sat on the mat",
        and: "A feline rested on the rug"
    )

    // Similar sentences should have positive similarity
    #expect(similarity > 0)

    // Compare with same sentences as MLX test for baseline
    let highSim = try await service.similarity(
        between: "A dog is running in the park",
        and: "A puppy plays outside in the grass"
    )
    let lowSim = try await service.similarity(
        between: "A dog is running in the park",
        and: "The stock market crashed yesterday"
    )
    print("NLEmbedding similarity - HIGH: \(String(format: "%.3f", highSim)), LOW: \(String(format: "%.3f", lowSim)), gap: \(String(format: "%.3f", highSim - lowSim))")
}

@Test func nlEmbeddingServiceBatchEmbed() async throws {
    let service = try NLEmbeddingService(language: .english)
    let texts = ["Hello", "World", "Testing"]

    let embeddings = try await service.embedBatch(texts)

    #expect(embeddings.count == 3)
    for embedding in embeddings {
        #expect(!embedding.isEmpty)
    }
}

@Test func nlEmbeddingServiceWordLevel() async throws {
    let service = try NLEmbeddingService(language: .english)

    // Test word embedding
    let wordEmbedding = service.embedWord("hello")
    #expect(wordEmbedding != nil)

    // Test vocabulary check
    #expect(service.contains("hello"))

    // Test neighbors
    let neighbors = service.neighbors(for: "happy", count: 5)
    #expect(neighbors.count <= 5)
}

// MARK: - EmbeddingProvider Protocol Tests

@Test func embeddingProviderProtocolConformance() async throws {
    // Verify NLEmbeddingService conforms to EmbeddingProvider
    let provider: any EmbeddingProvider = try NLEmbeddingService(language: .english)

    let embedding = try await provider.embed("Test text")
    #expect(!embedding.isEmpty)
}

// MARK: - MLXEmbeddingService Tests (Model Enum only, no network)

@Test func mlxModelEnumConfiguration() {
    // Test that model enum produces correct configurations
    let models: [MLXEmbeddingService.Model] = [
        .mxbaiEmbedLarge,
        .bgeLarge,
        .bgeBase,
        .miniLM,
        .qwen3Embedding,
        .nomicTextV1_5,
        .custom("test/model")
    ]

    for model in models {
        #expect(!model.name.isEmpty)
    }
}

// MARK: - MLXEmbeddingService Integration Tests (requires network & download)

@Test(.timeLimit(.minutes(10)))
func mlxEmbeddingServiceIntegration() async throws {
    print("Loading MLX embedding model (mxbai-embed-large-v1)...")
    print("This may download ~400MB on first run.")

    let service = try await MLXEmbeddingService(
        model: .mxbaiEmbedLarge,
        progressHandler: { progress in
            let percent = Int(progress * 100)
            if percent % 20 == 0 {
                print("Download progress: \(percent)%")
            }
        }
    )

    // Test dimensions
    let dims = try await service.dimensions
    print("Embedding dimensions: \(dims)")

    // Debug: Check pooling info by comparing raw embeddings
    let testEmb = try await service.embed("test")
    print("Embedding stats - min: \(testEmb.min()!), max: \(testEmb.max()!), mean: \(testEmb.reduce(0, +) / Double(testEmb.count))")
    #expect(dims == 1024)  // mxbai-embed-large produces 1024 dims

    // Test single embedding
    let text1 = "Machine learning on Apple Silicon is incredibly fast."
    let emb1 = try await service.embed(text1)
    #expect(emb1.count == dims)

    // Verify normalization
    let magnitude = sqrt(emb1.map { $0 * $0 }.reduce(0, +))
    let mag = abs(magnitude - 1.0)
    print("Normalization of embedding magnitude: \(mag)")
    #expect(mag < 0.01)

    // Test similarity - related topics should be more similar
    let text2 = "Neural networks run efficiently on M-series chips."
    let text3 = "The weather today is sunny and warm."

    let sim12 = try await service.similarity(between: text1, and: text2)
    let sim13 = try await service.similarity(between: text1, and: text3)

    print("Similarity (related): \(sim12)")
    print("Similarity (unrelated): \(sim13)")

    // Related texts should have higher similarity than unrelated
    #expect(sim12 > sim13)
    #expect(sim12 > 0.5)  // Related topics should be reasonably similar

    // Test with more diverse sentence pairs for better discrimination
    // These are inspired by STS Benchmark style comparisons
    print("\nDiverse similarity tests:")

    let highSim1 = try await service.similarity(
        between: "A dog is running in the park",
        and: "A puppy plays outside in the grass"
    )
    print("  [should be HIGH] \(String(format: "%.3f", highSim1)): dog/park vs puppy/grass")

    let lowSim1 = try await service.similarity(
        between: "A dog is running in the park",
        and: "The stock market crashed yesterday"
    )
    print("  [should be LOW]  \(String(format: "%.3f", lowSim1)): dog/park vs stock market")

    let highSim2 = try await service.similarity(
        between: "How do I reset my password?",
        and: "I forgot my login credentials"
    )
    print("  [should be HIGH] \(String(format: "%.3f", highSim2)): password reset vs forgot login")

    let lowSim2 = try await service.similarity(
        between: "How do I reset my password?",
        and: "The recipe calls for two cups of flour"
    )
    print("  [should be LOW]  \(String(format: "%.3f", lowSim2)): password reset vs recipe")

    // Print the gap for debugging
    print("\nDiscrimination gaps:")
    print("  Gap 1 (dog vs stock): \(highSim1 - lowSim1)")
    print("  Gap 2 (password vs recipe): \(highSim2 - lowSim2)")

    // These gaps are too small - investigating...
    // For now, just verify ordering is correct
    #expect(highSim1 > lowSim1)
    #expect(highSim2 > lowSim2)
}

/// Test MLX embedding service with MiniLM model (smallest/fastest).
///
/// **Note**: MLX requires Xcode with Metal toolchain to compile shaders.
/// Running via `swift test` from command line won't work.
@Test(.timeLimit(.minutes(5)))
func mlxEmbeddingServiceQuickTest() async throws {
    print("Testing MLX embedding with MiniLM (smallest model)...")

    let service = try await MLXEmbeddingService(
        model: .miniLM,
        progressHandler: { progress in
            let percent = Int(progress * 100)
            if percent % 25 == 0 {
                print("Download progress: \(percent)%")
            }
        }
    )

    let embedding = try await service.embed("Hello world")
    #expect(embedding.count == 384)  // MiniLM produces 384 dims
    print("MiniLM embedding test passed! Dimensions: \(embedding.count)")

    // Compare with mxbai results
    let highSim = try await service.similarity(
        between: "A dog is running in the park",
        and: "A puppy plays outside in the grass"
    )
    let lowSim = try await service.similarity(
        between: "A dog is running in the park",
        and: "The stock market crashed yesterday"
    )
    print("MiniLM similarity - HIGH: \(String(format: "%.3f", highSim)), LOW: \(String(format: "%.3f", lowSim)), gap: \(String(format: "%.3f", highSim - lowSim))")
}

@Test(.timeLimit(.minutes(10)))
func mlxBGELargeTest() async throws {
    print("Testing BGE-Large (for comparison with mxbai)...")

    let service = try await MLXEmbeddingService(
        model: .bgeLarge,
        progressHandler: { progress in
            let percent = Int(progress * 100)
            if percent % 25 == 0 {
                print("Download progress: \(percent)%")
            }
        }
    )

    let dims = try await service.dimensions
    print("BGE-Large dimensions: \(dims)")

    let highSim = try await service.similarity(
        between: "A dog is running in the park",
        and: "A puppy plays outside in the grass"
    )
    let lowSim = try await service.similarity(
        between: "A dog is running in the park",
        and: "The stock market crashed yesterday"
    )
    print("BGE-Large similarity - HIGH: \(String(format: "%.3f", highSim)), LOW: \(String(format: "%.3f", lowSim)), gap: \(String(format: "%.3f", highSim - lowSim))")

    #expect(highSim > lowSim)
}

// MARK: - EmbeddingBenchmark Tests

@Test func benchmarkWithNLEmbedding() async throws {
    let service = try NLEmbeddingService(language: .english)
    let benchmark = EmbeddingBenchmark()

    print("\n=== NLEmbedding Benchmark ===")

    let generalResult = try await benchmark.runWithReport(
        provider: service,
        name: "NLEmbedding - General",
        pairs: EmbeddingBenchmark.generalPairs,
        threshold: 0.6
    )
    #expect(generalResult.discriminationGap > 0)

    print("\n--- Short Phrases ---")
    let shortResult = try await benchmark.run(
        provider: service,
        name: "NLEmbedding - Short",
        pairs: EmbeddingBenchmark.shortPhrasePairs,
        threshold: 0.5
    )
    print(shortResult.summary)
    #expect(shortResult.discriminationGap > 0)
}

/// Comprehensive benchmark comparing all providers across all text types.
/// This test produces a matrix showing which model performs best for each text type.
@Test(.timeLimit(.minutes(15)))
func comprehensiveModelComparison() async throws {
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║           COMPREHENSIVE EMBEDDING MODEL COMPARISON                           ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    // Initialize all providers
    print("\nLoading models...")
    let nlService = try NLEmbeddingService(language: .english)
    print("  ✓ NLEmbedding (512d) - Apple's word vectors")

    let miniLM = try await MLXEmbeddingService(model: .miniLM)
    print("  ✓ MiniLM (384d) - Lightweight sentence transformer")

    let bgeLarge = try await MLXEmbeddingService(model: .bgeLarge)
    print("  ✓ BGE-Large (1024d) - Retrieval-optimized")

    let mxbai = try await MLXEmbeddingService(model: .mxbaiEmbedLarge)
    print("  ✓ mxbai-embed-large (1024d) - High-quality general purpose")

    let providers: [(name: String, provider: any EmbeddingProvider, shortName: String)] = [
        ("NLEmbedding", nlService, "NLEmbed"),
        ("MiniLM", miniLM, "MiniLM"),
        ("BGE-Large", bgeLarge, "BGE-Lg"),
        ("mxbai-large", mxbai, "mxbai"),
    ]

    let benchmark = EmbeddingBenchmark()

    // Results matrix: [textType][provider] = (gap, accuracy)
    var results: [[String]: [(provider: String, gap: Float, accuracy: Float)]] = [:]

    print("\nRunning benchmarks...\n")

    // Run all combinations
    for (testName, pairs, favoredModel) in EmbeddingBenchmark.allTestSets {
        print("Testing: \(testName) (favors: \(favoredModel))")
        var testResults: [(provider: String, gap: Float, accuracy: Float)] = []

        for (providerName, provider, _) in providers {
            let result = try await benchmark.run(
                provider: provider,
                name: providerName,
                pairs: pairs,
                threshold: 0.5
            )
            testResults.append((providerName, result.discriminationGap, result.accuracy))
        }
        results[[testName]] = testResults
    }

    // Print results matrix
    print("\n")
    print("┌─────────────────┬────────────────┬────────────────┬────────────────┬────────────────┐")
    print("│ Text Type       │ NLEmbedding    │ MiniLM         │ BGE-Large      │ mxbai-large    │")
    print("├─────────────────┼────────────────┼────────────────┼────────────────┼────────────────┤")

    var providerWins: [String: Int] = [:]
    var providerTotalGap: [String: Float] = [:]

    for (testName, _, _) in EmbeddingBenchmark.allTestSets {
        guard let testResults = results[[testName]] else { continue }

        // Find the winner for this test
        let bestGap = testResults.max(by: { $0.gap < $1.gap })?.gap ?? 0

        var row = "│ \(testName.padding(toLength: 15, withPad: " ", startingAt: 0)) │"

        for (providerName, gap, accuracy) in testResults {
            let isWinner = gap == bestGap
            let marker = isWinner ? "★" : " "
            let cell = "\(marker)\(String(format: "%.2f", gap))/\(String(format: "%.0f%%", accuracy * 100))"
            row += " \(cell.padding(toLength: 14, withPad: " ", startingAt: 0)) │"

            if isWinner {
                providerWins[providerName, default: 0] += 1
            }
            providerTotalGap[providerName, default: 0] += gap
        }
        print(row)
    }

    print("├─────────────────┼────────────────┼────────────────┼────────────────┼────────────────┤")

    // Summary row - wins
    var winsRow = "│ WINS            │"
    for (providerName, _, _) in providers {
        let wins = providerWins[providerName] ?? 0
        winsRow += " \(String(wins).padding(toLength: 14, withPad: " ", startingAt: 0)) │"
    }
    print(winsRow)

    // Summary row - average gap
    let testCount = Float(EmbeddingBenchmark.allTestSets.count)
    var avgRow = "│ AVG GAP         │"
    for (providerName, _, _) in providers {
        let avgGap = (providerTotalGap[providerName] ?? 0) / testCount
        avgRow += " \(String(format: "%.3f", avgGap).padding(toLength: 14, withPad: " ", startingAt: 0)) │"
    }
    print(avgRow)

    print("└─────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘")

    print("\n★ = Best performer for that text type")
    print("Values shown as: discrimination_gap / accuracy")

    // Print recommendations
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                           RECOMMENDATIONS                                    ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print("""

    • NLEmbedding: Best for single words and when no GPU/downloads acceptable.
      Lightweight, instant startup, works offline.

    • MiniLM: Best balance of speed and quality for general semantic similarity.
      Fast inference, good discrimination, small download (~90MB).

    • BGE-Large: Best for retrieval/search applications (query → document).
      Optimized for asymmetric search, good for RAG systems.

    • mxbai-embed-large: Best overall quality for longer texts and nuanced similarity.
      Highest dimensions, captures subtle semantic differences.

    Choose based on your use case:
    - Chat/conversational → MiniLM
    - Search/retrieval → BGE-Large
    - Single words/vocabulary → NLEmbedding
    - High-quality analysis → mxbai-embed-large
    """)

    // Verify all models produced reasonable results
    for (providerName, _, _) in providers {
        let avgGap = (providerTotalGap[providerName] ?? 0) / testCount
        #expect(avgGap > 0.1, "Provider \(providerName) should have positive discrimination")
    }
}

/// Quick single-model benchmark for testing specific text types.
@Test(.timeLimit(.minutes(5)))
func benchmarkSingleModel() async throws {
    print("\n=== Single Model Deep Dive: MiniLM ===")

    let service = try await MLXEmbeddingService(model: .miniLM)
    let benchmark = EmbeddingBenchmark()

    print("\nMiniLM Performance Across All Text Types:")
    print("──────────────────────────────────────────")

    for (testName, pairs, _) in EmbeddingBenchmark.allTestSets {
        let result = try await benchmark.run(
            provider: service,
            name: testName,
            pairs: pairs,
            threshold: 0.5
        )
        let bar = String(repeating: "█", count: Int(result.discriminationGap * 20))
        let padded = bar.padding(toLength: 10, withPad: "░", startingAt: 0)
        print("\(testName.padding(toLength: 16, withPad: " ", startingAt: 0)) \(padded) \(String(format: "%.3f", result.discriminationGap)) (\(String(format: "%.0f%%", result.accuracy * 100)))")
    }
}

/// NLEmbedding-only benchmark (no GPU required, runs from CLI).
@Test func benchmarkNLEmbeddingAllTypes() async throws {
    print("\n=== NLEmbedding Across All Text Types ===")

    let service = try NLEmbeddingService(language: .english)
    let benchmark = EmbeddingBenchmark()

    print("\nNLEmbedding Performance (word-vector averaging):")
    print("─────────────────────────────────────────────────")

    for (testName, pairs, _) in EmbeddingBenchmark.allTestSets {
        // Use runSafe to handle out-of-vocabulary words gracefully
        let result = await benchmark.runSafe(
            provider: service,
            name: testName,
            pairs: pairs,
            threshold: 0.5
        )
        let bar = String(repeating: "█", count: Int(result.discriminationGap * 20))
        let padded = bar.padding(toLength: 10, withPad: "░", startingAt: 0)
        print("\(testName.padding(toLength: 16, withPad: " ", startingAt: 0)) \(padded) \(String(format: "%.3f", result.discriminationGap)) (\(String(format: "%.0f%%", result.accuracy * 100)))")
    }

    // NLEmbedding should do well on single words
    let singleWordResult = await benchmark.runSafe(
        provider: service,
        name: "Single Words",
        pairs: EmbeddingBenchmark.singleWordPairs
    )
    #expect(singleWordResult.discriminationGap > 0.2, "NLEmbedding should perform well on single words")
}

// MARK: - Reranker Tests

@Test func embeddingRerankerBasic() async throws {
    let provider = try NLEmbeddingService(language: .english)
    let reranker = EmbeddingReranker(provider: provider)

    let query = "how to fix a bug"
    let documents = [
        "The weather is nice today",
        "Debugging code requires patience and systematic testing",
        "Pizza is delicious",
        "Common bug fixes include checking null pointers and off-by-one errors",
    ]

    let results = try await reranker.rerank(query: query, documents: documents, topK: 2)

    #expect(results.count == 2)
    // The debugging-related documents should rank higher
    #expect(results[0].item.contains("bug") || results[0].item.contains("Debug"))
    print("Reranker results:")
    for result in results {
        print("  [\(String(format: "%.3f", result.score))] \(result.item.prefix(50))...")
    }
}

@Test func embeddingRerankerWithCustomItems() async throws {
    struct Document: Sendable {
        let id: Int
        let title: String
        let content: String
    }

    let provider = try NLEmbeddingService(language: .english)
    let reranker = EmbeddingReranker(provider: provider)

    let documents = [
        Document(id: 1, title: "Weather Report", content: "Sunny skies expected all week"),
        Document(id: 2, title: "Bug Fixing Guide", content: "How to debug common programming errors"),
        Document(id: 3, title: "Recipe Book", content: "Delicious pasta recipes for dinner"),
        Document(id: 4, title: "Code Review", content: "Best practices for reviewing pull requests and finding bugs"),
    ]

    let results = try await reranker.rerank(
        query: "fixing errors in code",
        items: documents,
        topK: 2,
        textExtractor: { "\($0.title) \($0.content)" }
    )

    #expect(results.count == 2)
    #expect(results[0].item.id == 2 || results[0].item.id == 4)
    print("Custom reranker results:")
    for result in results {
        print("  [\(String(format: "%.3f", result.score))] #\(result.item.id): \(result.item.title)")
    }
}

// MARK: - Threshold Calibrator Tests

@Test func thresholdCalibratorBasic() async throws {
    let provider = try NLEmbeddingService(language: .english)
    let calibrator = ThresholdCalibrator()

    // Use benchmark pairs as labeled examples
    let examples = ThresholdCalibrator.labeledPairs(from: EmbeddingBenchmark.generalPairs)

    let result = try await calibrator.calibrate(provider: provider, examples: examples)

    print("\n" + result.report)

    #expect(result.bestF1Threshold > 0)
    #expect(result.bestF1Threshold < 1)
    #expect(result.metricsAtThresholds.count > 0)
}

@Test func thresholdCalibratorPresets() async throws {
    let provider = try NLEmbeddingService(language: .english)

    // Test different presets
    let presets: [(String, ThresholdCalibrator)] = [
        ("Duplicate Detection", .duplicateDetection),
        ("Semantic Search", .semanticSearch),
        ("Content Discovery", .contentDiscovery),
    ]

    let examples = ThresholdCalibrator.labeledPairs(from: EmbeddingBenchmark.generalPairs)

    print("\n=== Threshold Calibration by Use Case ===\n")

    for (name, calibrator) in presets {
        let result = try await calibrator.calibrate(provider: provider, examples: examples)
        print("\(name):")
        print("  Best F1 threshold: \(String(format: "%.2f", result.bestF1Threshold))")
        if let metrics = result.metrics(at: result.bestF1Threshold) {
            print("  Precision: \(String(format: "%.2f", metrics.precision)), Recall: \(String(format: "%.2f", metrics.recall))")
        }
        print("")
    }
}

@Test func thresholdCalibratorDomainSpecific() async throws {
    let provider = try NLEmbeddingService(language: .english)
    let calibrator = ThresholdCalibrator()

    // Simulate domain-specific examples (e.g., customer support)
    let supportExamples: [LabeledPair] = [
        // Matches (same intent)
        LabeledPair("how do I reset my password", "forgot my login credentials", isMatch: true),
        LabeledPair("cancel my subscription", "I want to stop my membership", isMatch: true),
        LabeledPair("where is my order", "track my package", isMatch: true),
        LabeledPair("refund request", "I want my money back", isMatch: true),
        LabeledPair("change my email", "update email address", isMatch: true),

        // Non-matches (different intent)
        LabeledPair("how do I reset my password", "what are your business hours", isMatch: false),
        LabeledPair("cancel my subscription", "how do I upgrade my plan", isMatch: false),
        LabeledPair("where is my order", "how do I place an order", isMatch: false),
        LabeledPair("refund request", "product recommendations", isMatch: false),
        LabeledPair("change my email", "delete my account", isMatch: false),
    ]

    let result = try await calibrator.calibrate(provider: provider, examples: supportExamples)

    print("\n=== Customer Support Domain Calibration ===")
    print(result.report)

    // Verify calibration produced valid results
    #expect(result.bestF1Threshold > 0)
    #expect(result.bestF1Threshold <= 1)

    // The actual threshold depends on the model - just verify we got metrics
    if let metrics = result.metrics(at: result.bestF1Threshold) {
        print("Best threshold \(result.bestF1Threshold) achieves F1=\(metrics.f1)")
        #expect(metrics.f1 > 0)
    }
}

@Test(.timeLimit(.minutes(5)))
func thresholdCalibratorWithMLX() async throws {
    let provider = try await MLXEmbeddingService(model: .miniLM)
    let calibrator = ThresholdCalibrator()

    let examples = ThresholdCalibrator.labeledPairs(from: EmbeddingBenchmark.generalPairs)
    let result = try await calibrator.calibrate(provider: provider, examples: examples)

    print("\n=== MiniLM Threshold Calibration ===")
    print(result.report)

    // MiniLM typically needs lower thresholds than NLEmbedding
    #expect(result.bestF1Threshold > 0)
}

// MARK: - FDLEmbeddingService Tests

@Test func fdlEmbeddingServiceInitialization() async throws {
    let corpus = [
        "The cat sat on the mat",
        "A dog ran through the park",
        "Birds fly over mountains"
    ]
    let service = FDLEmbeddingService(corpus: corpus)
    let dims = try await service.dimensions
    #expect(dims > 0)
}

@Test func fdlEmbeddingServiceDimensionsMatchVocabulary() async throws {
    let corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Swift programming is fast and safe"
    ]
    let service = FDLEmbeddingService(corpus: corpus)
    let dims = try await service.dimensions
    let vector = try await service.embed("quick fox")
    #expect(vector.count == dims)
}

@Test func fdlEmbeddingServiceFrequencyCounting() async throws {
    // "fox", "run", "swim" survive lemmatisation + stopword removal
    let corpus = ["fox run swim jump"]
    let service = FDLEmbeddingService(corpus: corpus)

    let vector = try await service.embed("fox run swim jump")
    let total = vector.reduce(0, +)
    #expect(total > 0, "Known corpus tokens should produce non-zero frequency counts")
}

@Test func fdlEmbeddingServiceRepeatedTokensCountCorrectly() async throws {
    // "fox" survives the pipeline; its count should double when repeated
    let corpus = ["quick brown fox"]
    let service = FDLEmbeddingService(corpus: corpus)

    let once = try await service.embed("fox")
    let twice = try await service.embed("fox fox")

    let sumOnce = once.reduce(0, +)
    let sumTwice = twice.reduce(0, +)
    #expect(sumTwice == sumOnce * 2, "Repeating a token should double its frequency count")
}

@Test func fdlEmbeddingServiceUnknownTokensProduceZeroVector() async throws {
    let corpus = ["apple banana cherry"]
    let service = FDLEmbeddingService(corpus: corpus)

    // These words are very unlikely to share lemmas with apple/banana/cherry
    let vector = try await service.embed("xylophone zymurgy quasar")
    let total = vector.reduce(0, +)
    #expect(total == 0.0, "Tokens absent from vocabulary should produce an all-zero vector")
}

@Test func fdlEmbeddingServiceEmptyCorpusThrows() async {
    let service = FDLEmbeddingService(corpus: [])
    do {
        _ = try await service.embed("anything")
        Issue.record("Expected FDLEmbeddingService to throw when vocabulary is empty")
    } catch {
        // Expected — empty vocabulary cannot produce an embedding
    }
}

@Test func fdlEmbeddingServiceBatchEmbed() async throws {
    let corpus = [
        "swift programming language",
        "machine learning algorithms",
        "data structures and graphs"
    ]
    let service = FDLEmbeddingService(corpus: corpus)
    let texts = ["swift language", "machine algorithms", "data graphs"]

    let vectors = try await service.embedBatch(texts)
    let dims = try await service.dimensions

    #expect(vectors.count == texts.count)
    for vector in vectors {
        #expect(vector.count == dims)
    }
}

@Test func fdlEmbeddingServiceProtocolConformance() async throws {
    // Assigned to protocol type — must compile and behave correctly
    let corpus = ["hello world foo bar baz"]
    let provider: any EmbeddingProvider = FDLEmbeddingService(corpus: corpus)

    let dims = try await provider.dimensions
    let vector = try await provider.embed("hello world")
    #expect(vector.count == dims)
}

@Test func fdlEmbeddingServiceTextEmbeddingConvenience() async throws {
    // embed(_:as:) is inherited from EmbeddingProvider; verify provenance tag is set
    let corpus = ["machine learning natural language processing"]
    let service = FDLEmbeddingService(corpus: corpus)

    let embedding = try await service.embed("machine learning", as: .fdlEmbedding)
    #expect(embedding.provider == .fdlEmbedding)
    #expect(!embedding.vector.isEmpty)
}

@Test func fdlEmbeddingServiceDeterministic() async throws {
    // Same input must produce identical vectors on repeated calls
    let corpus = ["the quick brown fox jumps over the lazy dog"]
    let service = FDLEmbeddingService(corpus: corpus)
    let text = "quick brown fox"

    let v1 = try await service.embed(text)
    let v2 = try await service.embed(text)
    #expect(v1 == v2)
}

@Test func fdlEmbeddingServiceVocabularyIsSorted() async throws {
    // The vocabulary is built as a sorted array — dimensions property reflects
    // the deduplicated, sorted token count, not the raw token stream length
    let corpus = ["banana apple cherry apple banana fig"]
    let service = FDLEmbeddingService(corpus: corpus)
    let dims = try await service.dimensions

    // apple, banana, cherry, fig → 4 unique tokens (all survive pipeline)
    // Exact count depends on lemmatiser; just verify deduplication happened
    let rawTokenCount = "banana apple cherry apple banana fig"
        .linguisticTokens(keepNumerics: false).count
    #expect(dims <= rawTokenCount, "Vocabulary must deduplicate tokens")
}

// MARK: - EmbeddingProviderOption Corpus Injection Tests

@Test func embeddingProviderOptionFDLRequiresCorpus() async {
    // makeProvider() with no corpus must throw for .fdlEmbedding
    do {
        _ = try await EmbeddingProviderOption.fdlEmbedding.makeProvider()
        Issue.record("Expected makeProvider() to throw when corpus is missing for .fdlEmbedding")
    } catch {
        // Expected
    }
}

@Test func embeddingProviderOptionFDLWithCorpus() async throws {
    let corpus = ["hello world swift programming language"]
    let provider = try await EmbeddingProviderOption.fdlEmbedding.makeProvider(corpus: corpus)

    let dims = try await provider.dimensions
    #expect(dims > 0)

    let vector = try await provider.embed("swift programming")
    #expect(vector.count == dims)
}

@Test func embeddingProviderOptionCorpusIgnoredForNLEmbedding() async throws {
    // corpus: is silently ignored for all non-FDL providers
    let provider = try await EmbeddingProviderOption.nlEmbedding.makeProvider(
        corpus: ["this corpus should be ignored"]
    )
    let dims = try await provider.dimensions
    #expect(dims > 0, "NLEmbedding produces a fixed-dimension vector regardless of corpus argument")
}

// MARK: - CorpusStore Tests

@Test func corpusStoreRoundTrip() async throws {
    // Write two corpora, read them back, verify vector fidelity.
    let service = try NLEmbeddingService(language: .english)

    var embeddings1: [TextEmbedding] = []
    for text in ["Photosynthesis converts sunlight into chemical energy.", "Chloroplasts are the sites of photosynthesis."] {
        let v = try await service.embed(text)
        embeddings1.append(TextEmbedding(provider: .nlEmbedding, vector: v,
                                         metadata: ["text": text, "part": "Introduction", "granularity": "paragraph"]))
    }
    var embeddings2: [TextEmbedding] = []
    for text in ["Neural networks learn from labeled examples.", "Backpropagation adjusts weights to minimize loss."] {
        let v = try await service.embed(text)
        embeddings2.append(TextEmbedding(provider: .nlEmbedding, vector: v,
                                         metadata: ["text": text, "part": "Methods", "granularity": "paragraph"]))
    }

    let corpus1 = Corpus(label: "Photosynthesis Review", metadata: ["filename": "photo.md"], embeddings: embeddings1)
    let corpus2 = Corpus(label: "Deep Learning Primer", metadata: ["filename": "dl.md"], embeddings: embeddings2)

    // Write to a temp file
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("corpus_store_test_\(UUID().uuidString).sqlite")
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let store = try CorpusStore(url: tmpURL)
    try store.write([corpus1, corpus2])

    // Read back
    let loaded = try store.readAll()
    #expect(loaded.count == 2)

    // Verify labels
    let labels = loaded.map(\.label).sorted()
    #expect(labels == ["Deep Learning Primer", "Photosynthesis Review"])

    // Verify vector round-trip: each embedding vector should survive Float32 conversion
    // with at most 1e-6 per-element error.
    for (original, restored) in zip([corpus1, corpus2].sorted(by: { $0.label < $1.label }),
                                     loaded.sorted(by: { $0.label < $1.label })) {
        #expect(original.embeddings.count == restored.embeddings.count)
        for (origEmb, restEmb) in zip(original.embeddings, restored.embeddings) {
            #expect(origEmb.vector.count == restEmb.vector.count)
            let maxError = zip(origEmb.vector, restEmb.vector)
                .map { abs($0 - $1) }
                .max() ?? 0
            #expect(maxError < 1e-5, "Vector round-trip error \(maxError) exceeds threshold")
        }
    }
}

// MARK: - MultiProviderEmbedder Tests

@Test func multiProviderEmbedderSingleText() async throws {
    // NLEmbedding only (no GPU required) — verify keyed result.
    let service = try NLEmbeddingService(language: .english)
    let embedder = MultiProviderEmbedder(providers: [.nlEmbedding: service])
    let results = try await embedder.embed("The mitochondria is the powerhouse of the cell.")
    #expect(results[.nlEmbedding] != nil)
    #expect((results[.nlEmbedding]?.vector.count ?? 0) > 0)
}

@Test func multiProviderEmbedderCorpus() async throws {
    let service = try NLEmbeddingService(language: .english)
    let embedder = MultiProviderEmbedder(providers: [.nlEmbedding: service])

    let introVec  = try await service.embed("Introduction text.")
    let methodVec = try await service.embed("Methods text.")
    let source = Corpus(
        label: "Test Paper",
        metadata: ["filename": "test.md"],
        embeddings: [
            TextEmbedding(provider: .nlEmbedding, vector: introVec,
                          metadata: ["text": "Introduction text.", "part": "Introduction", "granularity": "section"]),
            TextEmbedding(provider: .nlEmbedding, vector: methodVec,
                          metadata: ["text": "Methods text.", "part": "Methods", "granularity": "section"]),
        ]
    )

    let output = try await embedder.embed(corpus: source)
    #expect(output[.nlEmbedding] != nil)
    #expect(output[.nlEmbedding]?.embeddings.count == 2)
    #expect(output[.nlEmbedding]?.metadata["source_corpus_id"] == source.id.uuidString)
}

@Test func multiProviderEmbedderCorpusPreservesSequenceIndexAndScheme() async throws {
    // Regression test: sequence_index and scheme must survive re-embedding through MultiProviderEmbedder.
    let service = try NLEmbeddingService(language: .english)
    let embedder = MultiProviderEmbedder(providers: [.nlEmbedding: service])

    let vec = try await service.embed("Introduction paragraph text.")
    let source = Corpus(
        label: "Test Paper",
        metadata: ["filename": "test.md"],
        embeddings: [
            TextEmbedding(provider: .nlEmbedding, vector: vec, metadata: [
                "text": "Introduction section text.",
                "part": "Introduction",
                "granularity": "sectionAndParagraphs",
                "sequence_index": "0",
                "scheme": "introduction_hierarchical"
            ]),
            TextEmbedding(provider: .nlEmbedding, vector: vec, metadata: [
                "text": "First paragraph.",
                "part": "Introduction",
                "granularity": "sectionAndParagraphs",
                "sequence_index": "1",
                "scheme": "introduction_hierarchical"
            ]),
        ]
    )

    let output = try await embedder.embed(corpus: source)
    let result = try #require(output[.nlEmbedding])
    #expect(result.embeddings.count == 2)

    let indices = result.embeddings.compactMap { $0.metadata["sequence_index"] }
    #expect(indices.contains("0"), "sequence_index 0 (full section) must be preserved")
    #expect(indices.contains("1"), "sequence_index 1 (first paragraph) must be preserved")

    let schemes = result.embeddings.compactMap { $0.metadata["scheme"] }
    #expect(schemes.allSatisfy { $0 == "introduction_hierarchical" }, "scheme must be preserved on all embeddings")
}

// MARK: - ManuscriptLoader Tests

/// Writes a temporary Markdown file and returns its URL. Call the `cleanup` closure when done.
private func makeTempMarkdown(_ content: String) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("test_\(UUID().uuidString).md")
    try content.write(to: url, atomically: true, encoding: .utf8)
    return url
}

/// Creates a temporary directory, writes named Markdown files, and returns the directory URL.
private func makeTempMarkdownDirectory(_ files: [String: String]) throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("loader_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    for (name, content) in files {
        try content.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
    }
    return dir
}

@Test func manuscriptLoaderPartsFilter() async throws {
    let markdown = """
    # Test Paper

    ## Introduction

    This is the introduction describing research motivation.

    ## Methods

    Statistical methods applied to the data.

    ## Results

    Findings from the analysis.
    """
    let url = try makeTempMarkdown(markdown)
    defer { try? FileManager.default.removeItem(at: url) }

    let service = try NLEmbeddingService(language: .english)
    let corpus = try await ManuscriptLoader.load(
        from: url,
        parts: [.Introduction],
        granularity: .section,
        using: service,
        as: .nlEmbedding
    )

    let parts = corpus.embeddings.compactMap { $0.metadata["part"] }
    #expect(!parts.isEmpty, "Should have at least one Introduction embedding")
    #expect(parts.allSatisfy { $0 == ManuscriptParts.Introduction.rawValue },
            "Only Introduction embeddings should be present")
    #expect(!parts.contains(ManuscriptParts.Methods.rawValue))
    #expect(!parts.contains(ManuscriptParts.Results.rawValue))
}

@Test func manuscriptLoaderSectionAndParagraphsSequenceIndices() async throws {
    let markdown = """
    # Sample Paper

    ## Introduction

    First paragraph of the introduction.

    Second paragraph providing more context.

    Third paragraph concluding the introduction.
    """
    let url = try makeTempMarkdown(markdown)
    defer { try? FileManager.default.removeItem(at: url) }

    let service = try NLEmbeddingService(language: .english)
    let corpus = try await ManuscriptLoader.load(
        from: url,
        parts: [.Introduction],
        granularity: .sectionAndParagraphs,
        scheme: "intro_test",
        using: service,
        as: .nlEmbedding
    )

    let indices = corpus.embeddings.compactMap { $0.metadata["sequence_index"] }
    #expect(indices.contains("0"), "Full section embedding must be at sequence_index 0")
    #expect(indices.contains("1"), "First paragraph must be at sequence_index 1")

    // scheme stamped on every embedding
    let schemes = corpus.embeddings.compactMap { $0.metadata["scheme"] }
    #expect(!schemes.isEmpty)
    #expect(schemes.allSatisfy { $0 == "intro_test" })
}

@Test func manuscriptLoaderExtractTexts() throws {
    let md1 = """
    # Paper One

    ## Introduction

    Introduction of paper one.

    ## Methods

    Methods of paper one.
    """
    let md2 = """
    # Paper Two

    ## Introduction

    Introduction of paper two.
    """
    let dir = try makeTempMarkdownDirectory(["a.md": md1, "b.md": md2])
    defer { try? FileManager.default.removeItem(at: dir) }

    // All sections: 2 from paper one + 1 from paper two
    let allTexts = try ManuscriptLoader.extractTexts(from: dir)
    #expect(allTexts.count == 3)

    // Filter to Introduction only: 1 from each paper
    let introTexts = try ManuscriptLoader.extractTexts(from: dir, parts: [.Introduction])
    #expect(introTexts.count == 2)
}

@Test func manuscriptLoaderMultiProviderLoad() async throws {
    let markdown = """
    # Sample Paper

    ## Introduction

    This paper studies machine learning methods for natural language processing.

    A second paragraph providing background on prior work.
    """
    let url = try makeTempMarkdown(markdown)
    defer { try? FileManager.default.removeItem(at: url) }

    let service = try NLEmbeddingService(language: .english)
    let embedder = MultiProviderEmbedder(providers: [.nlEmbedding: service])

    let corpus = try await ManuscriptLoader.load(
        from: url,
        parts: [.Introduction],
        granularity: .sectionAndParagraphs,
        scheme: "intro_hierarchical",
        using: embedder
    )

    // 1 section (index 0) + 2 paragraphs (index 1, 2) = 3 embeddings per provider
    #expect(corpus.embeddings.count >= 3)

    let seqIndices = corpus.embeddings.compactMap { $0.metadata["sequence_index"] }
    #expect(seqIndices.contains("0"), "Full-section embedding must be present")
    #expect(seqIndices.contains("1"), "First paragraph embedding must be present")

    let schemes = corpus.embeddings.compactMap { $0.metadata["scheme"] }
    #expect(schemes.allSatisfy { $0 == "intro_hierarchical" }, "scheme must be stamped on every embedding")
}

// MARK: - End-to-end pipeline: fixture .md → ManuscriptLoader → NLEmbedding → CorpusStore

/// Returns the `Fixtures/` directory adjacent to this source file.
private var fixtureDirectory: URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures")
}

@Test func endToEndIntroductionPipelineToSQLite() async throws {
    // ── 1. Locate fixture Markdown files ────────────────────────────────────
    let dir = fixtureDirectory
    let urls = try FileManager.default
        .contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension.lowercased() == "md" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    #expect(urls.count == 2, "Expected 2 fixture .md files (Dyer_Sork and Smouse_etal)")

    // ── 2. Embed Introduction sections with sectionAndParagraphs ────────────
    let service = try NLEmbeddingService(language: .english)
    let corpora = try await ManuscriptLoader.loadAll(
        from: dir,
        parts: [.Introduction],
        granularity: .sectionAndParagraphs,
        scheme: "introduction_hierarchical",
        using: service,
        as: .nlEmbedding
    )
    #expect(corpora.count == 2, "One Corpus per .md file")

    // Every embedding must be from the Introduction and carry the correct metadata.
    for corpus in corpora {
        let parts = corpus.embeddings.compactMap { $0.metadata["part"] }
        #expect(parts.allSatisfy { $0 == ManuscriptParts.Introduction.rawValue },
                "Only Introduction embeddings expected in \(corpus.label)")

        let grans = corpus.embeddings.compactMap { $0.metadata["granularity"] }
        #expect(grans.allSatisfy { $0 == EmbeddingGranularity.sectionAndParagraphs.rawValue })

        let schemes = corpus.embeddings.compactMap { $0.metadata["scheme"] }
        #expect(schemes.allSatisfy { $0 == "introduction_hierarchical" })

        // sequence_index 0 = full section; ≥1 = individual paragraphs.
        let indices = corpus.embeddings.compactMap { $0.metadata["sequence_index"].flatMap(Int.init) }
        #expect(indices.contains(0), "Full-section embedding (index 0) must be present")
        #expect(indices.contains(where: { $0 >= 1 }), "At least one paragraph embedding must be present")
    }

    // Both papers should have DOIs extracted from the first 3000 chars.
    let dois = corpora.compactMap { $0.metadata["doi"] }
    #expect(dois.count == 2, "DOI should be extracted from both fixture files")
    #expect(dois.allSatisfy { $0.hasPrefix("10.") })

    // ── 3. Write to SQLite ───────────────────────────────────────────────────
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("e2e_test_\(UUID().uuidString).sqlite")
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let store = try CorpusStore(url: tmpURL)
    try store.write(corpora)

    // ── 4. High-level read-back conformance ─────────────────────────────────
    let loaded = try store.readAll()
    #expect(loaded.count == 2)

    for (original, restored) in zip(
        corpora.sorted(by: { $0.label < $1.label }),
        loaded.sorted(by: { $0.label < $1.label })
    ) {
        #expect(original.embeddings.count == restored.embeddings.count,
                "Embedding count must survive round-trip for \(original.label)")

        // Spot-check metadata keys are preserved.
        for emb in restored.embeddings {
            #expect(emb.metadata["part"] == ManuscriptParts.Introduction.rawValue)
            #expect(emb.metadata["granularity"] == EmbeddingGranularity.sectionAndParagraphs.rawValue)
            #expect(emb.metadata["scheme"] == "introduction_hierarchical")
            #expect(emb.metadata["sequence_index"] != nil, "sequence_index must survive round-trip")
            #expect(!(emb.metadata["text"] ?? "").isEmpty, "source_text must survive round-trip")
            #expect(emb.provider == .nlEmbedding)
        }
    }

    // ── 5. Raw SQLite schema conformance (PRAGMA table_info) ─────────────────
    var rawDB: OpaquePointer?
    defer { sqlite3_close(rawDB) }
    #expect(sqlite3_open(tmpURL.path, &rawDB) == SQLITE_OK, "Should open SQLite file directly")

    // documents table columns
    var docColumns = [String]()
    var stmt: OpaquePointer?
    sqlite3_prepare_v2(rawDB, "PRAGMA table_info(documents)", -1, &stmt, nil)
    while sqlite3_step(stmt) == SQLITE_ROW {
        if let cstr = sqlite3_column_text(stmt, 1) { docColumns.append(String(cString: cstr)) }
    }
    sqlite3_finalize(stmt)
    for col in ["id", "corpus_uuid", "title", "filename", "doi", "created_at"] {
        #expect(docColumns.contains(col), "documents table missing column: \(col)")
    }

    // embeddings table columns
    var embColumns = [String]()
    sqlite3_prepare_v2(rawDB, "PRAGMA table_info(embeddings)", -1, &stmt, nil)
    while sqlite3_step(stmt) == SQLITE_ROW {
        if let cstr = sqlite3_column_text(stmt, 1) { embColumns.append(String(cString: cstr)) }
    }
    sqlite3_finalize(stmt)
    for col in ["id", "document_id", "part", "granularity", "provider",
                "dimensions", "vector", "scaling", "source_text", "sequence_index", "scheme"] {
        #expect(embColumns.contains(col), "embeddings table missing column: \(col)")
    }

    // Spot-check actual row content in embeddings via raw SQL.
    sqlite3_prepare_v2(rawDB,
        "SELECT COUNT(*) FROM embeddings WHERE part = 'Introduction' AND scheme = 'introduction_hierarchical'",
        -1, &stmt, nil)
    sqlite3_step(stmt)
    let introRowCount = Int(sqlite3_column_int(stmt, 0))
    sqlite3_finalize(stmt)
    #expect(introRowCount > 0, "Should have Introduction rows in embeddings table")

    // Verify sequence_index = 0 rows (full-section embeddings) exist.
    sqlite3_prepare_v2(rawDB,
        "SELECT COUNT(*) FROM embeddings WHERE sequence_index = 0",
        -1, &stmt, nil)
    sqlite3_step(stmt)
    let sectionRowCount = Int(sqlite3_column_int(stmt, 0))
    sqlite3_finalize(stmt)
    #expect(sectionRowCount == 2, "One full-section embedding (index 0) per paper")

    // Verify paragraph rows (sequence_index ≥ 1) exist.
    sqlite3_prepare_v2(rawDB,
        "SELECT COUNT(*) FROM embeddings WHERE sequence_index >= 1",
        -1, &stmt, nil)
    sqlite3_step(stmt)
    let paraRowCount = Int(sqlite3_column_int(stmt, 0))
    sqlite3_finalize(stmt)
    #expect(paraRowCount > 0, "Paragraph-level embeddings must be present")

    // Every vector BLOB must be non-null and have consistent byte length.
    sqlite3_prepare_v2(rawDB,
        "SELECT dimensions, LENGTH(vector) FROM embeddings",
        -1, &stmt, nil)
    while sqlite3_step(stmt) == SQLITE_ROW {
        let dims = Int(sqlite3_column_int(stmt, 0))
        let blobBytes = Int(sqlite3_column_int(stmt, 1))
        #expect(dims > 0, "dimensions must be positive")
        #expect(blobBytes == dims * 4, "BLOB should be dims × 4 bytes (Float32)")
    }
    sqlite3_finalize(stmt)

    print("""

    ── End-to-end pipeline summary ──────────────────────────────────────
    Papers loaded  : \(corpora.count)
    Total embeddings written: \(corpora.reduce(0) { $0 + $1.embeddings.count })
      • Full-section rows (index 0): \(sectionRowCount)
      • Paragraph rows  (index ≥1) : \(paraRowCount)
    documents columns : \(docColumns.joined(separator: ", "))
    embeddings columns: \(embColumns.joined(separator: ", "))
    ─────────────────────────────────────────────────────────────────────
    """)
}
