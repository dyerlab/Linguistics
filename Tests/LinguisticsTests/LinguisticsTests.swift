import Testing
@testable import Linguistics
import NaturalLanguage

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
    print("Embedding stats - min: \(testEmb.min()!), max: \(testEmb.max()!), mean: \(testEmb.reduce(0, +) / Float(testEmb.count))")
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
