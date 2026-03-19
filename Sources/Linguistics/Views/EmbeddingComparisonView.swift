//
//  EmbeddingComparisonView.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/5/26.
//

import SwiftUI
import Charts

// MARK: - EmbeddingComparisonViewModel

/// Observable view model for embedding comparison UI.
///
/// `EmbeddingComparisonViewModel` manages the state for testing embedding models
/// on custom text pairs and analyzing the results. Use it to drive an interactive
/// comparison workflow.
///
/// ## Overview
///
/// The view model handles:
/// - Managing user-defined text pairs
/// - Running comparisons against multiple embedding providers
/// - Computing calibration curves for threshold selection
/// - Generating recommendations based on results
///
/// ## Usage
///
/// ```swift
/// struct MyComparisonView: View {
///     @State private var viewModel = EmbeddingComparisonViewModel()
///
///     var body: some View {
///         // Display input UI...
///
///         Button("Run Comparison") {
///             Task {
///                 await viewModel.runNLEmbeddingComparison()
///             }
///         }
///
///         // Display results from viewModel.comparisonResults...
///     }
/// }
/// ```
///
/// ## Topics
///
/// ### Managing Pairs
/// - ``customPairs``
/// - ``addPair()``
/// - ``removePair(at:)``
/// - ``loadPresetPairs(_:)``
///
/// ### Running Comparisons
/// - ``runNLEmbeddingComparison()``
/// - ``runFullComparison()``
///
/// ### Results
/// - ``comparisonResults``
/// - ``calibrationResult``
/// - ``recommendation``
@MainActor
@Observable
public final class EmbeddingComparisonViewModel {

    // MARK: - State

    /// Whether a comparison is currently running.
    public var isLoading = false

    /// Status message displayed during loading.
    public var loadingMessage = ""

    /// Error message if comparison failed.
    public var errorMessage: String?

    /// User-defined text pairs for testing.
    ///
    /// Edit these pairs to test embedding models on your specific domain.
    public var customPairs: [EditableTextPair] = []

    /// Results from running comparisons.
    ///
    /// Contains metrics for each provider tested.
    public var comparisonResults: [ProviderResult] = []

    /// Calibration results for threshold analysis.
    ///
    /// Used to generate precision/recall curves and find optimal thresholds.
    public var calibrationResult: CalibrationResult?

    /// Selected provider for detailed view.
    public var selectedProvider: String?

    /// Current analysis recommendation based on results.
    public var recommendation: Recommendation?

    // MARK: - Types

    /// An editable text pair for user input.
    ///
    /// Used in the UI to let users create custom test pairs for their domain.
    public struct EditableTextPair: Identifiable {

        /// Unique identifier for SwiftUI list operations.
        public let id = UUID()

        /// First text in the pair (e.g., query).
        public var text1: String

        /// Second text in the pair (e.g., document).
        public var text2: String

        /// Whether these texts should be considered a match.
        public var isMatch: Bool

        /// Descriptive label for this pair.
        public var label: String

        /// Creates a new editable text pair.
        ///
        /// - Parameters:
        ///   - text1: First text (default: empty)
        ///   - text2: Second text (default: empty)
        ///   - isMatch: Whether texts should match (default: true)
        ///   - label: Descriptive label (default: empty)
        public init(text1: String = "", text2: String = "", isMatch: Bool = true, label: String = "") {
            self.text1 = text1
            self.text2 = text2
            self.isMatch = isMatch
            self.label = label
        }
    }

    /// Results from comparing a single embedding provider.
    ///
    /// Contains aggregate metrics and individual scores for analysis.
    public struct ProviderResult: Identifiable {

        /// Unique identifier.
        public let id = UUID()

        /// Provider name (e.g., "MiniLM", "BGE-Large").
        public let name: String

        /// Embedding vector dimensions.
        public let dimensions: Int

        /// Average similarity for matching pairs.
        public let avgHighSimilarity: Float

        /// Average similarity for non-matching pairs.
        public let avgLowSimilarity: Float

        /// Discrimination gap (high - low).
        public let discriminationGap: Float

        /// Classification accuracy at threshold 0.5.
        public let accuracy: Float

        /// Individual scores for each pair.
        public let scores: [(label: String, similarity: Float, isMatch: Bool)]
    }

    /// Recommendation generated from comparison results.
    ///
    /// Provides guidance on which provider to use, whether a reranker
    /// is needed, and what similarity threshold to set.
    public struct Recommendation {

        /// Name of the best-performing provider.
        public let bestProvider: String

        /// Explanation of why this provider is recommended.
        public let reasoning: String

        /// Whether a reranker is recommended for this use case.
        public let needsReranker: Bool

        /// Explanation of the reranker recommendation.
        public let rerankerReason: String

        /// Recommended similarity threshold.
        public let suggestedThreshold: Float

        /// Explanation of the threshold recommendation.
        public let thresholdReason: String
    }

    // MARK: - Initialization

    /// Creates a new view model with example pairs.
    public init() {
        // Start with some example pairs
        customPairs = [
            EditableTextPair(text1: "Example query text", text2: "Similar document text", isMatch: true, label: "Example match"),
            EditableTextPair(text1: "Example query text", text2: "Unrelated content here", isMatch: false, label: "Example non-match"),
        ]
    }

    // MARK: - Actions

    /// Adds an empty text pair for user input.
    public func addPair() {
        customPairs.append(EditableTextPair())
    }

    /// Removes a text pair at the specified index.
    ///
    /// - Parameter index: The index of the pair to remove
    public func removePair(at index: Int) {
        guard customPairs.indices.contains(index) else { return }
        customPairs.remove(at: index)
    }

    /// Loads a preset collection of text pairs.
    ///
    /// Replaces current custom pairs with the preset's pairs.
    ///
    /// - Parameter preset: The preset to load
    public func loadPresetPairs(_ preset: PresetPairs) {
        customPairs = preset.pairs.map { pair in
            EditableTextPair(
                text1: pair.text1,
                text2: pair.text2,
                isMatch: pair.shouldBeSimilar,
                label: pair.label ?? ""
            )
        }
    }

    /// Available preset pair collections.
    ///
    /// Each preset contains curated pairs for a specific domain.
    public enum PresetPairs: String, CaseIterable {

        /// General-purpose pairs covering everyday topics.
        case general = "General"

        /// Technical and programming-related pairs.
        case technical = "Technical"

        /// Question-answering pairs for FAQ applications.
        case questions = "Questions"

        /// Scientific and academic pairs.
        case scientific = "Scientific"

        /// Casual and conversational pairs.
        case conversational = "Conversational"

        /// The underlying text pairs for this preset.
        var pairs: [TextPair] {
            switch self {
            case .general: return EmbeddingBenchmark.generalPairs
            case .technical: return EmbeddingBenchmark.technicalPairs
            case .questions: return EmbeddingBenchmark.questionPairs
            case .scientific: return EmbeddingBenchmark.scientificPairs
            case .conversational: return EmbeddingBenchmark.conversationalPairs
            }
        }
    }

    /// Runs a quick comparison using NLEmbedding only.
    ///
    /// This is a fast test that doesn't require GPU or model downloads.
    /// Use this for initial testing before running a full comparison.
    ///
    /// Results are stored in ``comparisonResults``, ``calibrationResult``,
    /// and ``recommendation``.
    public func runNLEmbeddingComparison() async {
        guard !customPairs.isEmpty else {
            errorMessage = "Add at least one text pair to compare"
            return
        }

        isLoading = true
        loadingMessage = "Running NLEmbedding comparison..."
        errorMessage = nil
        comparisonResults = []

        do {
            let service = try NLEmbeddingService(language: .english)
            let result = try await runComparison(provider: service, name: "NLEmbedding", dimensions: 512)
            comparisonResults = [result]

            // Run calibration
            await runCalibration(provider: service)

            // Generate recommendation
            generateRecommendation()

        } catch {
            errorMessage = "Error: \(error.localizedDescription)"
        }

        isLoading = false
    }

    /// Runs a full comparison with all available embedding providers.
    ///
    /// Tests NLEmbedding, MiniLM, BGE-Large, and mxbai-large models.
    /// Requires GPU (Apple Silicon) and may download models (~500MB total).
    ///
    /// Results are stored in ``comparisonResults``, ``calibrationResult``,
    /// and ``recommendation``.
    ///
    /// - Note: Must run from Xcode for Metal shader compilation.
    @available(macOS 14, iOS 17, *)
    public func runFullComparison() async {
        guard !customPairs.isEmpty else {
            errorMessage = "Add at least one text pair to compare"
            return
        }

        isLoading = true
        errorMessage = nil
        comparisonResults = []

        do {
            // NLEmbedding
            loadingMessage = "Loading NLEmbedding..."
            let nlService = try NLEmbeddingService(language: .english)
            let nlResult = try await runComparison(provider: nlService, name: "NLEmbedding", dimensions: 512)
            comparisonResults.append(nlResult)

            // MiniLM
            loadingMessage = "Loading MiniLM (384d)..."
            let miniLM = try await MLXEmbeddingService(model: .miniLM)
            let miniLMResult = try await runComparison(provider: miniLM, name: "MiniLM", dimensions: 384)
            comparisonResults.append(miniLMResult)

            // BGE-Large
            loadingMessage = "Loading BGE-Large (1024d)..."
            let bgeLarge = try await MLXEmbeddingService(model: .bgeLarge)
            let bgeResult = try await runComparison(provider: bgeLarge, name: "BGE-Large", dimensions: 1024)
            comparisonResults.append(bgeResult)

            // mxbai
            loadingMessage = "Loading mxbai-embed-large (1024d)..."
            let mxbai = try await MLXEmbeddingService(model: .mxbaiEmbedLarge)
            let mxbaiResult = try await runComparison(provider: mxbai, name: "mxbai-large", dimensions: 1024)
            comparisonResults.append(mxbaiResult)

            // Run calibration with best provider
            if let best = comparisonResults.max(by: { $0.discriminationGap < $1.discriminationGap }) {
                loadingMessage = "Calibrating thresholds..."
                if best.name == "NLEmbedding" {
                    await runCalibration(provider: nlService)
                } else if best.name == "MiniLM" {
                    await runCalibration(provider: miniLM)
                } else if best.name == "BGE-Large" {
                    await runCalibration(provider: bgeLarge)
                } else {
                    await runCalibration(provider: mxbai)
                }
            }

            // Generate recommendation
            generateRecommendation()

        } catch {
            errorMessage = "Error: \(error.localizedDescription)"
        }

        isLoading = false
    }

    private func runComparison(provider: any EmbeddingProvider, name: String, dimensions: Int) async throws -> ProviderResult {
        var scores: [(label: String, similarity: Float, isMatch: Bool)] = []
        var highSims: [Float] = []
        var lowSims: [Float] = []

        for pair in customPairs where !pair.text1.isEmpty && !pair.text2.isEmpty {
            let similarity = try await provider.similarity(between: pair.text1, and: pair.text2)
            scores.append((pair.label.isEmpty ? "Pair" : pair.label, similarity, pair.isMatch))

            if pair.isMatch {
                highSims.append(similarity)
            } else {
                lowSims.append(similarity)
            }
        }

        let avgHigh = highSims.isEmpty ? 0 : highSims.reduce(0, +) / Float(highSims.count)
        let avgLow = lowSims.isEmpty ? 0 : lowSims.reduce(0, +) / Float(lowSims.count)
        let threshold: Float = 0.5
        let correct = scores.filter { score in
            (score.isMatch && score.similarity >= threshold) || (!score.isMatch && score.similarity < threshold)
        }.count
        let accuracy = scores.isEmpty ? 0 : Float(correct) / Float(scores.count)

        return ProviderResult(
            name: name,
            dimensions: dimensions,
            avgHighSimilarity: avgHigh,
            avgLowSimilarity: avgLow,
            discriminationGap: avgHigh - avgLow,
            accuracy: accuracy,
            scores: scores
        )
    }

    private func runCalibration(provider: any EmbeddingProvider) async {
        let calibrator = ThresholdCalibrator()
        let labeledPairs = customPairs.compactMap { pair -> LabeledPair? in
            guard !pair.text1.isEmpty && !pair.text2.isEmpty else { return nil }
            return LabeledPair(pair.text1, pair.text2, isMatch: pair.isMatch, label: pair.label)
        }

        guard !labeledPairs.isEmpty else { return }

        do {
            calibrationResult = try await calibrator.calibrate(provider: provider, examples: labeledPairs)
        } catch {
            // Calibration failed, continue without it
        }
    }

    private func generateRecommendation() {
        guard !comparisonResults.isEmpty else { return }

        let best = comparisonResults.max(by: { $0.discriminationGap < $1.discriminationGap })!
        let worst = comparisonResults.min(by: { $0.discriminationGap < $1.discriminationGap })!

        // Determine if reranker is needed
        let needsReranker = best.discriminationGap < 0.3 || best.accuracy < 0.8
        let rerankerReason: String
        if best.discriminationGap < 0.2 {
            rerankerReason = "Very low discrimination gap (\(String(format: "%.2f", best.discriminationGap))) suggests embeddings alone may not distinguish your content well. A reranker can provide 2-stage filtering for better precision."
        } else if best.accuracy < 0.75 {
            rerankerReason = "Accuracy below 75% indicates many misclassifications. A cross-encoder reranker processes query+document together for more accurate relevance scoring."
        } else if best.discriminationGap < 0.3 {
            rerankerReason = "Moderate discrimination gap. Consider a reranker if precision is critical (e.g., RAG, customer support)."
        } else {
            rerankerReason = "Good discrimination suggests embeddings are sufficient for most use cases. Add a reranker only if you need maximum precision."
        }

        // Threshold suggestion
        let suggestedThreshold = calibrationResult?.bestF1Threshold ?? 0.5
        let thresholdReason: String
        if let cal = calibrationResult, let metrics = cal.metrics(at: suggestedThreshold) {
            thresholdReason = "Threshold \(String(format: "%.2f", suggestedThreshold)) achieves \(String(format: "%.0f%%", metrics.precision * 100)) precision and \(String(format: "%.0f%%", metrics.recall * 100)) recall on your data."
        } else {
            thresholdReason = "Default threshold 0.5. Add more labeled examples for better calibration."
        }

        // Best provider reasoning
        let reasoning: String
        if comparisonResults.count == 1 {
            reasoning = "Only NLEmbedding tested. Run full comparison with GPU to evaluate transformer models."
        } else {
            let gapDiff = best.discriminationGap - worst.discriminationGap
            if gapDiff < 0.05 {
                reasoning = "All models perform similarly on your data. Choose based on other factors: NLEmbedding (no downloads, instant), MiniLM (fast, small), BGE/mxbai (highest quality)."
            } else {
                reasoning = "\(best.name) achieves \(String(format: "%.0f%%", (best.discriminationGap / worst.discriminationGap - 1) * 100)) better discrimination than \(worst.name) on your content."
            }
        }

        recommendation = Recommendation(
            bestProvider: best.name,
            reasoning: reasoning,
            needsReranker: needsReranker,
            rerankerReason: rerankerReason,
            suggestedThreshold: suggestedThreshold,
            thresholdReason: thresholdReason
        )
    }
}

// MARK: - EmbeddingComparisonView

/// Interactive SwiftUI view for comparing embedding providers on custom text data.
///
/// `EmbeddingComparisonView` provides a complete UI for:
/// - Entering custom text pairs for testing
/// - Loading preset pair collections
/// - Running comparisons against multiple embedding providers
/// - Visualizing results with charts and recommendations
///
/// ## Overview
///
/// Use this view to determine which embedding model works best for your
/// specific content type. The view guides users through the comparison
/// workflow and provides actionable recommendations.
///
/// ## Usage
///
/// ```swift
/// struct ContentView: View {
///     var body: some View {
///         EmbeddingComparisonView()
///     }
/// }
/// ```
///
/// ## Features
///
/// - **Text Pair Editor**: Add/remove pairs, mark as match/non-match
/// - **Preset Loading**: Quick-start with curated domain-specific pairs
/// - **Quick Test**: Fast comparison using NLEmbedding (no GPU)
/// - **Full Comparison**: Complete test with all transformer models
/// - **Visual Charts**: Discrimination gaps, distributions, scatter plots
/// - **Recommendations**: Best provider, reranker guidance, threshold suggestions
///
/// ## Topics
///
/// ### Creating the View
/// - ``init()``
public struct EmbeddingComparisonView: View {

    /// The view model managing comparison state.
    @State private var viewModel = EmbeddingComparisonViewModel()

    /// Currently selected preset (for menu state).
    @State private var selectedPreset: EmbeddingComparisonViewModel.PresetPairs?

    /// Creates a new embedding comparison view.
    public init() {}

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header
                    headerSection

                    Divider()

                    // Input Section
                    inputSection

                    Divider()

                    // Action Buttons
                    actionSection

                    // Results Section
                    if !viewModel.comparisonResults.isEmpty {
                        Divider()
                        resultsSection
                    }

                    // Recommendation Section
                    if let recommendation = viewModel.recommendation {
                        Divider()
                        recommendationSection(recommendation)
                    }

                    // Error Display
                    if let error = viewModel.errorMessage {
                        errorSection(error)
                    }
                }
                .padding()
            }
            .navigationTitle("Embedding Comparison")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.large)
            #endif
        }
        .overlay {
            if viewModel.isLoading {
                loadingOverlay
            }
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Test Embedding Models")
                .font(.headline)

            Text("Enter text pairs from your domain to find which embedding model works best for your content. Mark pairs as matching (similar) or non-matching (unrelated).")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Text Pairs")
                    .font(.headline)

                Spacer()

                Menu("Load Preset") {
                    ForEach(EmbeddingComparisonViewModel.PresetPairs.allCases, id: \.self) { preset in
                        Button(preset.rawValue) {
                            viewModel.loadPresetPairs(preset)
                        }
                    }
                }
                .menuStyle(.borderlessButton)

                Button(action: viewModel.addPair) {
                    Label("Add Pair", systemImage: "plus.circle")
                }
            }

            ForEach(Array(viewModel.customPairs.enumerated()), id: \.element.id) { index, pair in
                TextPairEditor(
                    pair: Binding(
                        get: { viewModel.customPairs[index] },
                        set: { viewModel.customPairs[index] = $0 }
                    ),
                    onDelete: { viewModel.removePair(at: index) }
                )
            }
        }
    }

    private var actionSection: some View {
        VStack(spacing: 12) {
            HStack(spacing: 16) {
                Button(action: { Task { await viewModel.runNLEmbeddingComparison() } }) {
                    Label("Quick Test (NLEmbedding)", systemImage: "bolt")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isLoading)

                if #available(macOS 14, iOS 17, *) {
                    Button(action: { Task { await viewModel.runFullComparison() } }) {
                        Label("Full Comparison (GPU)", systemImage: "cpu")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isLoading)
                }
            }

            Text("Quick Test uses Apple's built-in embeddings. Full Comparison requires GPU and may download models (~500MB total).")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Results")
                .font(.headline)

            // Results Table
            ResultsTable(results: viewModel.comparisonResults)

            // Visual Charts
            chartsSection
        }
    }

    @ViewBuilder
    private var chartsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Visual Analysis")
                .font(.subheadline.bold())
                .padding(.top, 8)

            // Row 1: Gap and Distribution side by side on larger screens
            #if os(macOS)
            HStack(alignment: .top, spacing: 16) {
                DiscriminationGapChart(results: viewModel.comparisonResults)
                SimilarityDistributionChart(results: viewModel.comparisonResults)
            }
            #else
            DiscriminationGapChart(results: viewModel.comparisonResults)
            SimilarityDistributionChart(results: viewModel.comparisonResults)
            #endif

            // Scatter plot showing individual scores
            ScoresScatterChart(results: viewModel.comparisonResults)

            // Accuracy comparison
            AccuracyComparisonChart(results: viewModel.comparisonResults)

            // Multi-metric comparison
            if viewModel.comparisonResults.count > 1 {
                ProviderMetricsChart(results: viewModel.comparisonResults)
            }

            // Threshold calibration curves
            if let calibration = viewModel.calibrationResult {
                ThresholdCalibrationChart(result: calibration)
            }
        }
    }

    private func recommendationSection(_ recommendation: EmbeddingComparisonViewModel.Recommendation) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Recommendations")
                .font(.headline)

            // Best Provider
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Best Provider: \(recommendation.bestProvider)", systemImage: "star.fill")
                        .font(.subheadline.bold())
                        .foregroundStyle(.yellow)

                    Text(recommendation.reasoning)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            // Reranker Recommendation
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Label(
                        recommendation.needsReranker ? "Consider a Reranker" : "Reranker Optional",
                        systemImage: recommendation.needsReranker ? "exclamationmark.triangle" : "checkmark.circle"
                    )
                    .font(.subheadline.bold())
                    .foregroundStyle(recommendation.needsReranker ? .orange : .green)

                    Text(recommendation.rerankerReason)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    if recommendation.needsReranker {
                        Text("Use `EmbeddingReranker` or `MLXCrossEncoderReranker` as a second stage after initial retrieval.")
                            .font(.caption)
                            .padding(.top, 4)
                    }
                }
            }

            // Threshold Recommendation
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Suggested Threshold: \(String(format: "%.2f", recommendation.suggestedThreshold))", systemImage: "slider.horizontal.3")
                        .font(.subheadline.bold())

                    Text(recommendation.thresholdReason)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func errorSection(_ error: String) -> some View {
        GroupBox {
            Label(error, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
        }
    }

    private var loadingOverlay: some View {
        ZStack {
            Color.black.opacity(0.3)
                .ignoresSafeArea()

            VStack(spacing: 16) {
                ProgressView()
                    .scaleEffect(1.5)

                Text(viewModel.loadingMessage)
                    .font(.headline)
            }
            .padding(32)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
    }
}

// MARK: - Supporting Views

struct TextPairEditor: View {
    @Binding var pair: EmbeddingComparisonViewModel.EditableTextPair
    let onDelete: () -> Void

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    TextField("Label", text: $pair.label)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 200)

                    Spacer()

                    Picker("", selection: $pair.isMatch) {
                        Text("Match").tag(true)
                        Text("Non-match").tag(false)
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 150)

                    Button(action: onDelete) {
                        Image(systemName: "trash")
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.borderless)
                }

                TextField("Text 1 (e.g., query)", text: $pair.text1)
                    .textFieldStyle(.roundedBorder)

                TextField("Text 2 (e.g., document)", text: $pair.text2)
                    .textFieldStyle(.roundedBorder)
            }
            .padding(.vertical, 4)
        }
    }
}

struct ResultsTable: View {
    let results: [EmbeddingComparisonViewModel.ProviderResult]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack {
                Text("Provider").frame(width: 100, alignment: .leading)
                Text("Dims").frame(width: 50)
                Text("High").frame(width: 50)
                Text("Low").frame(width: 50)
                Text("Gap").frame(width: 60)
                Text("Acc").frame(width: 50)
                Spacer()
            }
            .font(.caption.bold())
            .foregroundStyle(.secondary)

            Divider()

            // Rows
            let bestGap = results.max(by: { $0.discriminationGap < $1.discriminationGap })?.discriminationGap ?? 0

            ForEach(results) { result in
                HStack {
                    HStack(spacing: 4) {
                        if result.discriminationGap == bestGap {
                            Image(systemName: "star.fill")
                                .foregroundStyle(.yellow)
                                .font(.caption2)
                        }
                        Text(result.name)
                    }
                    .frame(width: 100, alignment: .leading)

                    Text("\(result.dimensions)").frame(width: 50)
                    Text(String(format: "%.2f", result.avgHighSimilarity)).frame(width: 50)
                    Text(String(format: "%.2f", result.avgLowSimilarity)).frame(width: 50)

                    Text(String(format: "%.3f", result.discriminationGap))
                        .frame(width: 60)
                        .foregroundStyle(gapColor(result.discriminationGap))
                        .fontWeight(.semibold)

                    Text(String(format: "%.0f%%", result.accuracy * 100)).frame(width: 50)

                    Spacer()

                    // Gap visualization
                    GapBar(gap: result.discriminationGap, maxGap: 0.6)
                        .frame(width: 80)
                }
                .font(.system(.caption, design: .monospaced))
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }

    private func gapColor(_ gap: Float) -> Color {
        if gap >= 0.4 { return .green }
        if gap >= 0.25 { return .yellow }
        return .red
    }
}

struct GapBar: View {
    let gap: Float
    let maxGap: Float

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.secondary.opacity(0.2))

                RoundedRectangle(cornerRadius: 2)
                    .fill(barColor)
                    .frame(width: geo.size.width * CGFloat(min(gap / maxGap, 1.0)))
            }
        }
        .frame(height: 8)
    }

    private var barColor: Color {
        if gap >= 0.4 { return .green }
        if gap >= 0.25 { return .yellow }
        return .red
    }
}


// MARK: - Previews

#if !SPM_BUILD
#Preview("Empty State") {
    EmbeddingComparisonView()
}

#Preview("With Results") {
    let view = EmbeddingComparisonView()
    // Results would be populated after running comparison
    return view
}
#endif
