//
//  EmbeddingComparisonPreviews.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/5/26.
//

import SwiftUI

// MARK: - PreviewViewModels

/// Factory for creating pre-configured view models for previews and testing.
///
/// Use `PreviewViewModels` to create view models with realistic mock data
/// for SwiftUI previews, unit tests, or demonstrations.
///
/// ## Overview
///
/// Each factory method creates a view model populated with appropriate
/// test data for a specific scenario:
///
/// - ``goodDiscrimination()``: Models successfully distinguish content
/// - ``poorDiscrimination()``: Models struggle with ambiguous terms
/// - ``scientificDomain()``: Technical/scientific content testing
/// - ``loading()``: Shows loading state
/// - ``withError()``: Shows error state
///
/// ## Example
///
/// ```swift
/// #Preview("Good Results") {
///     let viewModel = PreviewViewModels.goodDiscrimination()
///     PreviewComparisonView(viewModel: viewModel)
/// }
/// ```
@MainActor
public enum PreviewViewModels {

    /// Creates a view model showing good discrimination across providers.
    ///
    /// MiniLM achieves the highest discrimination gap (0.59) with perfect
    /// accuracy. Use this to preview the "success" state.
    public static func goodDiscrimination() -> EmbeddingComparisonViewModel {
        let vm = EmbeddingComparisonViewModel()

        vm.customPairs = [
            .init(text1: "How do I reset my password?", text2: "I forgot my login credentials", isMatch: true, label: "Password help"),
            .init(text1: "How do I reset my password?", text2: "The weather is nice today", isMatch: false, label: "Password vs weather"),
            .init(text1: "Track my order status", text2: "Where is my package?", isMatch: true, label: "Order tracking"),
            .init(text1: "Track my order status", text2: "How do I return an item?", isMatch: false, label: "Track vs return"),
        ]

        vm.comparisonResults = [
            .init(
                name: "NLEmbedding",
                dimensions: 512,
                avgHighSimilarity: 0.72,
                avgLowSimilarity: 0.35,
                discriminationGap: 0.37,
                accuracy: 0.875,
                scores: [
                    ("Password help", 0.78, true),
                    ("Password vs weather", 0.28, false),
                    ("Order tracking", 0.66, true),
                    ("Track vs return", 0.42, false),
                ]
            ),
            .init(
                name: "MiniLM",
                dimensions: 384,
                avgHighSimilarity: 0.81,
                avgLowSimilarity: 0.22,
                discriminationGap: 0.59,
                accuracy: 1.0,
                scores: [
                    ("Password help", 0.85, true),
                    ("Password vs weather", 0.15, false),
                    ("Order tracking", 0.77, true),
                    ("Track vs return", 0.29, false),
                ]
            ),
            .init(
                name: "BGE-Large",
                dimensions: 1024,
                avgHighSimilarity: 0.78,
                avgLowSimilarity: 0.31,
                discriminationGap: 0.47,
                accuracy: 1.0,
                scores: [
                    ("Password help", 0.82, true),
                    ("Password vs weather", 0.25, false),
                    ("Order tracking", 0.74, true),
                    ("Track vs return", 0.37, false),
                ]
            ),
        ]

        vm.recommendation = .init(
            bestProvider: "MiniLM",
            reasoning: "MiniLM achieves 59% better discrimination than NLEmbedding on your content, with perfect accuracy.",
            needsReranker: false,
            rerankerReason: "Good discrimination (0.59 gap) suggests embeddings are sufficient. Add a reranker only if you need maximum precision.",
            suggestedThreshold: 0.55,
            thresholdReason: "Threshold 0.55 achieves 100% precision and 100% recall on your data."
        )

        return vm
    }

    /// Creates a view model showing poor discrimination where a reranker is recommended.
    ///
    /// Models struggle with ambiguous terms like "bank" and "apple".
    /// The recommendation suggests using a reranker for better precision.
    public static func poorDiscrimination() -> EmbeddingComparisonViewModel {
        let vm = EmbeddingComparisonViewModel()

        vm.customPairs = [
            .init(text1: "bank account", text2: "financial institution", isMatch: true, label: "Bank finance"),
            .init(text1: "bank account", text2: "river bank erosion", isMatch: false, label: "Bank ambiguity"),
            .init(text1: "apple product", text2: "macbook laptop", isMatch: true, label: "Apple tech"),
            .init(text1: "apple product", text2: "fruit orchard harvest", isMatch: false, label: "Apple ambiguity"),
        ]

        vm.comparisonResults = [
            .init(
                name: "NLEmbedding",
                dimensions: 512,
                avgHighSimilarity: 0.58,
                avgLowSimilarity: 0.45,
                discriminationGap: 0.13,
                accuracy: 0.50,
                scores: [
                    ("Bank finance", 0.62, true),
                    ("Bank ambiguity", 0.51, false),
                    ("Apple tech", 0.54, true),
                    ("Apple ambiguity", 0.39, false),
                ]
            ),
            .init(
                name: "MiniLM",
                dimensions: 384,
                avgHighSimilarity: 0.65,
                avgLowSimilarity: 0.48,
                discriminationGap: 0.17,
                accuracy: 0.625,
                scores: [
                    ("Bank finance", 0.68, true),
                    ("Bank ambiguity", 0.55, false),
                    ("Apple tech", 0.62, true),
                    ("Apple ambiguity", 0.41, false),
                ]
            ),
        ]

        vm.recommendation = .init(
            bestProvider: "MiniLM",
            reasoning: "MiniLM performs slightly better, but all models struggle with ambiguous terms in your domain.",
            needsReranker: true,
            rerankerReason: "Very low discrimination gap (0.17) suggests embeddings alone may not distinguish your content well. A reranker can provide 2-stage filtering for better precision.",
            suggestedThreshold: 0.58,
            thresholdReason: "Threshold 0.58 achieves best F1, but consider using a reranker for ambiguous cases."
        )

        return vm
    }

    /// Creates a view model showing scientific/technical domain content.
    ///
    /// Tests biology, neuroscience, and cellular biology concepts.
    /// BGE-Large achieves best results on this specialized content.
    public static func scientificDomain() -> EmbeddingComparisonViewModel {
        let vm = EmbeddingComparisonViewModel()

        vm.customPairs = [
            .init(text1: "DNA replication", text2: "genetic material copying", isMatch: true, label: "Genetics"),
            .init(text1: "DNA replication", text2: "protein synthesis", isMatch: false, label: "DNA vs protein"),
            .init(text1: "mitochondrial function", text2: "cellular energy production", isMatch: true, label: "Cell energy"),
            .init(text1: "mitochondrial function", text2: "cell membrane transport", isMatch: false, label: "Mito vs membrane"),
            .init(text1: "neural plasticity", text2: "brain adaptation and learning", isMatch: true, label: "Neuro"),
            .init(text1: "neural plasticity", text2: "muscle tissue regeneration", isMatch: false, label: "Brain vs muscle"),
        ]

        vm.comparisonResults = [
            .init(
                name: "NLEmbedding",
                dimensions: 512,
                avgHighSimilarity: 0.68,
                avgLowSimilarity: 0.42,
                discriminationGap: 0.26,
                accuracy: 0.83,
                scores: [
                    ("Genetics", 0.71, true),
                    ("DNA vs protein", 0.55, false),
                    ("Cell energy", 0.65, true),
                    ("Mito vs membrane", 0.38, false),
                    ("Neuro", 0.68, true),
                    ("Brain vs muscle", 0.33, false),
                ]
            ),
            .init(
                name: "BGE-Large",
                dimensions: 1024,
                avgHighSimilarity: 0.76,
                avgLowSimilarity: 0.29,
                discriminationGap: 0.47,
                accuracy: 1.0,
                scores: [
                    ("Genetics", 0.82, true),
                    ("DNA vs protein", 0.38, false),
                    ("Cell energy", 0.71, true),
                    ("Mito vs membrane", 0.25, false),
                    ("Neuro", 0.75, true),
                    ("Brain vs muscle", 0.24, false),
                ]
            ),
        ]

        vm.recommendation = .init(
            bestProvider: "BGE-Large",
            reasoning: "BGE-Large excels at technical/scientific content with 81% better discrimination than NLEmbedding.",
            needsReranker: false,
            rerankerReason: "Good discrimination (0.47 gap) with perfect accuracy. Reranker optional but could help with edge cases in related scientific subfields.",
            suggestedThreshold: 0.52,
            thresholdReason: "Threshold 0.52 cleanly separates related from unrelated scientific concepts."
        )

        return vm
    }

    /// Creates a view model in the loading state.
    ///
    /// Shows a loading overlay with a progress message.
    public static func loading() -> EmbeddingComparisonViewModel {
        let vm = EmbeddingComparisonViewModel()
        vm.isLoading = true
        vm.loadingMessage = "Loading BGE-Large (1024d)..."
        return vm
    }

    /// Creates a view model showing an error state.
    ///
    /// Displays a common MLX error that occurs when running
    /// outside of Xcode without GPU access.
    public static func withError() -> EmbeddingComparisonViewModel {
        let vm = EmbeddingComparisonViewModel()
        vm.errorMessage = "MLX error: Failed to load the default metallib. Run from Xcode with GPU access."
        return vm
    }
}

// MARK: - PreviewComparisonView

/// A simplified view for previewing comparison results with pre-populated state.
///
/// Use this view with ``PreviewViewModels`` to create SwiftUI previews
/// that display various states of the comparison UI.
///
/// ## Example
///
/// ```swift
/// #Preview("Good Results") {
///     PreviewComparisonView(viewModel: PreviewViewModels.goodDiscrimination())
/// }
/// ```
public struct PreviewComparisonView: View {

    /// The view model containing comparison data to display.
    public let viewModel: EmbeddingComparisonViewModel

    /// Creates a preview comparison view with the specified view model.
    ///
    /// - Parameter viewModel: A pre-configured view model from ``PreviewViewModels``
    public init(viewModel: EmbeddingComparisonViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    headerSection
                    Divider()
                    inputSection
                    Divider()
                    actionSection

                    if !viewModel.comparisonResults.isEmpty {
                        Divider()
                        resultsSection
                    }

                    if let recommendation = viewModel.recommendation {
                        Divider()
                        recommendationSection(recommendation)
                    }

                    if let error = viewModel.errorMessage {
                        errorSection(error)
                    }
                }
                .padding()
            }
            .navigationTitle("Embedding Comparison")
        }
        .overlay {
            if viewModel.isLoading {
                loadingOverlay
            }
        }
    }

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Test Embedding Models")
                .font(.headline)

            Text("Enter text pairs from your domain to find which embedding model works best for your content.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Text Pairs (\(viewModel.customPairs.count))")
                    .font(.headline)
                Spacer()
            }

            ForEach(viewModel.customPairs) { pair in
                GroupBox {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(pair.label.isEmpty ? "Pair" : pair.label)
                                .font(.caption.bold())
                            Spacer()
                            Text(pair.isMatch ? "Match" : "Non-match")
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2)
                                .background(pair.isMatch ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                                .clipShape(Capsule())
                        }
                        Text(pair.text1)
                            .font(.caption)
                            .lineLimit(1)
                        Text(pair.text2)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
        }
    }

    private var actionSection: some View {
        HStack(spacing: 16) {
            Button(action: {}) {
                Label("Quick Test", systemImage: "bolt")
            }
            .buttonStyle(.bordered)

            Button(action: {}) {
                Label("Full Comparison", systemImage: "cpu")
            }
            .buttonStyle(.borderedProminent)
        }
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Results")
                .font(.headline)

            ResultsTable(results: viewModel.comparisonResults)
        }
    }

    private func recommendationSection(_ recommendation: EmbeddingComparisonViewModel.Recommendation) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Recommendations")
                .font(.headline)

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
                }
            }

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

// MARK: - Previews

#if !SPM_BUILD
#Preview("Good Discrimination") {
    PreviewComparisonView(viewModel: PreviewViewModels.goodDiscrimination())
}

#Preview("Poor Discrimination (Reranker Needed)") {
    PreviewComparisonView(viewModel: PreviewViewModels.poorDiscrimination())
}

#Preview("Scientific Domain") {
    PreviewComparisonView(viewModel: PreviewViewModels.scientificDomain())
}

#Preview("Loading State") {
    PreviewComparisonView(viewModel: PreviewViewModels.loading())
}

#Preview("Error State") {
    PreviewComparisonView(viewModel: PreviewViewModels.withError())
}
#endif
