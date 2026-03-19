//
//  EmbeddingCharts.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/5/26.
//

import SwiftUI
import Charts

// MARK: - DiscriminationGapChart

/// Bar chart comparing discrimination gaps across embedding providers.
///
/// Displays a color-coded bar for each provider showing its discrimination gap:
/// - **Green** (> 0.4): Good discrimination
/// - **Yellow** (0.25-0.4): Moderate discrimination
/// - **Red** (< 0.25): Poor discrimination
///
/// ## Example
///
/// ```swift
/// DiscriminationGapChart(results: viewModel.comparisonResults)
/// ```
public struct DiscriminationGapChart: View {

    /// The provider results to display.
    public let results: [EmbeddingComparisonViewModel.ProviderResult]

    /// Creates a discrimination gap chart.
    ///
    /// - Parameter results: Provider comparison results to visualize
    public init(results: [EmbeddingComparisonViewModel.ProviderResult]) {
        self.results = results
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Discrimination Gap by Provider")
                .font(.subheadline.bold())

            Chart(results) { result in
                BarMark(
                    x: .value("Provider", result.name),
                    y: .value("Gap", result.discriminationGap)
                )
                .foregroundStyle(gapGradient(for: result.discriminationGap))
                .annotation(position: .top) {
                    Text(String(format: "%.2f", result.discriminationGap))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .chartYScale(domain: 0...0.7)
            .chartYAxis {
                AxisMarks(position: .leading)
            }
            .frame(height: 200)

            // Legend
            HStack(spacing: 16) {
                LegendItem(color: .red, label: "Poor (<0.25)")
                LegendItem(color: .yellow, label: "Moderate (0.25-0.4)")
                LegendItem(color: .green, label: "Good (>0.4)")
            }
            .font(.caption2)
        }
        .padding()
        .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    private func gapGradient(for gap: Float) -> Color {
        if gap >= 0.4 { return .green }
        if gap >= 0.25 { return .yellow }
        return .red
    }
}

// MARK: - SimilarityDistributionChart

/// Chart showing similarity ranges for matching vs non-matching pairs.
///
/// Displays a range bar for each provider with:
/// - **Top**: Average similarity for matching pairs (green)
/// - **Bottom**: Average similarity for non-matching pairs (blue)
///
/// Larger gaps between top and bottom indicate better discrimination.
///
/// ## Example
///
/// ```swift
/// SimilarityDistributionChart(results: viewModel.comparisonResults)
/// ```
public struct SimilarityDistributionChart: View {

    /// The provider results to display.
    public let results: [EmbeddingComparisonViewModel.ProviderResult]

    /// Creates a similarity distribution chart.
    ///
    /// - Parameter results: Provider comparison results to visualize
    public init(results: [EmbeddingComparisonViewModel.ProviderResult]) {
        self.results = results
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Similarity Distributions")
                .font(.subheadline.bold())

            Chart {
                ForEach(results) { result in
                    // High similarity range (matches)
                    BarMark(
                        x: .value("Provider", result.name),
                        yStart: .value("Low", result.avgLowSimilarity),
                        yEnd: .value("High", result.avgHighSimilarity)
                    )
                    .foregroundStyle(
                        .linearGradient(
                            colors: [.blue.opacity(0.6), .green.opacity(0.8)],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    )
                    .annotation(position: .top) {
                        Text(String(format: "%.2f", result.avgHighSimilarity))
                            .font(.caption2)
                            .foregroundStyle(.green)
                    }
                    .annotation(position: .bottom) {
                        Text(String(format: "%.2f", result.avgLowSimilarity))
                            .font(.caption2)
                            .foregroundStyle(.blue)
                    }
                }
            }
            .chartYScale(domain: 0...1)
            .chartYAxis {
                AxisMarks(position: .leading, values: [0, 0.25, 0.5, 0.75, 1.0])
            }
            .frame(height: 200)

            HStack(spacing: 16) {
                LegendItem(color: .green, label: "Avg Match Similarity")
                LegendItem(color: .blue, label: "Avg Non-match Similarity")
            }
            .font(.caption2)
        }
        .padding()
        .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - ScoresScatterChart

/// Scatter plot showing individual pair scores by provider.
///
/// Each point represents a text pair:
/// - **Green circles**: Matching pairs (should have high similarity)
/// - **Red crosses**: Non-matching pairs (should have low similarity)
///
/// A dashed orange line shows the 0.5 threshold. Points should ideally
/// separate above and below this line based on their match status.
///
/// ## Example
///
/// ```swift
/// ScoresScatterChart(results: viewModel.comparisonResults)
/// ```
public struct ScoresScatterChart: View {

    /// The provider results to display.
    public let results: [EmbeddingComparisonViewModel.ProviderResult]

    /// Creates a scores scatter chart.
    ///
    /// - Parameter results: Provider comparison results to visualize
    public init(results: [EmbeddingComparisonViewModel.ProviderResult]) {
        self.results = results
    }

    private var chartData: [ScorePoint] {
        var points: [ScorePoint] = []
        for (providerIndex, result) in results.enumerated() {
            for (scoreIndex, score) in result.scores.enumerated() {
                points.append(ScorePoint(
                    provider: result.name,
                    providerIndex: providerIndex,
                    label: score.label,
                    similarity: score.similarity,
                    isMatch: score.isMatch,
                    scoreIndex: scoreIndex
                ))
            }
        }
        return points
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Individual Pair Scores")
                .font(.subheadline.bold())

            Chart(chartData) { point in
                PointMark(
                    x: .value("Provider", point.provider),
                    y: .value("Similarity", point.similarity)
                )
                .foregroundStyle(point.isMatch ? .green : .red)
                .symbolSize(100)
                .symbol(point.isMatch ? .circle : .cross)
            }
            .chartYScale(domain: 0...1)
            .chartYAxis {
                AxisMarks(position: .leading, values: [0, 0.25, 0.5, 0.75, 1.0]) { value in
                    AxisGridLine()
                    AxisValueLabel()
                }
            }
            // Threshold reference line
            .chartOverlay { proxy in
                GeometryReader { geo in
                    if let y = proxy.position(forY: 0.5) {
                        Path { path in
                            path.move(to: CGPoint(x: 0, y: y))
                            path.addLine(to: CGPoint(x: geo.size.width, y: y))
                        }
                        .stroke(style: StrokeStyle(lineWidth: 1, dash: [5, 5]))
                        .foregroundStyle(.orange)
                    }
                }
            }
            .frame(height: 200)

            HStack(spacing: 16) {
                LegendItem(color: .green, label: "Match pairs", symbol: "circle.fill")
                LegendItem(color: .red, label: "Non-match pairs", symbol: "xmark")
                LegendItem(color: .orange, label: "0.5 threshold", symbol: "minus")
            }
            .font(.caption2)
        }
        .padding()
        .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    /// A data point for the scatter chart.
    struct ScorePoint: Identifiable {
        let id = UUID()
        let provider: String
        let providerIndex: Int
        let label: String
        let similarity: Float
        let isMatch: Bool
        let scoreIndex: Int
    }
}

// MARK: - ThresholdCalibrationChart

/// Line chart showing precision, recall, and F1 across similarity thresholds.
///
/// Displays three curves:
/// - **Blue**: Precision (avoiding false positives)
/// - **Green**: Recall (finding all matches)
/// - **Purple**: F1 (balanced metric)
///
/// A yellow star marks the best F1 threshold.
///
/// ## Example
///
/// ```swift
/// if let calibration = viewModel.calibrationResult {
///     ThresholdCalibrationChart(result: calibration)
/// }
/// ```
public struct ThresholdCalibrationChart: View {

    /// The calibration result to display.
    public let result: CalibrationResult

    /// Creates a threshold calibration chart.
    ///
    /// - Parameter result: Calibration results containing metrics at each threshold
    public init(result: CalibrationResult) {
        self.result = result
    }

    private var chartData: [ThresholdMetricPoint] {
        result.metricsAtThresholds.flatMap { metrics in
            [
                ThresholdMetricPoint(threshold: metrics.threshold, value: metrics.precision, metric: "Precision"),
                ThresholdMetricPoint(threshold: metrics.threshold, value: metrics.recall, metric: "Recall"),
                ThresholdMetricPoint(threshold: metrics.threshold, value: metrics.f1, metric: "F1"),
            ]
        }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Threshold Calibration Curves")
                .font(.subheadline.bold())

            Chart(chartData) { point in
                LineMark(
                    x: .value("Threshold", point.threshold),
                    y: .value("Value", point.value)
                )
                .foregroundStyle(by: .value("Metric", point.metric))
                .interpolationMethod(.monotone)

                if point.metric == "F1" && abs(point.threshold - result.bestF1Threshold) < 0.01 {
                    PointMark(
                        x: .value("Threshold", point.threshold),
                        y: .value("Value", point.value)
                    )
                    .foregroundStyle(.yellow)
                    .symbolSize(200)
                    .annotation(position: .top) {
                        Text("Best")
                            .font(.caption2)
                            .foregroundStyle(.yellow)
                    }
                }
            }
            .chartXScale(domain: 0...1)
            .chartYScale(domain: 0...1)
            .chartXAxis {
                AxisMarks(values: [0.1, 0.3, 0.5, 0.7, 0.9])
            }
            .chartYAxis {
                AxisMarks(position: .leading, values: [0, 0.25, 0.5, 0.75, 1.0])
            }
            .chartForegroundStyleScale([
                "Precision": .blue,
                "Recall": .green,
                "F1": .purple
            ])
            .chartLegend(position: .bottom)
            .frame(height: 220)

            Text("Best F1 at threshold \(String(format: "%.2f", result.bestF1Threshold))")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    /// A data point for the calibration chart.
    struct ThresholdMetricPoint: Identifiable {
        let id = UUID()
        let threshold: Float
        let value: Float
        let metric: String
    }
}

// MARK: - AccuracyComparisonChart

/// Horizontal bar chart comparing classification accuracy across providers.
///
/// Shows accuracy percentage for each provider with color coding:
/// - **Green** (>= 90%): Excellent accuracy
/// - **Yellow** (75-90%): Good accuracy
/// - **Red** (< 75%): Poor accuracy
///
/// ## Example
///
/// ```swift
/// AccuracyComparisonChart(results: viewModel.comparisonResults)
/// ```
public struct AccuracyComparisonChart: View {

    /// The provider results to display.
    public let results: [EmbeddingComparisonViewModel.ProviderResult]

    /// Creates an accuracy comparison chart.
    ///
    /// - Parameter results: Provider comparison results to visualize
    public init(results: [EmbeddingComparisonViewModel.ProviderResult]) {
        self.results = results
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Accuracy Comparison")
                .font(.subheadline.bold())

            Chart(results) { result in
                BarMark(
                    x: .value("Accuracy", result.accuracy),
                    y: .value("Provider", result.name)
                )
                .foregroundStyle(accuracyColor(result.accuracy))
                .annotation(position: .trailing) {
                    Text(String(format: "%.0f%%", result.accuracy * 100))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .chartXScale(domain: 0...1)
            .chartXAxis {
                AxisMarks(values: [0, 0.25, 0.5, 0.75, 1.0]) { value in
                    AxisGridLine()
                    AxisValueLabel(format: FloatingPointFormatStyle<Double>.Percent())
                }
            }
            .frame(height: CGFloat(results.count * 50 + 20))
        }
        .padding()
        .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    private func accuracyColor(_ accuracy: Float) -> Color {
        if accuracy >= 0.9 { return .green }
        if accuracy >= 0.75 { return .yellow }
        return .red
    }
}

// MARK: - ProviderMetricsChart

/// Grouped bar chart comparing multiple metrics across providers.
///
/// Displays normalized scores for:
/// - **Gap**: Discrimination gap (higher = better separation)
/// - **Accuracy**: Classification accuracy
/// - **High Sim**: Average similarity for matching pairs
/// - **Separation**: Inverse of non-match similarity (higher = better)
///
/// ## Example
///
/// ```swift
/// if viewModel.comparisonResults.count > 1 {
///     ProviderMetricsChart(results: viewModel.comparisonResults)
/// }
/// ```
public struct ProviderMetricsChart: View {

    /// The provider results to display.
    public let results: [EmbeddingComparisonViewModel.ProviderResult]

    /// Creates a provider metrics chart.
    ///
    /// - Parameter results: Provider comparison results to visualize
    public init(results: [EmbeddingComparisonViewModel.ProviderResult]) {
        self.results = results
    }

    private var chartData: [ProviderMetric] {
        results.flatMap { result in
            [
                ProviderMetric(provider: result.name, metric: "Gap", value: result.discriminationGap, maxValue: 0.6),
                ProviderMetric(provider: result.name, metric: "Accuracy", value: result.accuracy, maxValue: 1.0),
                ProviderMetric(provider: result.name, metric: "High Sim", value: result.avgHighSimilarity, maxValue: 1.0),
                ProviderMetric(provider: result.name, metric: "Separation", value: 1.0 - result.avgLowSimilarity, maxValue: 1.0),
            ]
        }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Provider Metrics Comparison")
                .font(.subheadline.bold())

            Chart(chartData) { item in
                BarMark(
                    x: .value("Metric", item.metric),
                    y: .value("Normalized", item.normalizedValue)
                )
                .foregroundStyle(by: .value("Provider", item.provider))
                .position(by: .value("Provider", item.provider))
            }
            .chartYScale(domain: 0...1)
            .chartYAxis {
                AxisMarks(position: .leading, values: [0, 0.5, 1.0])
            }
            .chartLegend(position: .bottom)
            .frame(height: 200)

            Text("Higher is better for all metrics")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    /// A normalized metric data point for the chart.
    struct ProviderMetric: Identifiable {
        let id = UUID()
        let provider: String
        let metric: String
        let value: Float
        let maxValue: Float

        var normalizedValue: Float {
            min(value / maxValue, 1.0)
        }
    }
}

// MARK: - EmbeddingChartsView

/// A comprehensive collection of all embedding comparison charts.
///
/// Displays a complete visual analysis dashboard including:
/// - Discrimination gap comparison
/// - Similarity distributions
/// - Individual score scatter plots
/// - Accuracy comparison
/// - Multi-metric provider comparison
/// - Threshold calibration curves (if available)
///
/// ## Usage
///
/// ```swift
/// EmbeddingChartsView(
///     results: viewModel.comparisonResults,
///     calibration: viewModel.calibrationResult
/// )
/// ```
///
/// ## Layout
///
/// Charts are arranged in rows for optimal viewing:
/// 1. Gap and Distribution side by side
/// 2. Scatter and Accuracy side by side
/// 3. Provider metrics comparison
/// 4. Calibration curves (if calibration data provided)
public struct EmbeddingChartsView: View {

    /// Provider comparison results to visualize.
    public let results: [EmbeddingComparisonViewModel.ProviderResult]

    /// Optional calibration results for threshold curves.
    public let calibration: CalibrationResult?

    /// Creates a comprehensive charts view.
    ///
    /// - Parameters:
    ///   - results: Provider comparison results to visualize
    ///   - calibration: Optional calibration results for threshold curves
    public init(results: [EmbeddingComparisonViewModel.ProviderResult], calibration: CalibrationResult? = nil) {
        self.results = results
        self.calibration = calibration
    }

    public var body: some View {
        VStack(spacing: 20) {
            if !results.isEmpty {
                // Row 1: Gap and Distribution
                HStack(alignment: .top, spacing: 16) {
                    DiscriminationGapChart(results: results)
                    SimilarityDistributionChart(results: results)
                }

                // Row 2: Scatter and Accuracy
                HStack(alignment: .top, spacing: 16) {
                    ScoresScatterChart(results: results)
                    AccuracyComparisonChart(results: results)
                }

                // Row 3: Provider Metrics
                ProviderMetricsChart(results: results)

                // Row 4: Calibration (if available)
                if let calibration = calibration {
                    ThresholdCalibrationChart(result: calibration)
                }
            }
        }
    }
}

// MARK: - LegendItem

/// A reusable legend item for chart legends.
///
/// Displays a colored symbol with a label.
public struct LegendItem: View {

    /// The color for the symbol.
    public let color: Color

    /// The descriptive label text.
    public let label: String

    /// The SF Symbol name (default: "circle.fill").
    public var symbol: String = "circle.fill"

    /// Creates a legend item.
    ///
    /// - Parameters:
    ///   - color: Symbol color
    ///   - label: Descriptive text
    ///   - symbol: SF Symbol name (default: "circle.fill")
    public init(color: Color, label: String, symbol: String = "circle.fill") {
        self.color = color
        self.label = label
        self.symbol = symbol
    }

    public var body: some View {
        HStack(spacing: 4) {
            Image(systemName: symbol)
                .foregroundStyle(color)
                .font(.caption2)
            Text(label)
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Chart Previews

#if !SPM_BUILD
#Preview("Discrimination Gap Chart") {
    DiscriminationGapChart(results: PreviewViewModels.goodDiscrimination().comparisonResults)
        .padding()
}

#Preview("Similarity Distribution") {
    SimilarityDistributionChart(results: PreviewViewModels.goodDiscrimination().comparisonResults)
        .padding()
}

#Preview("Scores Scatter") {
    ScoresScatterChart(results: PreviewViewModels.goodDiscrimination().comparisonResults)
        .padding()
}

#Preview("Accuracy Comparison") {
    AccuracyComparisonChart(results: PreviewViewModels.goodDiscrimination().comparisonResults)
        .padding()
}

#Preview("Provider Metrics") {
    ProviderMetricsChart(results: PreviewViewModels.goodDiscrimination().comparisonResults)
        .padding()
}

#Preview("All Charts - Good Discrimination") {
    ScrollView {
        EmbeddingChartsView(
            results: PreviewViewModels.goodDiscrimination().comparisonResults,
            calibration: nil
        )
        .padding()
    }
}

#Preview("All Charts - Poor Discrimination") {
    ScrollView {
        EmbeddingChartsView(
            results: PreviewViewModels.poorDiscrimination().comparisonResults,
            calibration: nil
        )
        .padding()
    }
}
#endif
