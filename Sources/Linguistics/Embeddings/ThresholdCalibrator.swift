//
//  ThresholdCalibrator.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/5/26.
//

import Foundation

// MARK: - LabeledPair

/// A labeled text pair for threshold calibration.
///
/// Use labeled pairs to train the calibrator on your specific domain.
/// The calibrator will find the optimal threshold that maximizes
/// classification performance on your examples.
///
/// ## Example
///
/// ```swift
/// let pairs = [
///     LabeledPair("reset password", "forgot login", isMatch: true),
///     LabeledPair("reset password", "order status", isMatch: false),
/// ]
/// ```
public struct LabeledPair: Sendable {

    /// First text (e.g., query).
    public let text1: String

    /// Second text (e.g., document).
    public let text2: String

    /// Whether these texts should be considered a match/similar.
    public let isMatch: Bool

    /// Optional label for reporting and debugging.
    public let label: String?

    /// Creates a new labeled pair.
    ///
    /// - Parameters:
    ///   - text1: First text
    ///   - text2: Second text
    ///   - isMatch: True if texts should match, false otherwise
    ///   - label: Optional descriptive label
    public init(_ text1: String, _ text2: String, isMatch: Bool, label: String? = nil) {
        self.text1 = text1
        self.text2 = text2
        self.isMatch = isMatch
        self.label = label
    }
}

// MARK: - ThresholdMetrics

/// Classification metrics at a specific similarity threshold.
///
/// Use these metrics to understand the precision/recall tradeoff
/// at different threshold values.
public struct ThresholdMetrics: Sendable {

    /// The threshold value tested.
    public let threshold: Float

    /// True positives: matches correctly identified as matches.
    public let truePositives: Int

    /// False positives: non-matches incorrectly identified as matches.
    public let falsePositives: Int

    /// True negatives: non-matches correctly identified as non-matches.
    public let trueNegatives: Int

    /// False negatives: matches incorrectly identified as non-matches.
    public let falseNegatives: Int

    /// Creates metrics with the specified values.
    public init(
        threshold: Float,
        truePositives: Int,
        falsePositives: Int,
        trueNegatives: Int,
        falseNegatives: Int
    ) {
        self.threshold = threshold
        self.truePositives = truePositives
        self.falsePositives = falsePositives
        self.trueNegatives = trueNegatives
        self.falseNegatives = falseNegatives
    }

    /// Precision: TP / (TP + FP).
    ///
    /// Of all predicted matches, what fraction were correct?
    /// High precision means few false positives.
    public var precision: Float {
        let denominator = truePositives + falsePositives
        return denominator > 0 ? Float(truePositives) / Float(denominator) : 0
    }

    /// Recall: TP / (TP + FN).
    ///
    /// Of all actual matches, what fraction were found?
    /// High recall means few false negatives.
    public var recall: Float {
        let denominator = truePositives + falseNegatives
        return denominator > 0 ? Float(truePositives) / Float(denominator) : 0
    }

    /// F1 Score: harmonic mean of precision and recall.
    ///
    /// Balances precision and recall. Use when both matter equally.
    public var f1: Float {
        let p = precision
        let r = recall
        return (p + r) > 0 ? 2 * p * r / (p + r) : 0
    }

    /// Accuracy: (TP + TN) / total.
    ///
    /// Overall fraction of correct predictions.
    public var accuracy: Float {
        let total = truePositives + falsePositives + trueNegatives + falseNegatives
        return total > 0 ? Float(truePositives + trueNegatives) / Float(total) : 0
    }

    /// F-beta score with custom beta.
    ///
    /// - Parameter beta: Weight for recall vs precision.
    ///   - beta > 1: Favors recall (finding all matches)
    ///   - beta < 1: Favors precision (avoiding false matches)
    ///   - beta = 1: Equal weight (same as F1)
    /// - Returns: F-beta score
    public func fBeta(_ beta: Float) -> Float {
        let p = precision
        let r = recall
        let b2 = beta * beta
        let denominator = (b2 * p) + r
        return denominator > 0 ? (1 + b2) * p * r / denominator : 0
    }
}

// MARK: - CalibrationResult

/// Results from threshold calibration.
///
/// Contains metrics at each tested threshold and recommendations
/// for different use cases.
///
/// ## Example
///
/// ```swift
/// let result = try await calibrator.calibrate(provider: service, examples: pairs)
///
/// print("Best balanced threshold: \(result.bestF1Threshold)")
/// print("Best precision threshold: \(result.bestPrecisionThreshold)")
/// print(result.report)
/// ```
public struct CalibrationResult: Sendable {

    /// Metrics computed at each tested threshold.
    public let metricsAtThresholds: [ThresholdMetrics]

    /// Individual similarity scores for each pair.
    public let scores: [(pair: LabeledPair, similarity: Float)]

    /// Creates a calibration result.
    public init(
        metricsAtThresholds: [ThresholdMetrics],
        scores: [(pair: LabeledPair, similarity: Float)]
    ) {
        self.metricsAtThresholds = metricsAtThresholds
        self.scores = scores
    }

    /// Best threshold for maximizing F1 score (balanced precision/recall).
    public var bestF1Threshold: Float {
        metricsAtThresholds.max(by: { $0.f1 < $1.f1 })?.threshold ?? 0.5
    }

    /// Best threshold for maximizing precision (minimizing false positives).
    ///
    /// Use this when false matches are costly (e.g., duplicate detection).
    public var bestPrecisionThreshold: Float {
        let withRecall = metricsAtThresholds.filter { $0.recall > 0 }
        return withRecall.max(by: { $0.precision < $1.precision })?.threshold ?? 0.8
    }

    /// Best threshold for maximizing recall (minimizing false negatives).
    ///
    /// Use this when missing matches is costly (e.g., content discovery).
    public var bestRecallThreshold: Float {
        let withPrecision = metricsAtThresholds.filter { $0.precision > 0 }
        return withPrecision.max(by: { $0.recall < $1.recall })?.threshold ?? 0.3
    }

    /// Gets metrics at a specific threshold.
    ///
    /// - Parameter threshold: The threshold to look up
    /// - Returns: Metrics if threshold was tested, nil otherwise
    public func metrics(at threshold: Float) -> ThresholdMetrics? {
        metricsAtThresholds.first { abs($0.threshold - threshold) < 0.01 }
    }

    /// Formatted report string for display.
    ///
    /// Shows a table of metrics at each threshold with recommendations.
    public var report: String {
        var lines = [String]()
        lines.append("╔══════════════════════════════════════════════════════════════════╗")
        lines.append("║                   THRESHOLD CALIBRATION REPORT                   ║")
        lines.append("╚══════════════════════════════════════════════════════════════════╝")
        lines.append("")
        lines.append("Threshold │ Precision │ Recall │   F1   │ Accuracy")
        lines.append("──────────┼───────────┼────────┼────────┼─────────")

        for m in metricsAtThresholds {
            let marker = abs(m.threshold - bestF1Threshold) < 0.01 ? "★" : " "
            lines.append(String(format: "  %@%.2f   │   %.3f   │ %.3f  │ %.3f  │  %.3f",
                                marker, m.threshold, m.precision, m.recall, m.f1, m.accuracy))
        }

        lines.append("")
        lines.append("★ = Best F1 threshold")
        lines.append("")
        lines.append("Recommendations:")
        lines.append(String(format: "  • Balanced (F1):        %.2f (P=%.2f, R=%.2f)",
                            bestF1Threshold,
                            metrics(at: bestF1Threshold)?.precision ?? 0,
                            metrics(at: bestF1Threshold)?.recall ?? 0))
        lines.append(String(format: "  • High Precision:       %.2f (minimize false positives)",
                            bestPrecisionThreshold))
        lines.append(String(format: "  • High Recall:          %.2f (minimize false negatives)",
                            bestRecallThreshold))

        return lines.joined(separator: "\n")
    }
}

// MARK: - ThresholdCalibrator

/// Utility for calibrating similarity thresholds based on labeled examples.
///
/// Different use cases require different precision/recall tradeoffs. The
/// `ThresholdCalibrator` helps you find the optimal threshold for your
/// specific domain and requirements.
///
/// ## Overview
///
/// - **Duplicate detection**: High precision (avoid false matches)
/// - **Search/discovery**: High recall (don't miss relevant content)
/// - **FAQ matching**: Balanced (wrong answer worse than no answer)
///
/// ## Usage
///
/// ```swift
/// let calibrator = ThresholdCalibrator()
///
/// // Provide labeled examples from your domain
/// let examples = [
///     LabeledPair("reset password", "forgot login", isMatch: true),
///     LabeledPair("reset password", "order status", isMatch: false),
///     // ... more examples
/// ]
///
/// let result = try await calibrator.calibrate(
///     provider: embeddingService,
///     examples: examples
/// )
///
/// print(result.report)
/// print("Use threshold: \(result.bestF1Threshold)")
/// ```
///
/// ## Presets
///
/// Use built-in presets for common use cases:
///
/// ```swift
/// let duplicateCalibrator = ThresholdCalibrator.duplicateDetection
/// let searchCalibrator = ThresholdCalibrator.semanticSearch
/// let discoveryCalibrator = ThresholdCalibrator.contentDiscovery
/// ```
///
/// ## Topics
///
/// ### Creating a Calibrator
/// - ``init(thresholds:)``
/// - ``duplicateDetection``
/// - ``semanticSearch``
/// - ``contentDiscovery``
///
/// ### Calibrating
/// - ``calibrate(provider:examples:)``
/// - ``calibrate(reranker:examples:)``
/// - ``calibrate(scores:)``
public struct ThresholdCalibrator: Sendable {

    /// Thresholds to test during calibration.
    public let testThresholds: [Float]

    /// Creates a calibrator with custom thresholds to test.
    ///
    /// - Parameter thresholds: Array of thresholds to evaluate
    ///   (default: 0.1 to 0.9 by 0.05)
    public init(thresholds: [Float]? = nil) {
        self.testThresholds = thresholds ?? stride(from: 0.1, through: 0.9, by: 0.05).map { Float($0) }
    }

    /// Calibrates thresholds using an embedding provider.
    ///
    /// Computes similarity for all pairs and evaluates classification
    /// metrics at each threshold.
    ///
    /// - Parameters:
    ///   - provider: The embedding provider to calibrate
    ///   - examples: Labeled pairs for calibration
    /// - Returns: Calibration results with recommended thresholds
    /// - Throws: An error if similarity computation fails
    public func calibrate(
        provider: any EmbeddingProvider,
        examples: [LabeledPair]
    ) async throws -> CalibrationResult {
        var scores: [(pair: LabeledPair, similarity: Float)] = []

        for example in examples {
            let similarity = try await provider.similarity(between: example.text1, and: example.text2)
            scores.append((example, similarity))
        }

        return calculateMetrics(scores: scores)
    }

    /// Calibrates thresholds using a reranker.
    ///
    /// - Parameters:
    ///   - reranker: The reranker to calibrate
    ///   - examples: Labeled pairs for calibration
    /// - Returns: Calibration results with recommended thresholds
    /// - Throws: An error if scoring fails
    public func calibrate(
        reranker: any Reranker,
        examples: [LabeledPair]
    ) async throws -> CalibrationResult {
        var scores: [(pair: LabeledPair, similarity: Float)] = []

        for example in examples {
            let score = try await reranker.score(query: example.text1, document: example.text2)
            scores.append((example, score))
        }

        return calculateMetrics(scores: scores)
    }

    /// Calibrates using pre-computed scores.
    ///
    /// Useful for comparing models using the same scores.
    ///
    /// - Parameter scores: Pre-computed similarity scores
    /// - Returns: Calibration results
    public func calibrate(scores: [(pair: LabeledPair, similarity: Float)]) -> CalibrationResult {
        calculateMetrics(scores: scores)
    }

    /// Calculates metrics at all thresholds.
    private func calculateMetrics(scores: [(pair: LabeledPair, similarity: Float)]) -> CalibrationResult {
        var metricsAtThresholds: [ThresholdMetrics] = []

        for threshold in testThresholds {
            var tp = 0, fp = 0, tn = 0, fn = 0

            for (pair, similarity) in scores {
                let predictedMatch = similarity >= threshold
                let actualMatch = pair.isMatch

                if predictedMatch && actualMatch { tp += 1 }
                else if predictedMatch && !actualMatch { fp += 1 }
                else if !predictedMatch && !actualMatch { tn += 1 }
                else { fn += 1 }
            }

            metricsAtThresholds.append(ThresholdMetrics(
                threshold: threshold,
                truePositives: tp,
                falsePositives: fp,
                trueNegatives: tn,
                falseNegatives: fn
            ))
        }

        return CalibrationResult(
            metricsAtThresholds: metricsAtThresholds,
            scores: scores
        )
    }
}

// MARK: - Convenience Extensions

public extension ThresholdCalibrator {

    /// Creates labeled pairs from TextPairs (from ``EmbeddingBenchmark``).
    ///
    /// - Parameter textPairs: Array of benchmark text pairs
    /// - Returns: Array of labeled pairs for calibration
    static func labeledPairs(from textPairs: [TextPair]) -> [LabeledPair] {
        textPairs.map { pair in
            LabeledPair(pair.text1, pair.text2, isMatch: pair.shouldBeSimilar, label: pair.label)
        }
    }
}

// MARK: - Use Case Presets

public extension ThresholdCalibrator {

    /// Preset for duplicate detection (high precision).
    ///
    /// Tests high thresholds (0.7-0.99) where false positives are costly.
    static let duplicateDetection = ThresholdCalibrator(
        thresholds: [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99].map { Float($0) }
    )

    /// Preset for semantic search (balanced).
    ///
    /// Tests medium thresholds (0.3-0.8) for balanced precision/recall.
    static let semanticSearch = ThresholdCalibrator(
        thresholds: stride(from: 0.3, through: 0.8, by: 0.05).map { Float($0) }
    )

    /// Preset for content discovery (high recall).
    ///
    /// Tests low thresholds (0.1-0.6) where missing content is costly.
    static let contentDiscovery = ThresholdCalibrator(
        thresholds: stride(from: 0.1, through: 0.6, by: 0.05).map { Float($0) }
    )
}
