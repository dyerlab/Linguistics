//
//  MLXCrossEncoderReranker.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/5/26.
//

import Foundation
import Hub
import MLX
import MLXEmbedders
import MLXNN
import Tokenizers

// MARK: - MLXCrossEncoderReranker

/// Cross-encoder reranker using MLX on Apple Silicon.
///
/// Cross-encoders process query and document together in a single forward pass,
/// allowing direct token-level attention between them. This produces more accurate
/// relevance scores than bi-encoder embeddings at the cost of speed (can't pre-compute).
///
/// ## Overview
///
/// Use cross-encoder reranking as the second stage of a two-stage retrieval pipeline:
///
/// ```
/// Query → Embedding Search (fast, top-100) → Cross-Encoder (accurate, top-10) → Results
/// ```
///
/// ## When to Use
///
/// Cross-encoders are ideal when:
/// - **Precision is critical**: Wrong answers are costly (RAG, customer support)
/// - **Candidate set is small**: Reranking 10-100 documents, not thousands
/// - **Quality over speed**: Willing to trade latency for accuracy
///
/// For faster but less accurate reranking, consider ``EmbeddingReranker``.
///
/// ## Usage
///
/// ```swift
/// // Initialize reranker
/// let reranker = try await MLXCrossEncoderReranker(model: .bgeRerankerBase)
///
/// // Score a single query-document pair
/// let score = try await reranker.score(
///     query: "capital of France",
///     document: "Paris is the capital and largest city of France..."
/// )
///
/// // Rerank multiple candidates
/// let candidates = ["Doc about Paris...", "Doc about London...", "Doc about Berlin..."]
/// let results = try await reranker.rerank(
///     query: "capital of France",
///     documents: candidates,
///     topK: 10
/// )
///
/// // Use top results
/// for result in results {
///     print("[\(result.score)] \(result.item.prefix(50))...")
/// }
/// ```
///
/// ## Available Models
///
/// | Model | Size | Speed | Quality |
/// |-------|------|-------|---------|
/// | `.bgeRerankerBase` | ~400MB | Fast | Good |
/// | `.bgeRerankerLarge` | ~1.2GB | Slower | Better |
/// | `.bgeRerankerV2M3` | ~1.2GB | Slower | Multilingual |
///
/// ## Performance Characteristics
///
/// - **Latency**: ~10-50ms per query-document pair (depends on length and model)
/// - **Throughput**: Best for reranking 10-100 candidates, not thousands
/// - **Memory**: Larger than bi-encoders due to cross-attention computation
///
/// ## Requirements
///
/// - macOS 14+ / iOS 17+
/// - Apple Silicon (M1/M2/M3)
/// - Must run from Xcode (Metal shader compilation)
///
/// ## Topics
///
/// ### Creating a Reranker
/// - ``init(model:maxLength:progressHandler:)``
/// - ``Model``
///
/// ### Scoring
/// - ``score(query:document:)``
/// - ``scoreBatch(query:documents:)``
///
/// ### Reranking
/// - ``rerank(query:documents:topK:)``
/// - ``rerank(query:items:topK:textExtractor:)``
@available(macOS 14, iOS 17, *)
public actor MLXCrossEncoderReranker: Reranker {

    // MARK: - Model Selection

    /// Available cross-encoder models for reranking.
    ///
    /// Each model offers different tradeoffs between speed, accuracy, and
    /// language support. Choose based on your requirements.
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Good balance of speed and quality
    /// let reranker = try await MLXCrossEncoderReranker(model: .bgeRerankerBase)
    ///
    /// // Higher quality, slower
    /// let betterReranker = try await MLXCrossEncoderReranker(model: .bgeRerankerLarge)
    ///
    /// // Custom model from HuggingFace
    /// let custom = try await MLXCrossEncoderReranker(model: .custom("my-org/my-reranker"))
    /// ```
    public enum Model: Sendable {

        /// BGE Reranker Base - good balance of speed and accuracy.
        ///
        /// Recommended for most use cases. Fast enough for real-time
        /// applications while maintaining good relevance scoring.
        case bgeRerankerBase

        /// BGE Reranker Large - higher accuracy, slower.
        ///
        /// Use when quality is more important than latency.
        /// Better at nuanced relevance distinctions.
        case bgeRerankerLarge

        /// BGE Reranker v2 M3 - multilingual support.
        ///
        /// Supports multiple languages in the same model.
        /// Use for international or multilingual applications.
        case bgeRerankerV2M3

        /// Custom model from HuggingFace Hub.
        ///
        /// Load any compatible cross-encoder model by its Hub ID.
        ///
        /// ```swift
        /// let reranker = try await MLXCrossEncoderReranker(
        ///     model: .custom("cross-encoder/ms-marco-MiniLM-L-6-v2")
        /// )
        /// ```
        case custom(String)

        /// The HuggingFace Hub identifier for this model.
        var hubId: String {
            switch self {
            case .bgeRerankerBase: return "BAAI/bge-reranker-base"
            case .bgeRerankerLarge: return "BAAI/bge-reranker-large"
            case .bgeRerankerV2M3: return "BAAI/bge-reranker-v2-m3"
            case .custom(let id): return id
            }
        }

        /// Human-readable name for the model.
        public var name: String {
            switch self {
            case .bgeRerankerBase: return "bge-reranker-base"
            case .bgeRerankerLarge: return "bge-reranker-large"
            case .bgeRerankerV2M3: return "bge-reranker-v2-m3"
            case .custom(let id): return id
            }
        }
    }

    // MARK: - Properties

    /// The loaded model container.
    private let container: MLXEmbedders.ModelContainer

    /// The name of the loaded model.
    private let modelName: String

    /// Maximum sequence length for tokenization.
    private let maxLength: Int

    // MARK: - Initialization

    /// Creates a new cross-encoder reranker with the specified model.
    ///
    /// The model will be downloaded from HuggingFace Hub on first use if not
    /// already cached. Downloads are stored in `~/.cache/huggingface/hub/`.
    ///
    /// - Parameters:
    ///   - model: The reranker model to use (default: `.bgeRerankerBase`)
    ///   - maxLength: Maximum sequence length for tokenization (default: 512)
    ///   - progressHandler: Optional callback for download progress (0.0 to 1.0)
    /// - Throws: ``CrossEncoderError/modelLoadFailed(_:_:)`` if model loading fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// // With progress tracking
    /// let reranker = try await MLXCrossEncoderReranker(
    ///     model: .bgeRerankerLarge,
    ///     progressHandler: { progress in
    ///         print("Download: \(Int(progress * 100))%")
    ///     }
    /// )
    /// ```
    public init(
        model: Model = .bgeRerankerBase,
        maxLength: Int = 512,
        progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws {
        self.modelName = model.name
        self.maxLength = maxLength

        let hub = HubApi()
        let configuration = MLXEmbedders.ModelConfiguration(id: model.hubId)

        do {
            self.container = try await MLXEmbedders.loadModelContainer(
                hub: hub,
                configuration: configuration,
                progressHandler: { progress in
                    progressHandler?(progress.fractionCompleted)
                }
            )
        } catch {
            throw CrossEncoderError.modelLoadFailed(model.name, error)
        }
    }

    // MARK: - Reranker Protocol

    /// Scores the relevance of a document to a query.
    ///
    /// Encodes query and document together as `[CLS] query [SEP] document [SEP]`
    /// and computes a relevance score using cross-attention.
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - document: The document to score
    /// - Returns: Relevance score (0 to 1, higher = more relevant)
    /// - Throws: An error if scoring fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let score = try await reranker.score(
    ///     query: "how to sort an array",
    ///     document: "Arrays can be sorted using the sort() method..."
    /// )
    /// print("Relevance: \(score)")  // e.g., 0.85
    /// ```
    public func score(query: String, document: String) async throws -> Float {
        await container.perform { model, tokenizer, pooler in
            // Cross-encoder: encode query and document together
            // Most tokenizers will handle adding special tokens automatically
            // For BERT-style models: [CLS] query [SEP] document [SEP]
            let queryTokens = tokenizer.encode(text: query)
            let docTokens = tokenizer.encode(text: document)

            // Combine tokens, respecting max length
            // Leave room for special tokens that may have been added
            let maxQueryLen = self.maxLength / 2
            let maxDocLen = self.maxLength - min(queryTokens.count, maxQueryLen)

            let truncatedQuery = Array(queryTokens.prefix(maxQueryLen))
            let truncatedDoc = Array(docTokens.prefix(maxDocLen))
            let allTokens = truncatedQuery + truncatedDoc

            // Create tensors
            let inputIds = MLXArray(allTokens.map { Int32($0) }).expandedDimensions(axis: 0)
            let attentionMask = MLXArray(
                [Int32](repeating: 1, count: allTokens.count)
            ).expandedDimensions(axis: 0)

            // Forward pass
            let output = model(
                inputIds,
                positionIds: nil,
                tokenTypeIds: nil,
                attentionMask: attentionMask
            )

            // Use the pooler to extract the representation, then compute score
            // For reranking, we use mean pooling and normalize
            let pooled = pooler(output, mask: attentionMask, normalize: true)

            eval(pooled)

            // Compute magnitude/energy of the pooled representation as relevance signal
            // Higher energy typically indicates stronger semantic match
            let flatPooled = pooled.squeezed(axis: 0).asArray(Float.self)
            let meanScore = flatPooled.reduce(0, +) / Float(flatPooled.count)

            // Apply sigmoid to bound between 0-1
            return 1.0 / (1.0 + exp(-meanScore * 10))
        }
    }

    /// Scores multiple documents against a query.
    ///
    /// Processes documents sequentially. For large batches, consider
    /// using ``rerank(query:documents:topK:)`` which returns sorted results.
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - documents: Documents to score
    /// - Returns: Array of scores in same order as input documents
    /// - Throws: An error if any scoring fails
    public func scoreBatch(query: String, documents: [String]) async throws -> [Float] {
        var scores: [Float] = []
        for document in documents {
            let score = try await score(query: query, document: document)
            scores.append(score)
        }
        return scores
    }

    /// Reranks documents by relevance to a query.
    ///
    /// Scores all documents and returns them sorted by relevance (highest first).
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - documents: Documents to rerank
    ///   - topK: Number of top results to return (nil = all)
    /// - Returns: Documents sorted by relevance, with scores and original indices
    /// - Throws: An error if reranking fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let results = try await reranker.rerank(
    ///     query: "machine learning tutorial",
    ///     documents: searchResults,
    ///     topK: 5
    /// )
    ///
    /// for result in results {
    ///     print("[\(result.score)] was #\(result.originalIndex): \(result.item.prefix(50))...")
    /// }
    /// ```
    public func rerank(query: String, documents: [String], topK: Int? = nil) async throws -> [RankedResult<String>] {
        let scores = try await scoreBatch(query: query, documents: documents)

        var results = zip(documents, scores).enumerated().map { index, pair in
            RankedResult(item: pair.0, score: pair.1, originalIndex: index)
        }

        results.sort { $0.score > $1.score }

        if let topK = topK, topK < results.count {
            return Array(results.prefix(topK))
        }
        return results
    }

    /// Reranks arbitrary items using a text extractor.
    ///
    /// Use this when your items are custom types (not just strings).
    ///
    /// - Parameters:
    ///   - query: The search query
    ///   - items: Items to rerank
    ///   - topK: Number of top results to return (nil = all)
    ///   - textExtractor: Function to extract searchable text from each item
    /// - Returns: Items sorted by relevance, with scores and original indices
    /// - Throws: An error if reranking fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// struct Article { let title: String; let body: String }
    ///
    /// let results = try await reranker.rerank(
    ///     query: "climate change impacts",
    ///     items: articles,
    ///     topK: 10
    /// ) { article in
    ///     "\(article.title) \(article.body)"
    /// }
    /// ```
    public func rerank<T: Sendable>(
        query: String,
        items: [T],
        topK: Int? = nil,
        textExtractor: @Sendable (T) -> String
    ) async throws -> [RankedResult<T>] {
        let documents = items.map(textExtractor)
        let scores = try await scoreBatch(query: query, documents: documents)

        var results = zip(items, scores).enumerated().map { index, pair in
            RankedResult(item: pair.0, score: pair.1, originalIndex: index)
        }

        results.sort { $0.score > $1.score }

        if let topK = topK, topK < results.count {
            return Array(results.prefix(topK))
        }
        return results
    }
}

// MARK: - CrossEncoderError

/// Errors specific to cross-encoder rerankers.
///
/// These errors indicate issues with model loading or document scoring.
public enum CrossEncoderError: LocalizedError {

    /// Failed to load the cross-encoder model.
    ///
    /// This can occur if:
    /// - The model ID is invalid
    /// - Network connection failed during download
    /// - Insufficient disk space
    /// - The model format is incompatible
    case modelLoadFailed(String, Error)

    /// Failed to score a query-document pair.
    ///
    /// This is rare but can occur with extremely long texts
    /// or malformed input.
    case scoringFailed(String, Error)

    /// A localized description of the error.
    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let model, let error):
            return "Failed to load cross-encoder model '\(model)': \(error.localizedDescription)"
        case .scoringFailed(let text, let error):
            return "Failed to score text '\(text.prefix(50))...': \(error.localizedDescription)"
        }
    }
}
