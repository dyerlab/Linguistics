//
//  MLXEmbeddingService.swift
//  Linguistics
//
//  Created by Rodney Dyer on 2/4/26.
//

import Foundation
import Hub
import MLX
import MLXEmbedders
import Tokenizers

// MARK: - MLXEmbeddingService

/// Text embedding service using MLX on Apple Silicon.
///
/// `MLXEmbeddingService` provides high-quality sentence and paragraph embeddings
/// using transformer models running on Apple's Metal GPU via MLX. It supports
/// automatic model downloading from HuggingFace Hub with progress tracking.
///
/// ## Overview
///
/// This service offers state-of-the-art embedding quality using models like
/// BGE, MiniLM, and mxbai. Models are downloaded on first use and cached locally.
///
/// ## Usage
///
/// ```swift
/// // Initialize with default model (mxbai-embed-large)
/// let service = try await MLXEmbeddingService()
///
/// // Or choose a specific model
/// let service = try await MLXEmbeddingService(
///     model: .miniLM,
///     progressHandler: { progress in
///         print("Downloading: \(Int(progress * 100))%")
///     }
/// )
///
/// // Generate embeddings
/// let embedding = try await service.embed("Hello, world!")
/// print("Dimensions: \(embedding.count)")  // 384 for MiniLM
/// ```
///
/// ## Available Models
///
/// | Model | Dimensions | Size | Best For |
/// |-------|------------|------|----------|
/// | `.miniLM` | 384 | ~90MB | Fast, general purpose |
/// | `.bgeBase` | 768 | ~400MB | Balanced quality/speed |
/// | `.bgeLarge` | 1024 | ~1.2GB | High quality retrieval |
/// | `.mxbaiEmbedLarge` | 1024 | ~1.2GB | Best overall quality |
///
/// ## Performance
///
/// - **GPU Accelerated**: Runs on Apple's Metal GPU for fast inference
/// - **Batching**: Efficient batch processing for multiple texts
/// - **Caching**: Models cached in `~/.cache/huggingface/hub/`
///
/// ## Requirements
///
/// - macOS 14+ / iOS 17+
/// - Apple Silicon (M1/M2/M3)
/// - Must run from Xcode (Metal shader compilation)
///
/// ## Topics
///
/// ### Creating a Service
/// - ``init(model:cacheDirectory:progressHandler:)``
/// - ``Model``
///
/// ### Protocol Conformance
/// - ``EmbeddingProvider``
@available(macOS 14, iOS 17, *)
public actor MLXEmbeddingService: EmbeddingProvider {

    // MARK: - Model Selection

    /// Predefined embedding models available for use.
    ///
    /// Each model offers different tradeoffs between speed, quality, and memory usage.
    /// Choose based on your requirements:
    ///
    /// - For speed: `.miniLM` (384d, ~90MB)
    /// - For quality: `.mxbaiEmbedLarge` (1024d, ~1.2GB)
    /// - For retrieval: `.bgeLarge` (1024d, optimized for search)
    public enum Model: Sendable {

        /// Mixedbread AI Large v1 - 1024 dimensions, highest quality.
        ///
        /// Recommended for applications where embedding quality is paramount.
        /// Excellent for semantic similarity and clustering.
        case mxbaiEmbedLarge

        /// BGE Large English v1.5 - 1024 dimensions.
        ///
        /// Optimized for retrieval and semantic search applications.
        /// Excellent choice for RAG systems.
        case bgeLarge

        /// BGE Base English v1.5 - 768 dimensions.
        ///
        /// Good balance of quality and speed. Suitable for most applications.
        case bgeBase

        /// All-MiniLM-L6-v2 - 384 dimensions.
        ///
        /// Lightweight and fast. Best for resource-constrained environments
        /// or when latency is critical.
        case miniLM

        /// Qwen3 Embedding 0.6B - up to 1024 dimensions, 4-bit quantized.
        ///
        /// Modern architecture with flexible dimensions. Good multilingual support.
        case qwen3Embedding

        /// Nomic Embed Text v1.5 - supports Matryoshka embeddings.
        ///
        /// Allows truncating embeddings to smaller dimensions while
        /// preserving most of the semantic information.
        case nomicTextV1_5

        /// Custom model from HuggingFace Hub.
        ///
        /// Use any compatible model by providing its Hub ID:
        /// ```swift
        /// let service = try await MLXEmbeddingService(model: .custom("my-org/my-model"))
        /// ```
        case custom(String)

        /// Custom model from a local directory.
        ///
        /// Load a model from a local path:
        /// ```swift
        /// let url = URL(fileURLWithPath: "/path/to/model")
        /// let service = try await MLXEmbeddingService(model: .directory(url))
        /// ```
        case directory(URL)

        /// The MLXEmbedders ModelConfiguration for this model.
        public var configuration: MLXEmbedders.ModelConfiguration {
            switch self {
            case .mxbaiEmbedLarge:
                return .mixedbread_large
            case .bgeLarge:
                return .bge_large
            case .bgeBase:
                return .bge_base
            case .miniLM:
                return .minilm_l6
            case .qwen3Embedding:
                return .qwen3_embedding
            case .nomicTextV1_5:
                return .nomic_text_v1_5
            case .custom(let id):
                return MLXEmbedders.ModelConfiguration(id: id)
            case .directory(let url):
                return MLXEmbedders.ModelConfiguration(directory: url)
            }
        }

        /// Human-readable name for the model.
        public var name: String {
            switch self {
            case .mxbaiEmbedLarge: return "mxbai-embed-large-v1"
            case .bgeLarge: return "bge-large-en-v1.5"
            case .bgeBase: return "bge-base-en-v1.5"
            case .miniLM: return "all-MiniLM-L6-v2"
            case .qwen3Embedding: return "Qwen3-Embedding-0.6B"
            case .nomicTextV1_5: return "nomic-embed-text-v1.5"
            case .custom(let id): return id
            case .directory(let url): return url.lastPathComponent
            }
        }
    }

    // MARK: - Properties

    /// The loaded model container.
    private let container: MLXEmbedders.ModelContainer

    /// The name of the loaded model.
    public let modelName: String

    /// Cached dimensions value.
    private var _dimensions: Int?

    // MARK: - Initialization

    /// Creates a new MLX embedding service with the specified model.
    ///
    /// The model will be downloaded from HuggingFace Hub on first use if not
    /// already cached. Downloads are stored in `~/.cache/huggingface/hub/`.
    ///
    /// - Parameters:
    ///   - model: The embedding model to use (default: `.mxbaiEmbedLarge`)
    ///   - cacheDirectory: Optional custom directory for model cache
    ///   - progressHandler: Optional callback for download progress (0.0 to 1.0)
    /// - Throws: ``MLXEmbeddingError/modelLoadFailed(_:_:)`` if model loading fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let service = try await MLXEmbeddingService(
    ///     model: .bgeLarge,
    ///     progressHandler: { progress in
    ///         print("Download: \(Int(progress * 100))%")
    ///     }
    /// )
    /// ```
    public init(
        model: Model = .mxbaiEmbedLarge,
        cacheDirectory: URL? = nil,
        progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws {
        self.modelName = model.name

        let hub: HubApi
        if let cacheDirectory {
            hub = HubApi(downloadBase: cacheDirectory)
        } else {
            hub = HubApi()
        }

        do {
            self.container = try await MLXEmbedders.loadModelContainer(
                hub: hub,
                configuration: model.configuration,
                progressHandler: { progress in
                    progressHandler?(progress.fractionCompleted)
                }
            )
        } catch {
            throw MLXEmbeddingError.modelLoadFailed(model.name, error)
        }
    }

    // MARK: - EmbeddingProvider Conformance

    /// The dimensionality of the embedding vectors.
    ///
    /// This is determined by running a test embedding on first access.
    /// Dimensions vary by model (384 for MiniLM, 1024 for BGE-Large, etc.).
    public var dimensions: Int {
        get async throws {
            if let dims = _dimensions {
                return dims
            }
            // Determine dimensions by running a test embedding
            let testEmbedding = try await embed("test")
            _dimensions = testEmbedding.count
            return testEmbedding.count
        }
    }

    /// Generates an embedding vector for the given text.
    ///
    /// Uses the transformer model to encode the text into a dense vector.
    /// The vector is L2-normalized for cosine similarity computation.
    ///
    /// - Parameter text: The text to embed (sentence or paragraph)
    /// - Returns: L2-normalized float vector
    /// - Throws: ``MLXEmbeddingError/encodingFailed(_:_:)`` if encoding fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let embedding = try await service.embed("Machine learning is fascinating.")
    /// print("Vector dimensions: \(embedding.count)")
    /// ```
    public func embed(_ text: String) async throws -> [Float] {
        await container.perform { model, tokenizer, pooler in
            // Tokenize - returns [Int] of token IDs
            let tokenIds = tokenizer.encode(text: text)

            // Create input tensors with batch dimension
            let inputIds = MLXArray(tokenIds.map { Int32($0) }).expandedDimensions(axis: 0)

            // Create attention mask (1 for all real tokens)
            let attentionMask = MLXArray(
                [Int32](repeating: 1, count: tokenIds.count)
            ).expandedDimensions(axis: 0)

            // Forward pass
            let output = model(
                inputIds,
                positionIds: nil,
                tokenTypeIds: nil,
                attentionMask: attentionMask
            )

            // Mean pooling over token embeddings (best for sentence-transformer style models)
            // Note: The model's pooledOutput is often a classification projection, not suitable
            // for semantic similarity. Mean pooling of hidden states works better.
            let meanPooler = Pooling(strategy: .mean)
            let pooled = meanPooler(output, mask: attentionMask, normalize: true)

            eval(pooled)
            return pooled.squeezed(axis: 0).asArray(Float.self)
        }
    }

    /// Generates embedding vectors for multiple texts.
    ///
    /// Currently processes texts sequentially to avoid memory issues with large batches.
    /// Future versions may support true batching for improved throughput.
    ///
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Array of L2-normalized float vectors
    /// - Throws: ``MLXEmbeddingError/encodingFailed(_:_:)`` if encoding fails
    public func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        // Process texts sequentially to avoid memory issues with large batches
        var results: [[Float]] = []
        for text in texts {
            let embedding = try await embed(text)
            results.append(embedding)
        }
        return results
    }

    /// Calculates cosine similarity between two texts.
    ///
    /// - Parameters:
    ///   - text1: First text
    ///   - text2: Second text
    /// - Returns: Similarity score between -1 and 1
    /// - Throws: ``MLXEmbeddingError/encodingFailed(_:_:)`` if either text fails to encode
    public func similarity(between text1: String, and text2: String) async throws -> Float {
        let emb1 = try await embed(text1)
        let emb2 = try await embed(text2)
        // Vectors are already L2-normalized, so dot product = cosine similarity
        return zip(emb1, emb2).map(*).reduce(0, +)
    }
}

// MARK: - MLXEmbeddingError

/// Errors specific to MLX-based embedding services.
///
/// These errors indicate issues with model loading or text encoding.
public enum MLXEmbeddingError: LocalizedError {

    /// Failed to load the embedding model.
    ///
    /// This can occur if:
    /// - The model ID is invalid
    /// - Network connection failed during download
    /// - Insufficient disk space
    /// - The model format is incompatible
    case modelLoadFailed(String, Error)

    /// Failed to encode the text.
    ///
    /// This is rare with transformer models but can occur with
    /// extremely long texts or encoding issues.
    case encodingFailed(String, Error)

    /// A localized description of the error.
    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let modelName, let error):
            return "Failed to load MLX embedding model '\(modelName)': \(error.localizedDescription)"
        case .encodingFailed(let text, let error):
            let preview = String(text.prefix(50))
            return "Failed to encode text '\(preview)...': \(error.localizedDescription)"
        }
    }
}
