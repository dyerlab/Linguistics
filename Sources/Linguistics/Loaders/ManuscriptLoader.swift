//
//  ManuscriptLoader.swift
//  Linguistics
//
//  Created by Rodney Dyer on 3/10/26.
//

import Foundation
import NaturalLanguage

/// Loads Markdown-formatted research manuscripts and returns one ``Corpus`` per file.
///
/// **Expected input:** Markdown files produced by a PDF→Markdown converter (e.g., `marker`,
/// `nougat`). Section headings must be Markdown `#`-prefixed lines; body text is plain prose.
///
/// **Mapping:**
/// | Source | Model |
/// |--------|-------|
/// | One `.md` file | one ``Corpus`` |
/// | First level-1 heading (`# …`) | `Corpus.label` (title) |
/// | `"filename"`, `"doi"` | `Corpus.metadata` |
/// | Each classified section or paragraph | one ``TextEmbedding`` |
/// | `ManuscriptParts.rawValue` | `TextEmbedding.metadata["part"]` |
/// | `EmbeddingGranularity.rawValue` | `TextEmbedding.metadata["granularity"]` |
/// | Embedded text | `TextEmbedding.metadata["text"]` |
///
/// Lines appearing before the first heading (journal metadata, page numbers) are skipped.
/// Headings that don't match any ``SectionRule`` are assigned the profile's `fallbackType`.
///
/// ## Example
///
/// ```swift
/// let provider = try NLEmbeddingService()
/// let corpora = try await ManuscriptLoader.loadAll(
///     from: markdownDirectory,
///     granularity: .paragraph,
///     using: provider,
///     as: .nlEmbedding
/// )
/// ```
public enum ManuscriptLoader {

    // MARK: - Single-provider API

    /// Loads a single Markdown file and returns a ``Corpus``.
    ///
    /// - Parameters:
    ///   - url: Path to a `.md` file.
    ///   - profile: The ``DocumentProfile`` used to classify section headings.
    ///              Defaults to ``DocumentProfile/scientificPaper``.
    ///   - parts: Restrict embedding to these manuscript sections only.
    ///            Pass `nil` (default) to embed every classified section.
    ///   - granularity: Whether to embed whole sections, individual paragraphs, or both
    ///                  hierarchically (section + ordered paragraphs in one pass).
    ///   - scheme: An optional analysis-scheme label (e.g. `"introduction_hierarchical"`)
    ///             stamped on every ``TextEmbedding`` and written to the `scheme` column
    ///             in ``CorpusStore``. Pass `nil` to leave the column empty.
    ///   - provider: Embedding backend.
    ///   - option: Provider tag attached to every ``TextEmbedding``.
    public static func load(
        from url: URL,
        profile: DocumentProfile = .scientificPaper,
        parts: [ManuscriptParts]? = nil,
        granularity: EmbeddingGranularity = .section,
        scheme: String? = nil,
        using provider: any EmbeddingProvider,
        as option: EmbeddingProviderOption
    ) async throws -> Corpus {
        let markdown = try String(contentsOf: url, encoding: .utf8)
        let filename = url.deletingPathExtension().lastPathComponent
        var sections = parseSections(from: markdown, profile: profile)
        if let allowed = parts { sections = sections.filter { allowed.contains($0.part) } }

        let title = extractTitle(from: markdown) ?? filename
        var meta: [String: String] = ["filename": url.lastPathComponent]
        if let doi = extractDOI(from: markdown) { meta["doi"] = doi }

        let embeddings = try await makeEmbeddings(
            from: sections, granularity: granularity, scheme: scheme,
            provider: provider, option: option
        )

        return Corpus(label: title, metadata: meta, embeddings: embeddings)
    }

    /// Loads every `.md` file in `directory` and returns one ``Corpus`` per file.
    ///
    /// Files are processed in ascending filename order for deterministic output.
    ///
    /// - Parameters:
    ///   - directory: A directory URL containing `.md` files.
    ///   - profile: The ``DocumentProfile`` used to classify headings.
    ///   - parts: Restrict embedding to these manuscript sections only. Pass `nil` to embed all.
    ///   - granularity: Section-level, paragraph-level, or hierarchical chunking.
    ///   - scheme: Optional analysis-scheme label stamped on every ``TextEmbedding``.
    ///   - provider: Embedding backend.
    ///   - option: Provider tag attached to every ``TextEmbedding``.
    public static func loadAll(
        from directory: URL,
        profile: DocumentProfile = .scientificPaper,
        parts: [ManuscriptParts]? = nil,
        granularity: EmbeddingGranularity = .section,
        scheme: String? = nil,
        using provider: any EmbeddingProvider,
        as option: EmbeddingProviderOption
    ) async throws -> [Corpus] {
        let urls = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "md" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var corpora: [Corpus] = []
        for url in urls {
            let corpus = try await load(
                from: url, profile: profile, parts: parts, granularity: granularity,
                scheme: scheme, using: provider, as: option
            )
            corpora.append(corpus)
        }
        return corpora
    }

    // MARK: - Multi-provider API

    /// Returns the section-level texts that would be embedded for the given configuration,
    /// without performing any embedding.
    ///
    /// This is the recommended way to collect texts for building an ``FDLEmbeddingService``
    /// vocabulary before calling ``loadAll(from:profile:parts:granularity:scheme:using:)-multi``.
    /// Passing one text per matched section per file to `FDLEmbeddingService` ensures that
    /// document-frequency cutoffs are computed across manuscripts (not individual paragraphs).
    ///
    /// - Parameters:
    ///   - directory: A directory URL containing `.md` files.
    ///   - profile: The ``DocumentProfile`` used to classify headings.
    ///   - parts: Restrict extraction to these sections. Pass `nil` to include all sections.
    /// - Returns: One text per matched section per file, in filename-ascending order.
    public static func extractTexts(
        from directory: URL,
        profile: DocumentProfile = .scientificPaper,
        parts: [ManuscriptParts]? = nil
    ) throws -> [String] {
        let urls = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "md" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var texts: [String] = []
        for url in urls {
            let markdown = try String(contentsOf: url, encoding: .utf8)
            var sections = parseSections(from: markdown, profile: profile)
            if let allowed = parts { sections = sections.filter { allowed.contains($0.part) } }
            texts.append(contentsOf: sections.map { $0.text })
        }
        return texts
    }

    /// Loads a single Markdown file using all providers in `embedder`, returning one
    /// ``Corpus`` whose ``Corpus/embeddings`` contain vectors from every provider.
    ///
    /// For `.sectionAndParagraphs` granularity the full section is stored at
    /// `sequence_index = 0` and each paragraph at `sequence_index = 1…N`, with one
    /// ``TextEmbedding`` per (chunk × provider) combination.
    ///
    /// - Parameters:
    ///   - url: Path to a `.md` file.
    ///   - profile: The ``DocumentProfile`` used to classify section headings.
    ///   - parts: Restrict embedding to these manuscript sections only. Pass `nil` for all.
    ///   - granularity: Chunking strategy. Defaults to `.sectionAndParagraphs`.
    ///   - scheme: Optional analysis-scheme label stamped on every ``TextEmbedding``.
    ///   - embedder: A ``MultiProviderEmbedder`` with all desired providers already initialised.
    public static func load(
        from url: URL,
        profile: DocumentProfile = .scientificPaper,
        parts: [ManuscriptParts]? = nil,
        granularity: EmbeddingGranularity = .sectionAndParagraphs,
        scheme: String? = nil,
        using embedder: MultiProviderEmbedder
    ) async throws -> Corpus {
        let markdown = try String(contentsOf: url, encoding: .utf8)
        let filename = url.deletingPathExtension().lastPathComponent
        var sections = parseSections(from: markdown, profile: profile)
        if let allowed = parts { sections = sections.filter { allowed.contains($0.part) } }

        let title = extractTitle(from: markdown) ?? filename
        var meta: [String: String] = ["filename": url.lastPathComponent]
        if let doi = extractDOI(from: markdown) { meta["doi"] = doi }

        let embeddings = try await makeEmbeddings(
            from: sections, granularity: granularity, scheme: scheme,
            embedder: embedder
        )

        return Corpus(label: title, metadata: meta, embeddings: embeddings)
    }

    /// Loads every `.md` file in `directory` using all providers in `embedder`,
    /// returning one ``Corpus`` per file with embeddings from every provider interleaved.
    ///
    /// Documents are processed **concurrently** via a `TaskGroup`. CPU-bound providers
    /// (FDL, NL) run in true parallel across documents; the ``MLXEmbeddingService`` actor
    /// naturally serialises GPU calls without any additional synchronisation. Results are
    /// returned in ascending filename order regardless of completion order.
    ///
    /// - Parameters:
    ///   - directory: A directory URL containing `.md` files.
    ///   - profile: The ``DocumentProfile`` used to classify headings.
    ///   - parts: Restrict embedding to these manuscript sections only. Pass `nil` for all.
    ///   - granularity: Chunking strategy. Defaults to `.sectionAndParagraphs`.
    ///   - scheme: Optional analysis-scheme label stamped on every ``TextEmbedding``.
    ///   - embedder: A ``MultiProviderEmbedder`` with all desired providers already initialised.
    ///   - onProgress: Optional callback invoked after each document completes.
    ///                 Receives `(completedCount, totalCount, filename)`. Called from the
    ///                 collection loop — safe to update UI or print without additional locking.
    public static func loadAll(
        from directory: URL,
        profile: DocumentProfile = .scientificPaper,
        parts: [ManuscriptParts]? = nil,
        granularity: EmbeddingGranularity = .sectionAndParagraphs,
        scheme: String? = nil,
        using embedder: MultiProviderEmbedder,
        onProgress: (@Sendable (Int, Int, String) async -> Void)? = nil
    ) async throws -> [Corpus] {
        let urls = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "md" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        let total = urls.count
        // Collect (originalIndex, Corpus) pairs so we can restore filename-sorted order
        // after concurrent completion.
        var indexed = [(Int, Corpus)]()
        indexed.reserveCapacity(total)

        try await withThrowingTaskGroup(of: (Int, Corpus).self) { group in
            for (index, url) in urls.enumerated() {
                group.addTask {
                    let corpus = try await load(
                        from: url, profile: profile, parts: parts, granularity: granularity,
                        scheme: scheme, using: embedder
                    )
                    return (index, corpus)
                }
            }
            // Collect completions; the loop runs in the calling task so onProgress
            // is called serially — no locking needed.
            var completed = 0
            for try await (index, corpus) in group {
                completed += 1
                await onProgress?(completed, total, corpus.metadata["filename"] ?? "")
                indexed.append((index, corpus))
            }
        }

        // Restore ascending filename order.
        return indexed.sorted { $0.0 < $1.0 }.map(\.1)
    }

    // MARK: - Parsing

    /// Splits Markdown into classified sections.
    ///
    /// Returns an ordered list of `(part, text)` pairs. Each pair represents the
    /// text body that follows a heading classified to that ``ManuscriptParts`` case.
    /// Empty sections (heading with no subsequent body text) are omitted.
    private static func parseSections(
        from markdown: String,
        profile: DocumentProfile
    ) -> [(part: ManuscriptParts, text: String)] {

        var sections: [(part: ManuscriptParts, text: String)] = []
        var currentPart: ManuscriptParts?
        var currentLines: [String] = []
        var pastFirstHeading = false

        func flushCurrent() {
            guard let part = currentPart else { return }
            let text = currentLines
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { sections.append((part: part, text: text)) }
            currentLines = []
        }

        for line in markdown.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.hasPrefix("#") {
                flushCurrent()
                pastFirstHeading = true

                // Strip all leading '#' and whitespace to get the heading text.
                let headingText = trimmed
                    .drop(while: { $0 == "#" })
                    .trimmingCharacters(in: .whitespaces)

                // Level-1 headings that aren't classified by the profile are the title —
                // capture its content (usually empty) under .Title.
                let isLevelOne = !trimmed.hasPrefix("##")
                if isLevelOne && profile.classify(headingText) == nil {
                    currentPart = .Title
                } else {
                    currentPart = profile.classify(headingText) ?? profile.fallbackType
                }

            } else if pastFirstHeading {
                currentLines.append(line)
            }
            // Lines before the first heading are silently skipped.
        }
        flushCurrent()

        return sections
    }

    // MARK: - Embedding

    private static func makeEmbeddings(
        from sections: [(part: ManuscriptParts, text: String)],
        granularity: EmbeddingGranularity,
        scheme: String?,
        provider: any EmbeddingProvider,
        option: EmbeddingProviderOption
    ) async throws -> [TextEmbedding] {

        var embeddings: [TextEmbedding] = []

        for (part, sectionText) in sections {
            switch granularity {

            case .section:
                let vector = try await provider.embed(sectionText)
                embeddings.append(TextEmbedding(
                    provider: option,
                    vector: vector,
                    metadata: metadataDict(
                        part: part, granularity: granularity,
                        text: sectionText, sequenceIndex: nil, scheme: scheme
                    )
                ))

            case .paragraph:
                for (index, para) in splitParagraphs(sectionText).enumerated() {
                    let vector = try await provider.embed(para)
                    embeddings.append(TextEmbedding(
                        provider: option,
                        vector: vector,
                        metadata: metadataDict(
                            part: part, granularity: granularity,
                            text: para, sequenceIndex: index + 1, scheme: scheme
                        )
                    ))
                }

            case .sectionAndParagraphs:
                // sequence_index 0 — full section
                let sectionVector = try await provider.embed(sectionText)
                embeddings.append(TextEmbedding(
                    provider: option,
                    vector: sectionVector,
                    metadata: metadataDict(
                        part: part, granularity: granularity,
                        text: sectionText, sequenceIndex: 0, scheme: scheme
                    )
                ))
                // sequence_index 1…N — individual paragraphs in order
                for (index, para) in splitParagraphs(sectionText).enumerated() {
                    let vector = try await provider.embed(para)
                    embeddings.append(TextEmbedding(
                        provider: option,
                        vector: vector,
                        metadata: metadataDict(
                            part: part, granularity: granularity,
                            text: para, sequenceIndex: index + 1, scheme: scheme
                        )
                    ))
                }
            }
        }

        return embeddings
    }

    /// Multi-provider variant: embeds every chunk with all providers in `embedder`,
    /// producing one ``TextEmbedding`` per (chunk × provider) combination.
    private static func makeEmbeddings(
        from sections: [(part: ManuscriptParts, text: String)],
        granularity: EmbeddingGranularity,
        scheme: String?,
        embedder: MultiProviderEmbedder
    ) async throws -> [TextEmbedding] {

        var embeddings: [TextEmbedding] = []

        for (part, sectionText) in sections {
            switch granularity {

            case .section:
                let results = try await embedder.embed(sectionText)
                for (_, embedding) in results {
                    embeddings.append(TextEmbedding(
                        provider: embedding.provider,
                        vector: embedding.vector,
                        metadata: metadataDict(
                            part: part, granularity: granularity,
                            text: sectionText, sequenceIndex: nil, scheme: scheme
                        ),
                        scaling: embedding.scaling
                    ))
                }

            case .paragraph:
                for (index, para) in splitParagraphs(sectionText).enumerated() {
                    let results = try await embedder.embed(para)
                    for (_, embedding) in results {
                        embeddings.append(TextEmbedding(
                            provider: embedding.provider,
                            vector: embedding.vector,
                            metadata: metadataDict(
                                part: part, granularity: granularity,
                                text: para, sequenceIndex: index + 1, scheme: scheme
                            ),
                            scaling: embedding.scaling
                        ))
                    }
                }

            case .sectionAndParagraphs:
                // sequence_index 0 — full section
                let sectionResults = try await embedder.embed(sectionText)
                for (_, embedding) in sectionResults {
                    embeddings.append(TextEmbedding(
                        provider: embedding.provider,
                        vector: embedding.vector,
                        metadata: metadataDict(
                            part: part, granularity: granularity,
                            text: sectionText, sequenceIndex: 0, scheme: scheme
                        ),
                        scaling: embedding.scaling
                    ))
                }
                // sequence_index 1…N — individual paragraphs in order
                for (index, para) in splitParagraphs(sectionText).enumerated() {
                    let paraResults = try await embedder.embed(para)
                    for (_, embedding) in paraResults {
                        embeddings.append(TextEmbedding(
                            provider: embedding.provider,
                            vector: embedding.vector,
                            metadata: metadataDict(
                                part: part, granularity: granularity,
                                text: para, sequenceIndex: index + 1, scheme: scheme
                            ),
                            scaling: embedding.scaling
                        ))
                    }
                }
            }
        }

        return embeddings
    }

    /// Splits `text` into non-empty paragraphs using `NLTokenizer`.
    private static func splitParagraphs(_ text: String) -> [String] {
        var paragraphs: [String] = []
        let tokenizer = NLTokenizer(unit: .paragraph)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let para = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !para.isEmpty { paragraphs.append(para) }
            return true
        }
        return paragraphs
    }

    /// Builds the metadata dictionary for a ``TextEmbedding``.
    private static func metadataDict(
        part: ManuscriptParts,
        granularity: EmbeddingGranularity,
        text: String,
        sequenceIndex: Int?,
        scheme: String?
    ) -> [String: String] {
        var meta: [String: String] = [
            "part": part.rawValue,
            "granularity": granularity.rawValue,
            "text": text
        ]
        if let idx = sequenceIndex { meta["sequence_index"] = "\(idx)" }
        if let s = scheme          { meta["scheme"] = s }
        return meta
    }

    // MARK: - Metadata extraction

    /// Returns the text of the first level-1 Markdown heading (`# …`), if present.
    private static func extractTitle(from markdown: String) -> String? {
        for line in markdown.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("# ") && !trimmed.hasPrefix("## ") {
                return trimmed
                    .drop(while: { $0 == "#" })
                    .trimmingCharacters(in: .whitespaces)
            }
        }
        return nil
    }

    /// Searches the first 3 000 characters for a DOI (`10.XXXX/…`).
    private static func extractDOI(from markdown: String) -> String? {
        let searchWindow = String(markdown.prefix(3000))
        guard let range = searchWindow.range(
            of: #"10\.\d{4,9}/[^\s]+"#,
            options: .regularExpression
        ) else { return nil }
        return String(searchWindow[range])
    }
}
