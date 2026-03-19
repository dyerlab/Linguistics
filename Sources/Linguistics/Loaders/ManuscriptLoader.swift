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

    // MARK: - Public API

    /// Loads a single Markdown file and returns a ``Corpus``.
    ///
    /// - Parameters:
    ///   - url: Path to a `.md` file.
    ///   - profile: The ``DocumentProfile`` used to classify section headings.
    ///              Defaults to ``DocumentProfile/scientificPaper``.
    ///   - granularity: Whether to embed whole sections or individual paragraphs.
    ///   - provider: Embedding backend.
    ///   - option: Provider tag attached to every ``TextEmbedding``.
    public static func load(
        from url: URL,
        profile: DocumentProfile = .scientificPaper,
        granularity: EmbeddingGranularity = .section,
        using provider: any EmbeddingProvider,
        as option: EmbeddingProviderOption
    ) async throws -> Corpus {
        let markdown = try String(contentsOf: url, encoding: .utf8)
        let filename = url.deletingPathExtension().lastPathComponent
        let sections = parseSections(from: markdown, profile: profile)

        let title = extractTitle(from: markdown) ?? filename
        var meta: [String: String] = ["filename": url.lastPathComponent]
        if let doi = extractDOI(from: markdown) { meta["doi"] = doi }

        let embeddings = try await makeEmbeddings(
            from: sections, granularity: granularity, provider: provider, option: option
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
    ///   - granularity: Section-level or paragraph-level chunking.
    ///   - provider: Embedding backend.
    ///   - option: Provider tag attached to every ``TextEmbedding``.
    public static func loadAll(
        from directory: URL,
        profile: DocumentProfile = .scientificPaper,
        granularity: EmbeddingGranularity = .section,
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
                from: url, profile: profile, granularity: granularity,
                using: provider, as: option
            )
            corpora.append(corpus)
        }
        return corpora
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
        provider: any EmbeddingProvider,
        option: EmbeddingProviderOption
    ) async throws -> [TextEmbedding] {

        var embeddings: [TextEmbedding] = []

        for (part, sectionText) in sections {
            let chunks: [String]

            switch granularity {
            case .section:
                chunks = [sectionText]

            case .paragraph:
                var paragraphs: [String] = []
                let tokenizer = NLTokenizer(unit: .paragraph)
                tokenizer.string = sectionText
                tokenizer.enumerateTokens(in: sectionText.startIndex..<sectionText.endIndex) { range, _ in
                    let para = String(sectionText[range])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !para.isEmpty { paragraphs.append(para) }
                    return true
                }
                chunks = paragraphs
            }

            for chunk in chunks {
                let vector = try await provider.embed(chunk)
                embeddings.append(TextEmbedding(
                    provider: option,
                    vector: vector,
                    metadata: [
                        "part": part.rawValue,
                        "granularity": granularity.rawValue,
                        "text": chunk
                    ]
                ))
            }
        }

        return embeddings
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
