//
//  AcademicProgramLoader.swift
//  Linguistics
//
//  Created by Rodney Dyer on 3/10/26.
//

import Foundation

/// Loads academic program data from a CSV file and returns one ``Corpus`` per program.
///
/// **Expected CSV columns (header row required):**
/// `University`, `Program`, `Course`, `Title`, `Credits`, `Bulletin`
///
/// **Mapping:**
/// | CSV | Model |
/// |-----|-------|
/// | `(University, Program)` pair | one ``Corpus`` |
/// | `Program` | `Corpus.label` |
/// | `University` | `Corpus.metadata["university"]` |
/// | `Course` | `TextEmbedding.metadata["course"]` |
/// | `"\(Title) \(Bulletin)"` | embedded text; also stored in `TextEmbedding.metadata["text"]` |
/// | `Credits` | `TextEmbedding.scaling` (`Double`; defaults to `1.0` if unparseable) |
///
/// **Efficiency:** Within a single university, each unique course code is embedded once
/// and the resulting ``TextEmbedding`` is reused across every program that includes it.
/// Embeddings are never shared across universities — identical course codes may carry
/// different content at different institutions.
///
/// ## Example
///
/// ```swift
/// let provider = try NLEmbeddingService()
/// let corpora = try await AcademicProgramLoader.load(
///     from: csvURL,
///     using: provider,
///     as: .nlEmbedding
/// )
/// print(corpora.count)           // one per (University, Program) pair
/// print(corpora.first?.label)    // e.g. "Biology"
/// ```
public enum AcademicProgramLoader {

    /// Parses the CSV at `url`, embeds each unique course per university, and
    /// returns one ``Corpus`` for every `(University, Program)` pair found.
    ///
    /// Programs and courses are returned in the order they first appear in the file.
    /// When the same course code appears in multiple programs at the same university,
    /// the title, bulletin, and credits from the first occurrence are used for the
    /// shared embedding.
    ///
    /// - Parameters:
    ///   - url: Path to a UTF-8 encoded CSV file with the expected column layout.
    ///   - provider: The embedding backend used to encode course text.
    ///   - option: The ``EmbeddingProviderOption`` tag attached to every produced ``TextEmbedding``.
    /// - Returns: Array of ``Corpus`` values, one per program, in file order.
    /// - Throws: A `CocoaError` if the file cannot be read; any error thrown by `provider`.
    public static func load(
        from url: URL,
        using provider: any EmbeddingProvider,
        as option: EmbeddingProviderOption
    ) async throws -> [Corpus] {

        let rawText = try String(contentsOf: url, encoding: .utf8)
        let rows = parseCSV(rawText)
        guard rows.count > 1 else { return [] }

        struct RawRow {
            let university: String
            let program: String
            let course: String
            let title: String
            let credits: Double
            let bulletin: String
        }

        let parsed: [RawRow] = rows.dropFirst().compactMap { fields in
            guard fields.count >= 6 else { return nil }
            return RawRow(
                university: fields[0],
                program: fields[1],
                course: fields[2],
                title: fields[3],
                credits: Double(fields[4]) ?? 1.0,
                bulletin: fields[5]
            )
        }

        // Stable-ordered list of universities as first encountered in the file.
        var universityOrder: [String] = []
        var seenUniversities = Set<String>()
        for row in parsed where seenUniversities.insert(row.university).inserted {
            universityOrder.append(row.university)
        }

        let byUniversity = Dictionary(grouping: parsed, by: \.university)
        var corpora: [Corpus] = []

        for university in universityOrder {
            guard let uniRows = byUniversity[university] else { continue }

            // Embed each unique course code once (first-occurrence wins).
            var courseCache: [String: TextEmbedding] = [:]
            var seenCourses = Set<String>()

            for row in uniRows where seenCourses.insert(row.course).inserted {
                let embeddingText = "\(row.title) \(row.bulletin)"
                let vector = try await provider.embed(embeddingText)
                courseCache[row.course] = TextEmbedding(
                    provider: option,
                    vector: vector,
                    metadata: ["course": row.course, "text": embeddingText],
                    scaling: row.credits
                )
            }

            // Stable-ordered programs within this university.
            var programOrder: [String] = []
            var seenPrograms = Set<String>()
            for row in uniRows where seenPrograms.insert(row.program).inserted {
                programOrder.append(row.program)
            }

            let byProgram = Dictionary(grouping: uniRows, by: \.program)

            for program in programOrder {
                guard let programRows = byProgram[program] else { continue }
                let embeddings = programRows.compactMap { courseCache[$0.course] }
                corpora.append(Corpus(
                    label: program,
                    metadata: ["university": university],
                    embeddings: embeddings
                ))
            }
        }

        return corpora
    }
}

// MARK: - RFC 4180 CSV Parser

/// Parses a UTF-8 CSV string into a 2D array of field values.
///
/// Handles quoted fields containing commas, newlines, and escaped double-quotes (`""`),
/// as required by RFC 4180. Supports `\n`, `\r`, and `\r\n` line endings.
private func parseCSV(_ text: String) -> [[String]] {
    var rows: [[String]] = []
    var currentRow: [String] = []
    var currentField = ""
    var inQuotes = false
    var idx = text.startIndex

    while idx < text.endIndex {
        let ch = text[idx]
        let nextIdx = text.index(after: idx)

        if inQuotes {
            if ch == "\"" {
                if nextIdx < text.endIndex && text[nextIdx] == "\"" {
                    // Escaped double-quote: "" → "
                    currentField.append("\"")
                    idx = text.index(after: nextIdx)
                    continue
                } else {
                    inQuotes = false
                }
            } else {
                currentField.append(ch)
            }
        } else {
            switch ch {
            case "\"":
                inQuotes = true
            case ",":
                currentRow.append(currentField)
                currentField = ""
            case "\r":
                currentRow.append(currentField)
                currentField = ""
                rows.append(currentRow)
                currentRow = []
                if nextIdx < text.endIndex && text[nextIdx] == "\n" {
                    idx = nextIdx   // consume the \n of a \r\n pair
                }
            case "\n":
                currentRow.append(currentField)
                currentField = ""
                rows.append(currentRow)
                currentRow = []
            default:
                currentField.append(ch)
            }
        }
        idx = text.index(after: idx)
    }

    // Flush the final field and row.
    currentRow.append(currentField)
    if currentRow.contains(where: { !$0.isEmpty }) {
        rows.append(currentRow)
    }

    return rows
}
