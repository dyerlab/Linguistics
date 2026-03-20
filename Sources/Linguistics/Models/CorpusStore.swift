//
//  CorpusStore.swift
//  Linguistics
//
//  Created by Rodney Dyer on 3/19/26.
//

import Foundation
import SQLite3
import MatrixStuff

// MARK: - SQLITE_TRANSIENT shim
// The C macro is not imported into Swift; recreate the sentinel value manually.
private let SQLITE_TRANSIENT = unsafeBitCast(-1 as Int, to: sqlite3_destructor_type.self)

/// Persists and queries ``Corpus`` values in a local SQLite database.
///
/// `CorpusStore` maps the Linguistics type hierarchy to a two-table relational schema
/// that R consumers can query via `RSQLite` without additional preprocessing.
///
/// Embedding vectors are stored as little-endian `Float32` BLOBs for compact storage.
/// Deserialize in R with:
///
/// ```r
/// readBin(blob_col[[1]], what = "numeric", n = dims, size = 4, endian = "little")
/// ```
///
/// ## Schema
///
/// ```sql
/// documents  (id, corpus_uuid, title, filename, doi, created_at)
/// embeddings (id, document_id, part, granularity, provider,
///             dimensions, vector BLOB, scaling, source_text)
/// ```
///
/// ## Example
///
/// ```swift
/// let store = try CorpusStore(url: URL(fileURLWithPath: "results.sqlite"))
/// try store.write(corpora)
/// let loaded = try store.readAll()
/// ```
public final class CorpusStore: @unchecked Sendable {

    private var db: OpaquePointer?

    // MARK: - Init / deinit

    /// Opens (or creates) the SQLite database at `url` and ensures the schema exists.
    public init(url: URL) throws {
        let rc = sqlite3_open(url.path, &db)
        guard rc == SQLITE_OK else {
            throw CorpusStoreError.openFailed(dbError)
        }
        try createSchema()
        migrateSchema()
    }

    deinit { sqlite3_close(db) }

    // MARK: - Write

    /// Writes `corpora` to the store inside a single transaction.
    ///
    /// Each ``Corpus`` becomes one row in `documents`; each ``TextEmbedding``
    /// becomes one row in `embeddings` referencing that document row.
    public func write(_ corpora: [Corpus]) throws {
        try exec("BEGIN TRANSACTION")
        do {
            for corpus in corpora { try writeCorpus(corpus) }
            try exec("COMMIT")
        } catch {
            _ = try? exec("ROLLBACK")
            throw error
        }
    }

    /// Convenience for writing a single ``Corpus``.
    public func write(_ corpus: Corpus) throws {
        try write([corpus])
    }

    // MARK: - Read

    /// Reads all corpora from the store, reconstructing ``TextEmbedding`` values.
    ///
    /// Vectors are deserialized from little-endian `Float32` BLOBs and upcast to
    /// `Double` to reconstruct the ``MatrixStuff/Vector``.
    public func readAll() throws -> [Corpus] {
        let sql = "SELECT id, corpus_uuid, title, filename, doi FROM documents ORDER BY id"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CorpusStoreError.queryFailed(dbError)
        }
        defer { sqlite3_finalize(stmt) }

        var corpora: [Corpus] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let rowID    = sqlite3_column_int64(stmt, 0)
            let uuidStr  = columnString(stmt, 1) ?? UUID().uuidString
            let title    = columnString(stmt, 2) ?? ""
            let filename = columnString(stmt, 3)
            let doi      = columnString(stmt, 4)

            let id = UUID(uuidString: uuidStr) ?? UUID()
            var meta: [String: String] = [:]
            if let f = filename { meta["filename"] = f }
            if let d = doi      { meta["doi"] = d }

            let embeddings = try readEmbeddings(for: rowID)
            corpora.append(Corpus(id: id, label: title, metadata: meta, embeddings: embeddings))
        }
        return corpora
    }

    // MARK: - Schema

    private func createSchema() throws {
        try exec("""
            CREATE TABLE IF NOT EXISTS documents (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                corpus_uuid  TEXT    NOT NULL,
                title        TEXT,
                filename     TEXT,
                doi          TEXT,
                created_at   TEXT
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id    INTEGER REFERENCES documents(id),
                part           TEXT,
                granularity    TEXT,
                provider       TEXT,
                dimensions     INTEGER,
                vector         BLOB,
                scaling        REAL,
                source_text    TEXT,
                sequence_index INTEGER,
                scheme         TEXT
            );
            """)
    }

    /// Adds columns introduced after the initial schema.
    ///
    /// SQLite does not support `ADD COLUMN IF NOT EXISTS`, so each statement is
    /// attempted individually and any "duplicate column" error is silently ignored.
    /// This keeps existing databases forward-compatible without a full migration.
    private func migrateSchema() {
        _ = try? exec("ALTER TABLE embeddings ADD COLUMN sequence_index INTEGER")
        _ = try? exec("ALTER TABLE embeddings ADD COLUMN scheme TEXT")
    }

    // MARK: - Private write helpers

    private func writeCorpus(_ corpus: Corpus) throws {
        let sql = """
            INSERT INTO documents (corpus_uuid, title, filename, doi, created_at)
            VALUES (?, ?, ?, ?, ?)
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CorpusStoreError.queryFailed(dbError)
        }
        defer { sqlite3_finalize(stmt) }

        let now = ISO8601DateFormatter().string(from: Date())
        bindText(stmt, 1, corpus.id.uuidString)
        bindText(stmt, 2, corpus.label)
        bindText(stmt, 3, corpus.metadata["filename"])
        bindText(stmt, 4, corpus.metadata["doi"])
        bindText(stmt, 5, now)

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw CorpusStoreError.writeFailed(dbError)
        }

        let docID = sqlite3_last_insert_rowid(db)
        for embedding in corpus.embeddings {
            try writeEmbedding(embedding, docID: docID)
        }
    }

    private func writeEmbedding(_ embedding: TextEmbedding, docID: Int64) throws {
        let sql = """
            INSERT INTO embeddings
                (document_id, part, granularity, provider, dimensions, vector,
                 scaling, source_text, sequence_index, scheme)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CorpusStoreError.queryFailed(dbError)
        }
        defer { sqlite3_finalize(stmt) }

        let blob = vectorToBlob(embedding.vector)

        sqlite3_bind_int64(stmt, 1, docID)
        bindText(stmt, 2, embedding.metadata["part"])
        bindText(stmt, 3, embedding.metadata["granularity"])
        bindText(stmt, 4, embedding.provider.storeKey)
        sqlite3_bind_int(stmt, 5, Int32(embedding.vector.count))
        blob.withUnsafeBytes { bytes in
            _ = sqlite3_bind_blob(stmt, 6, bytes.baseAddress, Int32(blob.count), SQLITE_TRANSIENT)
        }
        sqlite3_bind_double(stmt, 7, embedding.scaling)
        bindText(stmt, 8, embedding.metadata["text"])
        if let idxStr = embedding.metadata["sequence_index"], let idx = Int(idxStr) {
            sqlite3_bind_int(stmt, 9, Int32(idx))
        } else {
            sqlite3_bind_null(stmt, 9)
        }
        bindText(stmt, 10, embedding.metadata["scheme"])

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw CorpusStoreError.writeFailed(dbError)
        }
    }

    // MARK: - Private read helpers

    private func readEmbeddings(for docID: Int64) throws -> [TextEmbedding] {
        let sql = """
            SELECT part, granularity, provider, dimensions, vector, scaling,
                   source_text, sequence_index, scheme
            FROM embeddings WHERE document_id = ? ORDER BY id
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw CorpusStoreError.queryFailed(dbError)
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int64(stmt, 1, docID)

        var embeddings: [TextEmbedding] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let part        = columnString(stmt, 0)
            let granularity = columnString(stmt, 1)
            let providerKey = columnString(stmt, 2) ?? ""
            let dims        = Int(sqlite3_column_int(stmt, 3))
            let vector      = blobToVector(stmt, col: 4, dims: dims)
            let scaling     = sqlite3_column_double(stmt, 5)
            let sourceText  = columnString(stmt, 6)
            let seqIdx      = sqlite3_column_type(stmt, 7) != SQLITE_NULL
                                  ? Int(sqlite3_column_int(stmt, 7)) : nil
            let scheme      = columnString(stmt, 8)

            var meta: [String: String] = [:]
            if let p = part        { meta["part"] = p }
            if let g = granularity { meta["granularity"] = g }
            if let t = sourceText  { meta["text"] = t }
            if let i = seqIdx      { meta["sequence_index"] = "\(i)" }
            if let s = scheme      { meta["scheme"] = s }

            embeddings.append(TextEmbedding(
                provider: EmbeddingProviderOption(storeKey: providerKey),
                vector: vector,
                metadata: meta,
                scaling: scaling
            ))
        }
        return embeddings
    }

    // MARK: - Vector serialization

    /// Encodes a ``MatrixStuff/Vector`` as a little-endian `Float32` byte buffer.
    private func vectorToBlob(_ vector: Vector) -> Data {
        var floats = vector.map { Float($0) }
        return Data(bytes: &floats, count: floats.count * MemoryLayout<Float>.size)
    }

    /// Decodes a little-endian `Float32` BLOB back to a ``MatrixStuff/Vector``.
    private func blobToVector(_ stmt: OpaquePointer?, col: Int32, dims: Int) -> Vector {
        guard let rawPtr = sqlite3_column_blob(stmt, col) else {
            return Vector(repeating: 0, count: dims)
        }
        let byteCount  = Int(sqlite3_column_bytes(stmt, col))
        let floatCount = byteCount / MemoryLayout<Float>.size
        let floatPtr   = rawPtr.assumingMemoryBound(to: Float.self)
        return (0..<floatCount).map { Double(floatPtr[$0]) }
    }

    // MARK: - SQLite utilities

    @discardableResult
    private func exec(_ sql: String) throws -> Int32 {
        var errMsg: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(db, sql, nil, nil, &errMsg)
        if rc != SQLITE_OK {
            let msg = errMsg.map { String(cString: $0) } ?? "Unknown error"
            sqlite3_free(errMsg)
            throw CorpusStoreError.execFailed(msg)
        }
        return rc
    }

    private var dbError: String { String(cString: sqlite3_errmsg(db)) }

    private func columnString(_ stmt: OpaquePointer?, _ col: Int32) -> String? {
        guard let cStr = sqlite3_column_text(stmt, col) else { return nil }
        return String(cString: cStr)
    }

    private func bindText(_ stmt: OpaquePointer?, _ col: Int32, _ text: String?) {
        if let text {
            sqlite3_bind_text(stmt, col, text, -1, SQLITE_TRANSIENT)
        } else {
            sqlite3_bind_null(stmt, col)
        }
    }
}

// MARK: - EmbeddingProviderOption store key

private extension EmbeddingProviderOption {

    /// A stable, human-readable string used as the `provider` column value in SQLite.
    var storeKey: String {
        switch self {
        case .fdlEmbedding:     return "fdl"
        case .nlEmbedding:      return "nl"
        case .miniLM:           return "miniLM"
        case .bgeBase:          return "bgeBase"
        case .bgeLarge:         return "bgeLarge"
        case .mxbaiEmbedLarge:  return "mxbaiEmbedLarge"
        case .qwen3Embedding:   return "qwen3Embedding"
        case .nomicTextV1_5:    return "nomicTextV1_5"
        case .custom(let id):   return "custom:\(id)"
        }
    }

    /// Reconstructs an ``EmbeddingProviderOption`` from a ``storeKey`` string.
    init(storeKey: String) {
        switch storeKey {
        case "fdl":             self = .fdlEmbedding
        case "nl":              self = .nlEmbedding
        case "miniLM":          self = .miniLM
        case "bgeBase":         self = .bgeBase
        case "bgeLarge":        self = .bgeLarge
        case "mxbaiEmbedLarge": self = .mxbaiEmbedLarge
        case "qwen3Embedding":  self = .qwen3Embedding
        case "nomicTextV1_5":   self = .nomicTextV1_5
        default:
            let id = storeKey.hasPrefix("custom:") ? String(storeKey.dropFirst(7)) : storeKey
            self = .custom(id)
        }
    }
}

// MARK: - Error type

/// Errors thrown by ``CorpusStore``.
public enum CorpusStoreError: Error, LocalizedError {
    case openFailed(String)
    case execFailed(String)
    case queryFailed(String)
    case writeFailed(String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let msg):  return "Failed to open database: \(msg)"
        case .execFailed(let msg):  return "SQL execution failed: \(msg)"
        case .queryFailed(let msg): return "Query preparation failed: \(msg)"
        case .writeFailed(let msg): return "Write step failed: \(msg)"
        }
    }
}
