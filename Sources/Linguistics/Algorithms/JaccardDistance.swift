//
//  JaccardDistance.swift
//  Linguistics
//
//  Created by Rodney Dyer on 1/23/26.
//

import Foundation

/// Returns the Jaccard similarity between two sets of strings.
///
/// Jaccard similarity is defined as the size of the intersection divided by
/// the size of the union:
///
/// ```
/// J(A, B) = |A ∩ B| / |A ∪ B|
/// ```
///
/// The result ranges from `0.0` (completely disjoint sets) to `1.0` (identical
/// sets). Returns `0.0` when either set is empty.
///
/// Typical use: comparing token sets, tag clouds, or vocabulary overlap between
/// two documents before computing a more expensive embedding similarity.
///
/// - Parameters:
///   - a: The first set of strings.
///   - b: The second set of strings.
/// - Returns: A similarity score in `[0.0, 1.0]`.
public func jaccardSimilarity(a: Set<String>, b: Set<String>) -> CGFloat {
    guard !a.isEmpty, !b.isEmpty else { return 0 }
    let inter = a.intersection(b).count
    let uni = a.union(b).count
    return uni == 0 ? 0.0 : Double(inter) / Double(uni)
}
