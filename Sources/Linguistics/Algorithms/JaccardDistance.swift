//
//  JaccardDistance.swift
//  Linguistics
//
//  Created by Rodney Dyer on 1/23/26.
//

import Foundation


public func jaccardSimilarity( a: Set<String>, b: Set<String>) -> CGFloat {
    guard !a.isEmpty, !b.isEmpty else { return 0 }
    let inter = a.intersection(b).count
    let uni = a.union(b).count
    return uni == 0 ? 0.0 : Double(inter) / Double(uni)
}



