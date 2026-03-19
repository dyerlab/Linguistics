//
//  CorpusTypes.swift
//  TrendSpotting
//
//  Created by rodney on 3/5/26.
//

import Foundation

public enum ManuscriptParts: String, Codable, CaseIterable, Sendable {
    
    case Title = "Title"
    case Abstract = "Abstract"
    case Introduction = "Introduction"
    case Methods = "Methods"
    case Results = "Results"
    case Discussion = "Discussion"
    case Other = "Other"
    
}
