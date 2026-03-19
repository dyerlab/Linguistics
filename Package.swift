// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
//                      _                 _       _
//                   __| |_   _  ___ _ __| | __ _| |__
//                  / _` | | | |/ _ \ '__| |/ _` | '_ \
//                 | (_| | |_| |  __/ |  | | (_| | |_) |
//                  \__,_|\__, |\___|_|  |_|\__,_|_.__/
//                        |_ _/
//
//         Making Population Genetic Software That Doesn't Suck
//
//  Copyright (c) 2021-2026 Administravia LLC.  All Rights Reserved.
//


import PackageDescription

let package = Package(
    name: "Linguistics",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "Linguistics",
            targets: ["Linguistics"]
        ),
    ],
    dependencies: [
        // MLX for Apple Silicon acceleration
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        // MLX-LM includes MLXEmbedders for encoder models
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        // HuggingFace utilities for tokenizers and model hub
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // Shared math primitives (Vector = [Double], Matrix, etc.)
        .package(path: "../MatrixStuff"),
    ],
    targets: [
        .target(
            name: "Linguistics",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MatrixStuff", package: "MatrixStuff"),
            ],
            resources: [
                .process("Data"),
            ],
            swiftSettings: [
                // Guard #Preview macros that require Xcode's PreviewsMacros plugin.
                // SPM command-line builds (swift build / swift test) define this flag
                // automatically; use #if !SPM_BUILD around any #Preview blocks.
                .define("SPM_BUILD"),
            ],
            linkerSettings: [
                .linkedLibrary("sqlite3"),
            ]
        ),
        .testTarget(
            name: "LinguisticsTests",
            dependencies: ["Linguistics"]
        ),
    ]
)
