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
            .macOS(.v11),
            .iOS(.v14),
        ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "Linguistics",
            targets: ["Linguistics"]
        ),
    ],
    dependencies: [],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "Linguistics"
        ),
        .testTarget(
            name: "LinguisticsTests",
            dependencies: ["Linguistics"]
        ),
    ]
)
