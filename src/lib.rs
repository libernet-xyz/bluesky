// Copyright 2026 The Libernet Team
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

#[cfg(any(target_arch = "wasm32", test))]
mod scalar29;

#[cfg(any(not(target_arch = "wasm32"), test))]
mod scalar64;

#[cfg(target_arch = "wasm32")]
pub use scalar29::*;

#[cfg(not(target_arch = "wasm32"))]
pub use scalar64::*;
