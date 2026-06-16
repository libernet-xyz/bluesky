# BlueSky

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/bluesky/ci.yml?label=CI)](https://github.com/libernet-xyz/bluesky/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-bluesky)](https://crates.io/crates/starkom-bluesky)
[![license](https://img.shields.io/crates/l/starkom-bluesky)](https://github.com/libernet-xyz/bluesky/blob/main/LICENSE)

## Overview

Starkom and Libernet are entirely based on a finite scalar field known as **BlueSky**.

BlueSky is a ~255-bit prime field and its order is the prime
`0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000001`. We call this number `p`.

`p` takes slightly less than 50% of the 256-bit range, leaving the MSB unset so that it can be used
for arbitrary purposes.

## Factors of `p-1`

`p-1`, the greatest integer that fits in a BlueSky scalar, is factorized as follows:

$$
2^{62} \cdot 3^{39} \cdot 31 \cdot 47491 \cdot 20313611 \cdot 117579359 \cdot 880985987837994013
$$

## Adicity

The 2-adicity of 62 has been chosen to warrant a very large FFT capacity and evaluation domain for
zkSNARK proofs. We didn't pick a higher 2-adicity because we targeted 64-bit machines, where the
implementation of 64-bit exponents is most efficient, so we intentionally kept the exponent below 64.

In addition to that, BlueSky also has 3-adicity. That allows implementing the 3-adic versions of the
FFT and FRI folding algorithms. The latter are especially important for recursion because hashing a
ternary Merkle tree results in a smaller witness than hashing a binary one (even though the Merkle
proof itself is _larger_!). For certain circuit sizes that are closer to a power of 3, the 3-adic
protocol is significantly more efficient.

We chose the exponent 39 for the 3-adicity because $3^{39}$ is roughly equal to $2^{62}$:

$$
39 \cdot log_2(3) = 61.813...
$$

therefore the 3-adic evaluation domain has roughly the same size as the 2-adic one.

## S-box optimization

The factorization of `p-1` does not contain 5, so raising to 5 is a permutation in the field and
$x^5$ can be used as the S-box for algebraic hashes such as [Poseidon][poseidon].

$x^5$ is widely considered optimal for multiplicative complexity because it can be implemented using
only 3 gates with quadratic constraints, and still achieves strong nonlinearity and extensive
diffusion.

[poseidon]: https://www.poseidon-hash.info/

## Implementation

The implementation is based on the interface from the `ff` crate and uses Montgomery form with four
64-bit limbs.
