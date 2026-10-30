//! Helpers for multi-precision arithmetic over 29-bit limbs.
//!
//! Limbs are stored in 32-bit words, and products of limbs are summed in 64-bit accumulators. A
//! limb product takes at most 58 bits, so an accumulator can absorb up to 64 of them before
//! overflowing, which allows multiplication algorithms to defer carry propagation. That makes this
//! representation a good fit for targets that lack a 64x64->128-bit multiplication and a carry
//! flag, such as WebAssembly.
//!
//! Additions and subtractions involve no products, so they process the limbs in 58-bit pairs to
//! halve the length of their carry chains.

/// The number of bits in a limb.
pub const LIMB_BITS: u32 = 29;

/// Selects the lower `LIMB_BITS` bits of a word.
pub const LIMB_MASK: u32 = (1 << LIMB_BITS) - 1;

/// The number of limbs needed to store a 256-bit value.
pub const LIMBS: usize = 9;

/// Returns the 58-bit value made of the limbs `i` and `i + 1`, or just limb `i` if it's the last.
#[inline(always)]
const fn load_pair(limbs: &[u32; LIMBS], i: usize) -> u64 {
    if i + 1 < LIMBS {
        limbs[i] as u64 | (limbs[i + 1] as u64) << LIMB_BITS
    } else {
        limbs[i] as u64
    }
}

/// Stores the lower 58 bits of `value` in the limbs `i` and `i + 1`, or the lower 29 bits in limb
/// `i` if it's the last.
#[inline(always)]
const fn store_pair(limbs: &mut [u32; LIMBS], i: usize, value: u64) {
    limbs[i] = value as u32 & LIMB_MASK;
    if i + 1 < LIMBS {
        limbs[i + 1] = (value >> LIMB_BITS) as u32 & LIMB_MASK;
    }
}

/// Adds two numbers and returns the (sum, carry) pair, where the carry is 0 or 1.
///
/// The limbs are added in pairs, so that the carry chain takes 5 steps rather than 9: a 58-bit pair
/// leaves enough room in a 64-bit word to extract the carry with a shift.
#[inline(always)]
pub const fn add_limbs(a: &[u32; LIMBS], b: &[u32; LIMBS]) -> ([u32; LIMBS], u32) {
    let mut sum = [0u32; LIMBS];
    let mut carry = 0;
    let mut i = 0;
    while i < LIMBS {
        let value = load_pair(a, i) + load_pair(b, i) + carry;
        store_pair(&mut sum, i, value);
        carry = value
            >> if i + 1 < LIMBS {
                2 * LIMB_BITS
            } else {
                LIMB_BITS
            };
        i += 2;
    }
    (sum, carry as u32)
}

/// Subtracts `b` from `a` and returns the (difference, borrow) pair, where the borrow is 0 or 1.
///
/// If the borrow is 1 the difference has wrapped around 2^261. Like [`add_limbs`], this function
/// processes the limbs in pairs to shorten the borrow chain.
#[inline(always)]
pub const fn sub_limbs(a: &[u32; LIMBS], b: &[u32; LIMBS]) -> ([u32; LIMBS], u32) {
    let mut difference = [0u32; LIMBS];
    let mut borrow = 0;
    let mut i = 0;
    while i < LIMBS {
        let value = load_pair(a, i)
            .wrapping_sub(load_pair(b, i))
            .wrapping_sub(borrow);
        store_pair(&mut difference, i, value);
        borrow = value >> 63;
        i += 2;
    }
    (difference, borrow as u32)
}

/// Computes `a + b * c`, where `a` is an accumulator and `b` and `c` are limbs.
#[inline(always)]
pub const fn mac(a: u64, b: u32, c: u32) -> u64 {
    a + (b as u64) * (c as u64)
}

/// Splits an accumulator into the (limb, carry) pair, where the limb is made of the lower
/// `LIMB_BITS` bits and the carry of the remaining ones.
#[inline(always)]
pub const fn split(value: u64) -> (u32, u64) {
    (value as u32 & LIMB_MASK, value >> LIMB_BITS)
}

/// Converts a 256-bit value from four 64-bit words to nine limbs, both in little-endian order.
pub const fn from_words(words: [u64; 4]) -> [u32; LIMBS] {
    let mut limbs = [0u32; LIMBS];
    let mut i = 0;
    while i < LIMBS {
        let offset = i * LIMB_BITS as usize;
        let word_index = offset / 64;
        let shift = offset % 64;
        let mut value = words[word_index] >> shift;
        if shift + LIMB_BITS as usize > 64 && word_index + 1 < 4 {
            value |= words[word_index + 1] << (64 - shift);
        }
        limbs[i] = value as u32 & LIMB_MASK;
        i += 1;
    }
    limbs
}

/// Converts nine limbs to four 64-bit words, both in little-endian order.
///
/// Bits beyond the 256th are discarded.
pub const fn to_words(limbs: [u32; LIMBS]) -> [u64; 4] {
    let mut words = [0u64; 4];
    let mut i = 0;
    while i < LIMBS {
        let offset = i * LIMB_BITS as usize;
        let word_index = offset / 64;
        let shift = offset % 64;
        words[word_index] |= (limbs[i] as u64) << shift;
        if shift + LIMB_BITS as usize > 64 && word_index + 1 < 4 {
            words[word_index + 1] |= (limbs[i] as u64) >> (64 - shift);
        }
        i += 1;
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_limbs() {
        assert_eq!(add_limbs(&[0; LIMBS], &[0; LIMBS]), ([0; LIMBS], 0));
        assert_eq!(
            add_limbs(&[1, 0, 0, 0, 0, 0, 0, 0, 0], &[2, 0, 0, 0, 0, 0, 0, 0, 0]),
            ([3, 0, 0, 0, 0, 0, 0, 0, 0], 0)
        );
        assert_eq!(
            add_limbs(
                &[LIMB_MASK, 0, 0, 0, 0, 0, 0, 0, 0],
                &[1, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
            ([0, 1, 0, 0, 0, 0, 0, 0, 0], 0)
        );
        assert_eq!(
            add_limbs(
                &[LIMB_MASK, LIMB_MASK, 0, 0, 0, 0, 0, 0, 0],
                &[1, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
            ([0, 0, 1, 0, 0, 0, 0, 0, 0], 0)
        );
        assert_eq!(
            add_limbs(&[LIMB_MASK; LIMBS], &[1, 0, 0, 0, 0, 0, 0, 0, 0]),
            ([0; LIMBS], 1)
        );
        assert_eq!(
            add_limbs(&[LIMB_MASK; LIMBS], &[LIMB_MASK; LIMBS]),
            (
                [
                    LIMB_MASK - 1,
                    LIMB_MASK,
                    LIMB_MASK,
                    LIMB_MASK,
                    LIMB_MASK,
                    LIMB_MASK,
                    LIMB_MASK,
                    LIMB_MASK,
                    LIMB_MASK
                ],
                1
            )
        );
    }

    #[test]
    fn test_add_limbs_modulus() {
        assert_eq!(
            add_limbs(
                &[
                    0x00000001, 0x00000000, 0x06d551f0, 0x07bbe53d, 0x1fffe067, 0x1fffffff,
                    0x1fffffff, 0x1fffffff, 0x007fffff,
                ],
                &[
                    0x00000001, 0x00000000, 0x06d551f0, 0x07bbe53d, 0x1fffe067, 0x1fffffff,
                    0x1fffffff, 0x1fffffff, 0x007fffff,
                ]
            ),
            (
                [
                    0x00000002, 0x00000000, 0x0daaa3e0, 0x0f77ca7a, 0x1fffc0ce, 0x1fffffff,
                    0x1fffffff, 0x1fffffff, 0x00ffffff,
                ],
                0
            )
        );
    }

    #[test]
    fn test_sub_limbs() {
        assert_eq!(sub_limbs(&[0; LIMBS], &[0; LIMBS]), ([0; LIMBS], 0));
        assert_eq!(
            sub_limbs(&[3, 0, 0, 0, 0, 0, 0, 0, 0], &[2, 0, 0, 0, 0, 0, 0, 0, 0]),
            ([1, 0, 0, 0, 0, 0, 0, 0, 0], 0)
        );
        assert_eq!(
            sub_limbs(&[0, 1, 0, 0, 0, 0, 0, 0, 0], &[1, 0, 0, 0, 0, 0, 0, 0, 0]),
            ([LIMB_MASK, 0, 0, 0, 0, 0, 0, 0, 0], 0)
        );
        assert_eq!(
            sub_limbs(&[0, 0, 1, 0, 0, 0, 0, 0, 0], &[1, 0, 0, 0, 0, 0, 0, 0, 0]),
            ([LIMB_MASK, LIMB_MASK, 0, 0, 0, 0, 0, 0, 0], 0)
        );
        assert_eq!(
            sub_limbs(&[0; LIMBS], &[1, 0, 0, 0, 0, 0, 0, 0, 0]),
            ([LIMB_MASK; LIMBS], 1)
        );
        assert_eq!(
            sub_limbs(&[LIMB_MASK; LIMBS], &[LIMB_MASK; LIMBS]),
            ([0; LIMBS], 0)
        );
    }

    #[test]
    fn test_sub_limbs_modulus() {
        assert_eq!(
            sub_limbs(
                &[
                    0x00000001, 0x00000000, 0x06d551f0, 0x07bbe53d, 0x1fffe067, 0x1fffffff,
                    0x1fffffff, 0x1fffffff, 0x007fffff,
                ],
                &[1, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
            (
                [
                    0x00000000, 0x00000000, 0x06d551f0, 0x07bbe53d, 0x1fffe067, 0x1fffffff,
                    0x1fffffff, 0x1fffffff, 0x007fffff,
                ],
                0
            )
        );
        assert_eq!(
            sub_limbs(
                &[0; LIMBS],
                &[
                    0x00000001, 0x00000000, 0x06d551f0, 0x07bbe53d, 0x1fffe067, 0x1fffffff,
                    0x1fffffff, 0x1fffffff, 0x007fffff,
                ]
            ),
            (
                [
                    0x1fffffff, 0x1fffffff, 0x192aae0f, 0x18441ac2, 0x00001f98, 0x00000000,
                    0x00000000, 0x00000000, 0x1f800000,
                ],
                1
            )
        );
    }

    #[test]
    fn test_mac() {
        assert_eq!(mac(0, 0, 0), 0);
        assert_eq!(mac(10, 0, 3), 10);
        assert_eq!(mac(0, 2, 3), 6);
        assert_eq!(mac(10, 2, 3), 16);
        assert_eq!(mac(10, LIMB_MASK, LIMB_MASK), 0x03ffffffc000000b);
    }

    #[test]
    fn test_mac_headroom() {
        assert_eq!(
            mac(0xfbfffff04000003f, LIMB_MASK, LIMB_MASK),
            0xfffffff000000040
        );
    }

    #[test]
    fn test_split() {
        assert_eq!(split(0), (0, 0));
        assert_eq!(split(1), (1, 0));
        assert_eq!(split(0x1fffffff), (LIMB_MASK, 0));
        assert_eq!(split(0x20000000), (0, 1));
        assert_eq!(split(0x20000005), (5, 1));
        assert_eq!(split(0xfffffff000000040), (0x40, 0x7ffffff80));
    }

    #[test]
    fn test_from_words() {
        assert_eq!(from_words([0, 0, 0, 0]), [0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(from_words([1, 0, 0, 0]), [1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            from_words([
                0x8796a5b4c3d2e1f0,
                0x0f1e2d3c4b5a6978,
                0xfedcba9876543210,
                0x0123456789abcdef,
            ]),
            [
                0x03d2e1f0, 0x1cb52da6, 0x169a5e21, 0x1c5a7896, 0x032100f1, 0x1d4c3b2a, 0x17bffb72,
                0x0cf13579, 0x00012345,
            ]
        );
        assert_eq!(
            from_words([
                0xc000000000000001,
                0x0673ddf29e9b5547,
                0xfffffffffffffffe,
                0x7fffffffffffffff,
            ]),
            [
                0x00000001, 0x00000000, 0x06d551f0, 0x07bbe53d, 0x1fffe067, 0x1fffffff, 0x1fffffff,
                0x1fffffff, 0x007fffff,
            ]
        );
        assert_eq!(
            from_words([u64::MAX, u64::MAX, u64::MAX, u64::MAX]),
            [
                0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff,
                0x1fffffff, 0x00ffffff,
            ]
        );
    }

    #[test]
    fn test_to_words() {
        assert_eq!(to_words([0, 0, 0, 0, 0, 0, 0, 0, 0]), [0, 0, 0, 0]);
        assert_eq!(to_words([1, 0, 0, 0, 0, 0, 0, 0, 0]), [1, 0, 0, 0]);
        assert_eq!(
            to_words([
                0x03d2e1f0, 0x1cb52da6, 0x169a5e21, 0x1c5a7896, 0x032100f1, 0x1d4c3b2a, 0x17bffb72,
                0x0cf13579, 0x00012345,
            ]),
            [
                0x8796a5b4c3d2e1f0,
                0x0f1e2d3c4b5a6978,
                0xfedcba9876543210,
                0x0123456789abcdef,
            ]
        );
        assert_eq!(
            to_words([
                0x00000001, 0x00000000, 0x06d551f0, 0x07bbe53d, 0x1fffe067, 0x1fffffff, 0x1fffffff,
                0x1fffffff, 0x007fffff,
            ]),
            [
                0xc000000000000001,
                0x0673ddf29e9b5547,
                0xfffffffffffffffe,
                0x7fffffffffffffff,
            ]
        );
        assert_eq!(
            to_words([
                0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff,
                0x1fffffff, 0x00ffffff,
            ]),
            [u64::MAX, u64::MAX, u64::MAX, u64::MAX]
        );
    }

    #[test]
    fn test_to_words_discards_high_bits() {
        assert_eq!(
            to_words([LIMB_MASK, 0, 0, 0, 0, 0, 0, 0, LIMB_MASK]),
            [0x000000001fffffff, 0, 0, 0xffffff0000000000]
        );
    }
}
