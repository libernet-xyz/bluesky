//! Helpers for multi-precision arithmetic over 29-bit limbs.
//!
//! Limbs are stored in `u32`s, and products of limbs are summed in `u64` accumulators. A limb
//! product takes at most 58 bits, so an accumulator can absorb up to 64 of them before overflowing,
//! which allows multiplication algorithms to defer carry propagation. That makes this
//! representation a good fit for targets that lack a 64x64->128-bit multiplication and a carry
//! flag, such as WebAssembly.
//!
//! Additions and subtractions involve no products, so they can process the limbs in 58-bit pairs
//! (see [`join_pair`]) to halve the length of their carry chains.

/// The number of bits in a limb.
pub const LIMB_BITS: u32 = 29;

/// Selects the lower `LIMB_BITS` bits of a `u32`.
pub const LIMB_MASK: u32 = (1 << LIMB_BITS) - 1;

/// The number of limbs needed to store a 256-bit value.
pub const LIMBS: usize = 9;

/// The number of bits in a pair of limbs.
pub const PAIR_BITS: u32 = 2 * LIMB_BITS;

/// Joins two limbs into a 58-bit pair.
#[inline(always)]
pub const fn join_pair(low: u32, high: u32) -> u64 {
    low as u64 | (high as u64) << LIMB_BITS
}

/// Returns the lower limb of a pair, ie. its bits from 0 to 28.
#[inline(always)]
pub const fn low_limb(pair: u64) -> u32 {
    pair as u32 & LIMB_MASK
}

/// Returns the higher limb of a pair, ie. its bits from 29 to 57.
#[inline(always)]
pub const fn high_limb(pair: u64) -> u32 {
    (pair >> LIMB_BITS) as u32 & LIMB_MASK
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_pair() {
        assert_eq!(join_pair(0, 0), 0);
        assert_eq!(join_pair(1, 0), 1);
        assert_eq!(join_pair(0, 1), 0x20000000);
        assert_eq!(join_pair(0x12345678, 0x1abcdef0), 0x03579bde12345678);
        assert_eq!(join_pair(LIMB_MASK, LIMB_MASK), 0x03ffffffffffffff);
    }

    #[test]
    fn test_low_limb() {
        assert_eq!(low_limb(0), 0);
        assert_eq!(low_limb(0x20000000), 0);
        assert_eq!(low_limb(0x03579bde12345678), 0x12345678);
        assert_eq!(low_limb(0x03ffffffffffffff), LIMB_MASK);
        assert_eq!(low_limb(u64::MAX), LIMB_MASK);
    }

    #[test]
    fn test_high_limb() {
        assert_eq!(high_limb(0), 0);
        assert_eq!(high_limb(0x1fffffff), 0);
        assert_eq!(high_limb(0x20000000), 1);
        assert_eq!(high_limb(0x03579bde12345678), 0x1abcdef0);
        assert_eq!(high_limb(0x03ffffffffffffff), LIMB_MASK);
        assert_eq!(high_limb(u64::MAX), LIMB_MASK);
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
}
