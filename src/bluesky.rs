use anyhow::{Context, anyhow};
use ff::{
    Field, PrimeField,
    derive::{adc, mac, sbb},
};
use primitive_types::{U256, U512};
use rand_core::TryRng;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str::FromStr;
use std::sync::LazyLock;
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
    CtOption,
};

#[inline(always)]
const fn add(a: u64, b: u64) -> (u64, u64) {
    let sum = (a as u128) + (b as u128);
    (sum as u64, (sum >> 64) as u64)
}

#[inline(always)]
const fn sub(a: u64, b: u64) -> (u64, u64) {
    let ret = (a as u128).wrapping_sub(b as u128);
    (ret as u64, (ret >> 64) as u64)
}

#[inline(always)]
const fn mul(lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
    let product = (lhs as u128) * (rhs as u128) + carry as u128;
    (product as u64, (product >> 64) as u64)
}

/// Describes a prime field with a (3^T)-th root of unity.
pub trait ThreeAdicField: PrimeField {
    /// The 3-adicity of the field.
    const T: u32;

    /// Inverse of 3 in the field.
    const THREE_INV: Self;

    /// The primitive 3-adic root of unity, a number w such that w^(3^T) = 1.
    const THREE_ADIC_ROOT_OF_UNITY: Self;

    /// The inverse of the root of unity.
    const THREE_ADIC_ROOT_OF_UNITY_INV: Self;
}

/// The prime order of the BlueSky field stored as four 64-bit limbs in little endian order.
pub const MODULUS: [u64; 4] = [
    0xc000000000000001u64,
    0x0673ddf29e9b5547u64,
    0xfffffffffffffffeu64,
    0x7fffffffffffffffu64,
];

/// A scalar over the BlueSky prime field.
///
/// The prime order of the field is:
///
///   p = 0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000001
///
/// This field is well-suited for use in both binary and ternary FRI because it has a large 2- and
/// 3-adicity: p-1 is divided by both 2^62 and 3^39, supporting polynomials of extremely high
/// degree.
///
/// All our scalars are stored in Montgomery form with the four limbs stored in little-endian order.
#[derive(Default, Copy, Clone, PartialEq, Eq)]
pub struct Scalar(u64, u64, u64, u64);

impl Scalar {
    /// The largest value representable in the field, ie. p-1.
    pub const MAX: Self = Self(
        0x4000000000000003u64,
        0x135b99d7dbd1ffd7u64,
        0xfffffffffffffffau64,
        0x7fffffffffffffffu64,
    );

    /// The raw (non-Montgomery) little-endian representation of `MAX`.
    const MAX_RAW: Self = Self(
        0xc000000000000000u64,
        0x0673ddf29e9b5547u64,
        0xfffffffffffffffeu64,
        0x7fffffffffffffffu64,
    );

    /// `MAX` minus one, ie. p-2.
    ///
    /// By Fermat's little theorem, exponentiating a non-null scalar by this number yields the
    /// modular inverse of that scalar.
    pub const MAX_MINUS_ONE: Self = Self(
        0xc000000000000005u64,
        0x204355bd1908aa66u64,
        0xfffffffffffffff6u64,
        0x7fffffffffffffffu64,
    );

    /// The raw (non-Montgomery) little-endian representation of `MAX_MINUS_ONE`.
    const MAX_MINUS_ONE_RAW: [u64; 4] = [
        0xbfffffffffffffffu64,
        0x0673ddf29e9b5547u64,
        0xfffffffffffffffeu64,
        0x7fffffffffffffffu64,
    ];

    /// R in Montgomery form, ie. R^2 mod p.
    pub const R: Self = Self(
        0xbfffffffffffffe5u64,
        0xab970f33c00b568du64,
        0xb296e2afc92ce69du64,
        0x1968ac3835a4f8ddu64,
    );

    const P: [u64; 4] = MODULUS;

    const P_INV: u64 = 0xbfffffffffffffffu64;

    const TM1D2: [u64; 4] = [
        0x0ce7bbe53d36aa8fu64,
        0xfffffffffffffffcu64,
        0xffffffffffffffffu64,
        0x0000000000000000u64,
    ];

    /// Subtracts p. Assumes no underflow, ie. `self` must be greater than or equal to p.
    ///
    /// Used in several algorithms to bring a value back into the [0, p) range.
    const fn subp(&self) -> Self {
        let (s0, b0) = sub(self.0, Self::P[0]);
        let (s1, b1) = sbb(self.1, Self::P[1], b0);
        let (s2, b2) = sbb(self.2, Self::P[2], b1);
        let (s3, _) = sbb(self.3, Self::P[3], b2);
        Self(s0, s1, s2, s3)
    }

    /// Compares raw scalars, ignoring Montgomery form.
    const fn cmp_raw(&self, other: &Self) -> Ordering {
        if self.3 < other.3 {
            Ordering::Less
        } else if self.3 > other.3 {
            Ordering::Greater
        } else if self.2 < other.2 {
            Ordering::Less
        } else if self.2 > other.2 {
            Ordering::Greater
        } else if self.1 < other.1 {
            Ordering::Less
        } else if self.1 > other.1 {
            Ordering::Greater
        } else if self.0 < other.0 {
            Ordering::Less
        } else if self.0 > other.0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Performs Montgomery multiplication using CIOS over 64-bit limbs.
    const fn mont_mul(lhs: &Self, rhs: &Self) -> Self {
        let mut t0: u64;
        let mut t1: u64;
        let mut t2: u64;
        let mut t3: u64;
        let mut t4: u64;
        let mut carry: u64;
        let mut m: u64;

        // row 0
        (t0, carry) = mul(lhs.0, rhs.0, 0);
        (t1, carry) = mul(lhs.1, rhs.0, carry);
        (t2, carry) = mul(lhs.2, rhs.0, carry);
        (t3, t4) = mul(lhs.3, rhs.0, carry);

        // redc 0
        m = t0.wrapping_mul(Self::P_INV);
        (_, carry) = mac(t0, m, Self::P[0], 0);
        (t0, carry) = mac(t1, m, Self::P[1], carry);
        (t1, carry) = mac(t2, m, Self::P[2], carry);
        (t2, carry) = mac(t3, m, Self::P[3], carry);
        t3 = t4 + carry;

        // row 1
        (t0, carry) = mac(t0, lhs.0, rhs.1, 0);
        (t1, carry) = mac(t1, lhs.1, rhs.1, carry);
        (t2, carry) = mac(t2, lhs.2, rhs.1, carry);
        (t3, t4) = mac(t3, lhs.3, rhs.1, carry);

        // redc 1
        m = t0.wrapping_mul(Self::P_INV);
        (_, carry) = mac(t0, m, Self::P[0], 0);
        (t0, carry) = mac(t1, m, Self::P[1], carry);
        (t1, carry) = mac(t2, m, Self::P[2], carry);
        (t2, carry) = mac(t3, m, Self::P[3], carry);
        t3 = t4 + carry;

        // row 2
        (t0, carry) = mac(t0, lhs.0, rhs.2, 0);
        (t1, carry) = mac(t1, lhs.1, rhs.2, carry);
        (t2, carry) = mac(t2, lhs.2, rhs.2, carry);
        (t3, t4) = mac(t3, lhs.3, rhs.2, carry);

        // redc 2
        m = t0.wrapping_mul(Self::P_INV);
        (_, carry) = mac(t0, m, Self::P[0], 0);
        (t0, carry) = mac(t1, m, Self::P[1], carry);
        (t1, carry) = mac(t2, m, Self::P[2], carry);
        (t2, carry) = mac(t3, m, Self::P[3], carry);
        t3 = t4 + carry;

        // row 3
        (t0, carry) = mac(t0, lhs.0, rhs.3, 0);
        (t1, carry) = mac(t1, lhs.1, rhs.3, carry);
        (t2, carry) = mac(t2, lhs.2, rhs.3, carry);
        (t3, t4) = mac(t3, lhs.3, rhs.3, carry);

        // redc 3
        m = t0.wrapping_mul(Self::P_INV);
        (_, carry) = mac(t0, m, Self::P[0], 0);
        (t0, carry) = mac(t1, m, Self::P[1], carry);
        (t1, carry) = mac(t2, m, Self::P[2], carry);
        (t2, carry) = mac(t3, m, Self::P[3], carry);
        t3 = t4 + carry;

        let result = Self(t0, t1, t2, t3);
        match result.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => result.subp(),
            _ => result,
        }
    }

    /// Performs a Montgomery multiplication by 1, which results in converting from Montgomery form
    /// to raw form.
    ///
    /// This is exactly the same as `mont_mul(Scalar(1, 0, 0, 0))` but slightly faster because it
    /// exploits the fact that we're multiplying by (1, 0, 0, 0), so it skips all "row" phases and
    /// only performs the "redc" phases.
    const fn to_raw(&self) -> Self {
        let mut t0 = self.0;
        let mut t1 = self.1;
        let mut t2 = self.2;
        let mut t3 = self.3;
        let mut carry: u64;
        let mut m: u64;

        // redc 0
        m = t0.wrapping_mul(Self::P_INV);
        (_, carry) = mac(t0, m, Self::P[0], 0);
        (t0, carry) = mac(t1, m, Self::P[1], carry);
        (t1, carry) = mac(t2, m, Self::P[2], carry);
        (t2, carry) = mac(t3, m, Self::P[3], carry);
        t3 = carry;

        // redc 1
        m = t0.wrapping_mul(Self::P_INV);
        (_, carry) = mac(t0, m, Self::P[0], 0);
        (t0, carry) = mac(t1, m, Self::P[1], carry);
        (t1, carry) = mac(t2, m, Self::P[2], carry);
        (t2, carry) = mac(t3, m, Self::P[3], carry);
        t3 = carry;

        // redc 2
        m = t0.wrapping_mul(Self::P_INV);
        (_, carry) = mac(t0, m, Self::P[0], 0);
        (t0, carry) = mac(t1, m, Self::P[1], carry);
        (t1, carry) = mac(t2, m, Self::P[2], carry);
        (t2, carry) = mac(t3, m, Self::P[3], carry);
        t3 = carry;

        // redc 3
        m = t0.wrapping_mul(Self::P_INV);
        (_, carry) = mac(t0, m, Self::P[0], 0);
        (t0, carry) = mac(t1, m, Self::P[1], carry);
        (t1, carry) = mac(t2, m, Self::P[2], carry);
        (t2, carry) = mac(t3, m, Self::P[3], carry);
        t3 = carry;

        let result = Self(t0, t1, t2, t3);
        match result.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => result.subp(),
            _ => result,
        }
    }

    /// Constructs scalars at compile time.
    pub const fn from_const(value: u64) -> Scalar {
        let raw = Self(value, 0, 0, 0);
        Self::mont_mul(&raw, &Self::R)
    }

    /// Constructs a scalar from the given little-endian byte representation, returning `None` if
    /// the resulting value lies outside the field.
    ///
    /// NOTE: the length of `repr` MUST be 32.
    pub fn from_repr_vartime(repr: &[u8]) -> Option<Self> {
        let value = Self(
            u64::from_le_bytes(repr[0..8].try_into().unwrap()),
            u64::from_le_bytes(repr[8..16].try_into().unwrap()),
            u64::from_le_bytes(repr[16..24].try_into().unwrap()),
            u64::from_le_bytes(repr[24..32].try_into().unwrap()),
        );
        match value.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => None,
            _ => Some(Self::mont_mul(&value, &Self::R)),
        }
    }

    /// Constructs a scalar from the given little-endian byte representation, performing modular
    /// reduction if the resulting value lies outside the field.
    ///
    /// NOTE: the length of `repr` MUST be 32.
    pub fn from_repr_canonical(repr: &[u8]) -> Self {
        let mut value = Self(
            u64::from_le_bytes(repr[0..8].try_into().unwrap()),
            u64::from_le_bytes(repr[8..16].try_into().unwrap()),
            u64::from_le_bytes(repr[16..24].try_into().unwrap()),
            u64::from_le_bytes(repr[24..32].try_into().unwrap()),
        );
        if value.cmp_raw(&Self::MAX_RAW) == Ordering::Greater {
            value = value.subp();
            if value.cmp_raw(&Self::MAX_RAW) == Ordering::Greater {
                value = value.subp();
            }
        }
        Self::mont_mul(&value, &Self::R)
    }

    /// Constructs a scalar from the given little-endian byte representation of a 512-bit value,
    /// performing modular reduction to bring the value back into the BlueSky field.
    ///
    /// NOTE: the length of `repr` MUST be 64.
    pub fn from_repr_wide(repr: &[u8]) -> Self {
        static MODULUS: LazyLock<U512> = LazyLock::new(|| Scalar::MODULUS.parse().unwrap());
        let value = U512::from_little_endian(&repr);
        let repr = (value % *MODULUS).to_little_endian();
        Self::from_repr_vartime(&repr[0..32]).unwrap()
    }

    /// Constructs a scalar from the 4 64-bit limbs provided in little-endian order.
    pub fn from_le_u64_vartime(limbs: [u64; 4]) -> Option<Scalar> {
        let value = Self(limbs[0], limbs[1], limbs[2], limbs[3]);
        match value.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => None,
            _ => Some(Self::mont_mul(&value, &Self::R)),
        }
    }

    /// Constructs a scalar in constant time from the 4 64-bit limbs provided in little-endian
    /// order.
    pub fn from_le_u64(limbs: [u64; 4]) -> CtOption<Scalar> {
        let raw_value = Self(limbs[0], limbs[1], limbs[2], limbs[3]);
        let montgomery = Self::mont_mul(&raw_value, &Self::R);
        CtOption::new(montgomery, !raw_value.ct_gt(&Self::MAX_RAW))
    }

    /// Converts this scalar to its canonical integer representation as 4 little-endian 64-bit
    /// limbs.
    pub fn to_le_u64(&self) -> [u64; 4] {
        let raw = self.to_raw();
        [raw.0, raw.1, raw.2, raw.3]
    }

    /// Constructs a scalar in constant time from the given little-endian byte representation,
    /// returning `None` if the resulting value lies outside the field.
    ///
    /// REQUIRES: the length of `bytes` must be 32.
    pub fn from_little_endian(bytes: &[u8]) -> CtOption<Self> {
        assert!(bytes.len() == 32);
        let raw = Self(
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        );
        let value = Self::mont_mul(&raw, &Self::R);
        CtOption::new(value, !raw.ct_gt(&Self::MAX_RAW))
    }

    /// Converts this scalar to its canonical integer representation as a 32-byte little-endian
    /// array.
    pub fn to_little_endian(&self) -> [u8; 32] {
        let raw = self.to_raw();
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&raw.0.to_le_bytes());
        bytes[8..16].copy_from_slice(&raw.1.to_le_bytes());
        bytes[16..24].copy_from_slice(&raw.2.to_le_bytes());
        bytes[24..32].copy_from_slice(&raw.3.to_le_bytes());
        bytes
    }

    /// Constructs a scalar in constant time from the given big-endian byte representation,
    /// returning `None` if the resulting value lies outside the field.
    ///
    /// REQUIRES: the length of `bytes` must be 32.
    pub fn from_big_endian(bytes: &[u8]) -> CtOption<Self> {
        assert!(bytes.len() == 32);
        let raw = Self(
            u64::from_be_bytes(bytes[24..32].try_into().unwrap()),
            u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_be_bytes(bytes[0..8].try_into().unwrap()),
        );
        let value = Self::mont_mul(&raw, &Self::R);
        CtOption::new(value, !raw.ct_gt(&Self::MAX_RAW))
    }

    /// Converts this scalar to its canonical integer representation as a 32-byte big-endian
    /// array.
    pub fn to_big_endian(&self) -> [u8; 32] {
        let raw = self.to_raw();
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&raw.3.to_be_bytes());
        bytes[8..16].copy_from_slice(&raw.2.to_be_bytes());
        bytes[16..24].copy_from_slice(&raw.1.to_be_bytes());
        bytes[24..32].copy_from_slice(&raw.0.to_be_bytes());
        bytes
    }

    /// Converts this scalar to a [`U256`] integer.
    pub fn to_u256(&self) -> U256 {
        U256::from_little_endian(&self.to_repr())
    }

    /// Picks a uniformly distributed random scalar securely using the OS RNG.
    ///
    /// Relies on `getrandom::fill`.
    pub fn random_default() -> Scalar {
        let mut bytes = [0u8; 64];
        getrandom::fill(&mut bytes).unwrap();
        Self::from_repr_wide(&bytes)
    }

    fn parse(s: &str, radix: u8) -> Result<Self, anyhow::Error> {
        if s.is_empty() {
            return Err(anyhow!("cannot parse a scalar from an empty string"));
        }
        static CHARACTERS_UPPER_CASE: &'static [u8] = b"0123456789ABCDEF";
        static CHARACTERS_LOWER_CASE: &'static [u8] = b"0123456789abcdef";
        let mut value = U256::zero();
        for byte in s.bytes() {
            let digit = CHARACTERS_UPPER_CASE[..radix as usize]
                .iter()
                .position(|&c| c == byte)
                .or_else(|| {
                    CHARACTERS_LOWER_CASE[..radix as usize]
                        .iter()
                        .position(|&c| c == byte)
                })
                .ok_or_else(|| {
                    anyhow!("invalid character '{}' for base {}", byte as char, radix)
                })?;
            value = value
                .checked_mul(radix.into())
                .context("overflow")?
                .checked_add(digit.into())
                .context("overflow")?;
        }
        Scalar::try_from(value)
    }

    pub fn from_str_radix(s: &str, radix: u8) -> Result<Self, anyhow::Error> {
        match radix {
            2 | 8 | 10 | 16 => Self::parse(s, radix),
            _ => unimplemented!(),
        }
    }

    fn print_dec(&self, pad_to: usize) -> String {
        static CHARACTERS: &'static [u8] = b"0123456789";
        let mut value = self.to_u256();
        let mut s = String::default();
        let radix = U256::from(10);
        while !value.is_zero() {
            let digit = (value % radix).as_usize();
            s.push(CHARACTERS[digit] as char);
            value /= radix;
        }
        if s.is_empty() {
            s.push('0');
        }
        while s.len() < pad_to {
            s.push('0');
        }
        s.chars().rev().collect()
    }

    fn print_bin(&self, radix: u8, pad_to: usize, upper_case: bool) -> String {
        static CHARACTERS_UPPER_CASE: &'static [u8] = b"0123456789ABCDEF";
        static CHARACTERS_LOWER_CASE: &'static [u8] = b"0123456789abcdef";
        let characters = if upper_case {
            CHARACTERS_UPPER_CASE
        } else {
            CHARACTERS_LOWER_CASE
        };
        let mut value = self.to_u256();
        let mut s = String::default();
        let radix_log2: usize = match radix {
            2 => 1,
            8 => 3,
            16 => 4,
            _ => panic!(),
        };
        let mask = U256::from((1 << radix_log2) - 1);
        while !value.is_zero() {
            let digit = (value & mask).as_usize();
            s.push(characters[digit] as char);
            value >>= radix_log2;
        }
        if s.is_empty() {
            s.push('0');
        }
        while s.len() < pad_to {
            s.push('0');
        }
        s.chars().rev().collect()
    }

    pub fn to_str_radix(&self, radix: u8, pad_to: usize, upper_case: bool) -> String {
        match radix {
            10 => self.print_dec(pad_to),
            2 | 8 | 16 => self.print_bin(radix, pad_to, upper_case),
            _ => unimplemented!(),
        }
    }
}

impl Debug for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{}", self.to_str_radix(16, 64, false))
    }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{}", self.to_str_radix(16, 64, false))
    }
}

impl std::fmt::Binary for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0b{}", self.to_str_radix(2, 256, false))
    }
}

impl std::fmt::Octal for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0o{}", self.to_str_radix(8, 86, false))
    }
}

impl std::fmt::LowerHex for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{}", self.to_str_radix(16, 64, false))
    }
}

impl std::fmt::UpperHex for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{}", self.to_str_radix(16, 64, true))
    }
}

impl Ord for Scalar {
    fn cmp(&self, other: &Self) -> Ordering {
        let lhs = self.to_raw();
        let rhs = other.to_raw();
        lhs.cmp_raw(&rhs)
    }
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<u64> for Scalar {
    fn from(value: u64) -> Self {
        Self::mont_mul(&Self(value, 0, 0, 0), &Self::R)
    }
}

impl TryFrom<U256> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: U256) -> Result<Self, Self::Error> {
        Self::from_repr_vartime(&value.to_little_endian())
            .context("the provided value lies outside the BlueSky field")
    }
}

impl FromStr for Scalar {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with("0x") || s.starts_with("0X") {
            Scalar::from_str_radix(&s[2..], 16)
        } else if s.starts_with("0b") || s.starts_with("0B") {
            Scalar::from_str_radix(&s[2..], 2)
        } else if s.starts_with("0") || s.starts_with("0") {
            Scalar::from_str_radix(s, 8)
        } else {
            Scalar::from_str_radix(s, 10)
        }
    }
}

impl Add<&Scalar> for Scalar {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let (r0, c0) = add(self.0, rhs.0);
        let (r1, c1) = adc(self.1, rhs.1, c0);
        let (r2, c2) = adc(self.2, rhs.2, c1);
        let (r3, _) = adc(self.3, rhs.3, c2);
        let result = Self(r0, r1, r2, r3);
        match result.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => result.subp(),
            _ => result,
        }
    }
}

impl Add for Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl AddAssign<&Scalar> for Scalar {
    fn add_assign(&mut self, rhs: &Self) {
        *self = self.add(rhs);
    }
}

impl AddAssign for Scalar {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(&rhs);
    }
}

impl Sub<&Scalar> for Scalar {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        let (r0, b0) = sub(self.0, rhs.0);
        let (r1, b1) = sbb(self.1, rhs.1, b0);
        let (r2, b2) = sbb(self.2, rhs.2, b1);
        let (r3, b3) = sbb(self.3, rhs.3, b2);
        if b3 == 0 {
            return Self(r0, r1, r2, r3);
        }
        let (s0, c0) = add(r0, Self::P[0]);
        let (s1, c1) = adc(r1, Self::P[1], c0);
        let (s2, c2) = adc(r2, Self::P[2], c1);
        let (s3, _) = adc(r3, Self::P[3], c2);
        Self(s0, s1, s2, s3)
    }
}

impl Sub for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl SubAssign<&Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = self.sub(rhs);
    }
}

impl SubAssign for Scalar {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(&rhs);
    }
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero_vartime() {
            return self;
        }
        let (r0, b0) = sub(Self::P[0], self.0);
        let (r1, b1) = sbb(Self::P[1], self.1, b0);
        let (r2, b2) = sbb(Self::P[2], self.2, b1);
        let (r3, _) = sbb(Self::P[3], self.3, b2);
        Self(r0, r1, r2, r3)
    }
}

impl Mul<&Scalar> for Scalar {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        Self::mont_mul(&self, rhs)
    }
}

impl Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::mont_mul(&self, &rhs)
    }
}

impl MulAssign<&Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = Self::mont_mul(self, rhs);
    }
}

impl MulAssign for Scalar {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self::mont_mul(self, &rhs);
    }
}

impl Sum<Scalar> for Scalar {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a> Sum<&'a Scalar> for Scalar {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl Product<Scalar> for Scalar {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl<'a> Product<&'a Scalar> for Scalar {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl ConstantTimeEq for Scalar {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
            & self.1.ct_eq(&other.1)
            & self.2.ct_eq(&other.2)
            & self.3.ct_eq(&other.3)
    }
}

impl ConstantTimeGreater for Scalar {
    fn ct_gt(&self, other: &Self) -> Choice {
        let gt3 = self.3.ct_gt(&other.3);
        let gt2 = self.2.ct_gt(&other.2);
        let gt1 = self.1.ct_gt(&other.1);
        let gt0 = self.0.ct_gt(&other.0);
        let eq3 = self.3.ct_eq(&other.3);
        let eq2 = self.2.ct_eq(&other.2);
        let eq1 = self.1.ct_eq(&other.1);
        gt3 | eq3 & gt2 | eq3 & eq2 & gt1 | eq3 & eq2 & eq1 & gt0
    }
}

impl ConstantTimeLess for Scalar {}

impl ConditionallySelectable for Scalar {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Scalar(
            u64::conditional_select(&a.0, &b.0, choice),
            u64::conditional_select(&a.1, &b.1, choice),
            u64::conditional_select(&a.2, &b.2, choice),
            u64::conditional_select(&a.3, &b.3, choice),
        )
    }
}

impl Field for Scalar {
    const ZERO: Self = Self(0, 0, 0, 0);

    const ONE: Self = Self(
        0x7ffffffffffffffeu64,
        0xf318441ac2c95570u64,
        0x0000000000000003u64,
        0x0000000000000000u64,
    );

    fn try_random<R: TryRng + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        let mut bytes = [0u8; 64];
        rng.try_fill_bytes(&mut bytes)?;
        Ok(Self::from_repr_wide(&bytes))
    }

    fn square(&self) -> Self {
        Self::mont_mul(self, self)
    }

    fn double(&self) -> Self {
        let mut value = *self;
        value.3 = (value.3 << 1) | (value.2 >> 63);
        value.2 = (value.2 << 1) | (value.1 >> 63);
        value.1 = (value.1 << 1) | (value.0 >> 63);
        value.0 = value.0 << 1;
        match value.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => value.subp(),
            _ => value,
        }
    }

    fn invert(&self) -> CtOption<Self> {
        CtOption::new(self.pow(&Self::MAX_MINUS_ONE_RAW), !self.is_zero())
    }

    fn sqrt(&self) -> CtOption<Self> {
        ff::helpers::sqrt_tonelli_shanks(self, &Self::TM1D2)
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }
}

impl PrimeField for Scalar {
    type Repr = [u8; 32];

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        Self::from_little_endian(&repr)
    }

    fn to_repr(&self) -> Self::Repr {
        self.to_little_endian()
    }

    fn is_odd(&self) -> Choice {
        let raw = self.to_raw();
        Choice::from((raw.0 & 1) as u8)
    }

    const MODULUS: &'static str =
        "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000001";

    const NUM_BITS: u32 = 255;

    const CAPACITY: u32 = 254;

    const TWO_INV: Self = Self(
        0x3fffffffffffffffu64,
        0xf98c220d6164aab8u64,
        0x0000000000000001u64,
        0x0000000000000000u64,
    );

    const MULTIPLICATIVE_GENERATOR: Self = Self(
        0x7ffffffffffffff6u64,
        0xbf795485cdeeab32u64,
        0x0000000000000013u64,
        0x0000000000000000u64,
    );

    const S: u32 = 62;

    const ROOT_OF_UNITY: Self = Self(
        0x1c21299e7a6bf02cu64,
        0x9668eae30ea674fdu64,
        0x539332d8030750aau64,
        0x771ec3ece255e5ffu64,
    );

    const ROOT_OF_UNITY_INV: Self = Self(
        0x9d664a1b2af8b848u64,
        0x3eb41e848f20b29eu64,
        0x9ed9b5f1d9a9a30fu64,
        0x093d94fd0d3cc279u64,
    );

    const DELTA: Self = Self(
        0x0f81667ad386a65eu64,
        0xdaab9f0d961bebc7u64,
        0x1690cf9f051915cau64,
        0x0e26e41e50befeedu64,
    );
}

impl ThreeAdicField for Scalar {
    const T: u32 = 39;

    const THREE_INV: Self = Self(
        0x1555555555555555u64,
        0x532eb60475cc38e8u64,
        0xaaaaaaaaaaaaaaabu64,
        0x2aaaaaaaaaaaaaaau64,
    );

    const THREE_ADIC_ROOT_OF_UNITY: Self = Self(
        0x6bb97af29ca6dd9du64,
        0xa4274c2efcc4eac5u64,
        0x0e0ed5807a7ff8a3u64,
        0x41d8fc132eaa2afdu64,
    );

    const THREE_ADIC_ROOT_OF_UNITY_INV: Self = Self(
        0x35aa2ff058bc4166u64,
        0x359b814b693ec81eu64,
        0x7201659d84adb9e2u64,
        0x5587fd1dd27afee1u64,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format_scalar(x: Scalar) -> String {
        format!("{:#066x}", x)
    }

    fn parse_scalar(s: &str) -> Scalar {
        s.parse().unwrap()
    }

    #[test]
    fn test_print_constants() {
        assert_eq!(
            format_scalar(Scalar::ZERO),
            "0x0000000000000000000000000000000000000000000000000000000000000000",
        );
        assert_eq!(
            format_scalar(Scalar::ONE),
            "0x0000000000000000000000000000000000000000000000000000000000000001",
        );
        assert_eq!(
            format_scalar(Scalar::MAX),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000",
        );
        assert_eq!(
            format_scalar(Scalar::MAX_RAW * Scalar::R),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000",
        );
        assert_eq!(
            format_scalar(Scalar::R),
            "0x00000000000000000000000000000003f318441ac2c955707ffffffffffffffe"
        );
        assert_eq!(
            format_scalar(Scalar::MAX_MINUS_ONE),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547bfffffffffffffff"
        );
        assert_eq!(
            format_scalar(Scalar::TWO_INV),
            "0x3fffffffffffffffffffffffffffffff0339eef94f4daaa3e000000000000001"
        );
        assert_eq!(
            format_scalar(Scalar::MULTIPLICATIVE_GENERATOR),
            "0x0000000000000000000000000000000000000000000000000000000000000005"
        );
        assert_eq!(Scalar::S, 62);
        assert_eq!(
            format_scalar(Scalar::ROOT_OF_UNITY),
            "0x2772569d549e1249ca6891eceba43568f6e0a747a2afe898b3977ca1a5bbfc9c"
        );
        assert_eq!(
            format_scalar(Scalar::ROOT_OF_UNITY_INV),
            "0x76def406f046ef5ee7eeecd2c4e6ecd7cdedc4e2bcf6b19f1420121cd00b4cdb"
        );
        assert_eq!(
            format_scalar(Scalar::DELTA),
            "0x232680e97f4b6251c95edefd145053d6dea23905b503a1002987caddeb54cdce"
        );
        assert_eq!(Scalar::T, 39);
        assert_eq!(
            format_scalar(Scalar::THREE_ADIC_ROOT_OF_UNITY),
            "0x1b6292b3a6f8a32da98705fee2d66e8fe35f48642a417a72f1a4ca414adfd9e6"
        );
        assert_eq!(
            format_scalar(Scalar::THREE_ADIC_ROOT_OF_UNITY_INV),
            "0x4d369d8e1ff761fca979a0ffb5545471ceae8164a047dff8b5679e11c5dfdf72"
        );
    }

    #[test]
    fn test_constants() {
        assert_eq!(Scalar::ZERO, Scalar::default());
        assert_eq!(Scalar::ZERO, 0.into());
        assert_eq!(Scalar::ONE, 1.into());
        assert_eq!(Scalar::MAX, -Scalar::ONE);
        assert_eq!(Scalar::MAX_RAW * Scalar::R, -Scalar::ONE);
        assert_eq!(
            Scalar::R,
            Scalar::from_repr_wide(&[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ])
        );
        assert_eq!(Scalar::MAX_MINUS_ONE, -Scalar::from_const(2));
        assert_eq!(Scalar::NUM_BITS, 255);
        assert_eq!(Scalar::CAPACITY, 254);
        assert_eq!(Scalar::TWO_INV, Scalar::from_const(2).invert().unwrap());
        assert_eq!(Scalar::THREE_INV, Scalar::from_const(3).invert().unwrap());
        assert_eq!(Scalar::MULTIPLICATIVE_GENERATOR, 5.into());
        assert!(Scalar::ROOT_OF_UNITY > Scalar::ONE);
        for i in 0..Scalar::S {
            assert_ne!(
                Scalar::ROOT_OF_UNITY.pow_vartime(
                    Scalar::from_const(2)
                        .pow_vartime([i as u64, 0, 0, 0])
                        .to_le_u64()
                ),
                Scalar::ONE
            );
        }
        assert_eq!(
            Scalar::ROOT_OF_UNITY.pow_vartime(
                Scalar::from_const(2)
                    .pow_vartime([Scalar::S as u64, 0, 0, 0])
                    .to_le_u64()
            ),
            Scalar::ONE
        );
        assert_eq!(
            Scalar::ROOT_OF_UNITY * Scalar::ROOT_OF_UNITY_INV,
            Scalar::ONE
        );
        assert!(Scalar::THREE_ADIC_ROOT_OF_UNITY > Scalar::ONE);
        for i in 0..Scalar::T {
            assert_ne!(
                Scalar::THREE_ADIC_ROOT_OF_UNITY.pow_vartime(
                    Scalar::from_const(3)
                        .pow_vartime([i as u64, 0, 0, 0])
                        .to_le_u64()
                ),
                Scalar::ONE
            );
        }
        assert_eq!(
            Scalar::THREE_ADIC_ROOT_OF_UNITY.pow_vartime(
                Scalar::from_const(3)
                    .pow_vartime([Scalar::T as u64, 0, 0, 0])
                    .to_le_u64()
            ),
            Scalar::ONE
        );
        assert_eq!(
            Scalar::THREE_ADIC_ROOT_OF_UNITY * Scalar::THREE_ADIC_ROOT_OF_UNITY_INV,
            Scalar::ONE
        );
    }

    #[test]
    fn test_from_repr_vartime() {
        assert_eq!(
            format_scalar(
                Scalar::from_repr_vartime(&[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ])
                .unwrap()
            ),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_repr_vartime(&[
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ])
                .unwrap()
            ),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_repr_vartime(&[
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                ])
                .unwrap()
            ),
            "0x201f1e1d1c1b1a191817161514131211100f0e0d0c0b0a090807060504030201"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_repr_vartime(&[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 10, 150, 186, 185, 167, 254, 38,
                    41, 72, 122, 216, 10, 219, 186, 255, 255, 127
                ])
                .unwrap()
            ),
            "0x7fffffbadb0ad87a482926fea7b9ba960a300000000000000000000000000000"
        );
        assert!(
            Scalar::from_repr_vartime(&[
                1, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
            ])
            .is_none()
        );
    }

    #[test]
    fn test_from_repr_canonical() {
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ])),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ])),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32
            ])),
            "0x201f1e1d1c1b1a191817161514131211100f0e0d0c0b0a090807060504030201"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                255, 255, 255, 255, 255, 255, 255, 191, 71, 85, 155, 158, 242, 221, 115, 6, 254,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
            ])),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547bfffffffffffffff"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                0, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
            ])),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                1, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
            ])),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                2, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
            ])),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                1, 0, 0, 0, 0, 0, 0, 128, 143, 170, 54, 61, 229, 187, 231, 12, 252, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
            ])),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                2, 0, 0, 0, 0, 0, 0, 128, 143, 170, 54, 61, 229, 187, 231, 12, 252, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
            ])),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                3, 0, 0, 0, 0, 0, 0, 128, 143, 170, 54, 61, 229, 187, 231, 12, 252, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
            ])),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ])),
            "0x00000000000000000000000000000003f318441ac2c955707ffffffffffffffc"
        );
        assert_eq!(
            format_scalar(Scalar::from_repr_canonical(&[
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ])),
            "0x00000000000000000000000000000003f318441ac2c955707ffffffffffffffd"
        );
    }

    fn from_repr_wide(u512: U512) -> Scalar {
        Scalar::from_repr_wide(&u512.to_little_endian())
    }

    #[test]
    fn test_from_repr_wide() {
        assert_eq!(
            from_repr_wide("0x53acd3dc79d20203e9e60026cdb75d037f9cbb33eded8b767a8dfbee9bff090ecf26226e4d9d26b20b854b21b423ea28becd7445365e1fd349ed4af9c16baf97".parse().unwrap()),
            parse_scalar("0x11ae20d001e52b42470886f49871dc9c79c9021fd09a19eb527c213ab2f3db10"),
        );
        assert_eq!(
            from_repr_wide("0x76f63d96682cea5050cd80435b9c53c6b9298bb03e2fc5d726094917e80782c0f9cd0bc49eb092a199116130b24377ed6a5fe01bc95ce0a8cca77dbb1d10922b".parse().unwrap()),
            parse_scalar("0x45ce5572614df708a441e89f8982660fa7a904b470690ef76e4eb2a895402be7"),
        );
    }

    #[test]
    fn test_from_le_u64() {
        assert_eq!(
            Scalar::from_le_u64([1, 2, 3, 4]).unwrap(),
            parse_scalar("0x0000000000000004000000000000000300000000000000020000000000000001")
        );
        assert_eq!(
            Scalar::from_le_u64_vartime([1, 2, 3, 4]).unwrap(),
            parse_scalar("0x0000000000000004000000000000000300000000000000020000000000000001")
        );
        assert_eq!(
            Scalar::from_le_u64([
                0x12f64a812ff7b02eu64,
                0x1eaa2e32b4f74374u64,
                0x03dd6b282eece85bu64,
                0x6deb006ce96c1becu64,
            ])
            .unwrap(),
            parse_scalar("0x6deb006ce96c1bec03dd6b282eece85b1eaa2e32b4f7437412f64a812ff7b02e")
        );
        assert_eq!(
            Scalar::from_le_u64_vartime([
                0x12f64a812ff7b02eu64,
                0x1eaa2e32b4f74374u64,
                0x03dd6b282eece85bu64,
                0x6deb006ce96c1becu64,
            ])
            .unwrap(),
            parse_scalar("0x6deb006ce96c1bec03dd6b282eece85b1eaa2e32b4f7437412f64a812ff7b02e")
        );
        assert_eq!(
            Scalar::from_le_u64([
                0xc000000000000000u64,
                0x0673ddf29e9b5547u64,
                0xfffffffffffffffeu64,
                0x7fffffffffffffffu64,
            ])
            .unwrap(),
            parse_scalar("0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000")
        );
        assert_eq!(
            Scalar::from_le_u64_vartime([
                0xc000000000000000u64,
                0x0673ddf29e9b5547u64,
                0xfffffffffffffffeu64,
                0x7fffffffffffffffu64,
            ])
            .unwrap(),
            parse_scalar("0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000")
        );
        assert!(
            Scalar::from_le_u64([
                0xc000000000000001u64,
                0x0673ddf29e9b5547u64,
                0xfffffffffffffffeu64,
                0x7fffffffffffffffu64,
            ])
            .into_option()
            .is_none()
        );
        assert!(
            Scalar::from_le_u64_vartime([
                0xc000000000000001u64,
                0x0673ddf29e9b5547u64,
                0xfffffffffffffffeu64,
                0x7fffffffffffffffu64,
            ])
            .is_none()
        );
    }

    #[test]
    fn test_to_le_u64() {
        assert_eq!(
            parse_scalar("0x0000000000000004000000000000000300000000000000020000000000000001")
                .to_le_u64(),
            [1, 2, 3, 4]
        );
        assert_eq!(
            parse_scalar("0x6deb006ce96c1bec03dd6b282eece85b1eaa2e32b4f7437412f64a812ff7b02e")
                .to_le_u64(),
            [
                0x12f64a812ff7b02eu64,
                0x1eaa2e32b4f74374u64,
                0x03dd6b282eece85bu64,
                0x6deb006ce96c1becu64,
            ]
        );
        assert_eq!(
            parse_scalar("0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000")
                .to_le_u64(),
            [
                0xc000000000000000u64,
                0x0673ddf29e9b5547u64,
                0xfffffffffffffffeu64,
                0x7fffffffffffffffu64,
            ]
        );
    }

    #[test]
    fn test_from_u64() {
        assert_eq!(
            Scalar::from(0),
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000000")
        );
        assert_eq!(
            Scalar::from(1),
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000001")
        );
        assert_eq!(
            Scalar::from(2),
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000002")
        );
        assert_eq!(
            Scalar::from(42),
            parse_scalar("0x000000000000000000000000000000000000000000000000000000000000002a")
        );
        assert_eq!(
            Scalar::from(u64::MAX - 2),
            parse_scalar("0x000000000000000000000000000000000000000000000000fffffffffffffffd")
        );
        assert_eq!(
            Scalar::from(u64::MAX - 1),
            parse_scalar("0x000000000000000000000000000000000000000000000000fffffffffffffffe")
        );
        assert_eq!(
            Scalar::from(u64::MAX),
            parse_scalar("0x000000000000000000000000000000000000000000000000ffffffffffffffff")
        );
    }

    #[test]
    fn test_from_const() {
        assert_eq!(
            Scalar::from_const(0),
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000000")
        );
        assert_eq!(
            Scalar::from_const(1),
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000001")
        );
        assert_eq!(
            Scalar::from_const(2),
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000002")
        );
        assert_eq!(
            Scalar::from_const(42),
            parse_scalar("0x000000000000000000000000000000000000000000000000000000000000002a")
        );
        assert_eq!(
            Scalar::from_const(u64::MAX - 2),
            parse_scalar("0x000000000000000000000000000000000000000000000000fffffffffffffffd")
        );
        assert_eq!(
            Scalar::from_const(u64::MAX - 1),
            parse_scalar("0x000000000000000000000000000000000000000000000000fffffffffffffffe")
        );
        assert_eq!(
            Scalar::from_const(u64::MAX),
            parse_scalar("0x000000000000000000000000000000000000000000000000ffffffffffffffff")
        );
    }

    #[test]
    fn test_from_little_endian() {
        assert_eq!(
            format_scalar(
                Scalar::from_little_endian(&[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ])
                .unwrap()
            ),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_little_endian(&[
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ])
                .unwrap()
            ),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_little_endian(&[
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                ])
                .unwrap()
            ),
            "0x201f1e1d1c1b1a191817161514131211100f0e0d0c0b0a090807060504030201"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_little_endian(&[
                    0, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255,
                    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
                ])
                .unwrap()
            ),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
        );
        assert!(
            Scalar::from_little_endian(&[
                1, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
            ])
            .into_option()
            .is_none()
        );
    }

    #[test]
    fn test_to_little_endian() {
        assert_eq!(
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000000")
                .to_little_endian(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]
        );
        assert_eq!(
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000001")
                .to_little_endian(),
            [
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]
        );
        assert_eq!(
            parse_scalar("0x201f1e1d1c1b1a191817161514131211100f0e0d0c0b0a090807060504030201")
                .to_little_endian(),
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32
            ]
        );
        assert_eq!(
            parse_scalar("0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000")
                .to_little_endian(),
            [
                0, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
            ]
        );
    }

    #[test]
    fn test_from_big_endian() {
        assert_eq!(
            format_scalar(
                Scalar::from_big_endian(&[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ])
                .unwrap()
            ),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_big_endian(&[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1,
                ])
                .unwrap()
            ),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_big_endian(&[
                    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
                    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
                ])
                .unwrap()
            ),
            "0x201f1e1d1c1b1a191817161514131211100f0e0d0c0b0a090807060504030201"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_big_endian(&[
                    127, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254,
                    6, 115, 221, 242, 158, 155, 85, 71, 192, 0, 0, 0, 0, 0, 0, 0
                ])
                .unwrap()
            ),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
        );
        assert!(
            Scalar::from_big_endian(&[
                127, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 6,
                115, 221, 242, 158, 155, 85, 71, 192, 0, 0, 0, 0, 0, 0, 1
            ])
            .into_option()
            .is_none()
        );
    }

    #[test]
    fn test_to_big_endian() {
        assert_eq!(
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000000")
                .to_big_endian(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]
        );
        assert_eq!(
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000001")
                .to_big_endian(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1,
            ]
        );
        assert_eq!(
            parse_scalar("0x201f1e1d1c1b1a191817161514131211100f0e0d0c0b0a090807060504030201")
                .to_big_endian(),
            [
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
            ]
        );
        assert_eq!(
            parse_scalar("0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000")
                .to_big_endian(),
            [
                127, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 6,
                115, 221, 242, 158, 155, 85, 71, 192, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    #[test]
    fn test_try_from_u256() {
        assert_eq!(
            Scalar::try_from(
                "0x0000000000000004000000000000000300000000000000020000000000000001"
                    .parse::<U256>()
                    .unwrap()
            )
            .unwrap(),
            parse_scalar("0x0000000000000004000000000000000300000000000000020000000000000001")
        );
        assert_eq!(
            Scalar::try_from(
                "0x6deb006ce96c1bec03dd6b282eece85b1eaa2e32b4f7437412f64a812ff7b02e"
                    .parse::<U256>()
                    .unwrap()
            )
            .unwrap(),
            parse_scalar("0x6deb006ce96c1bec03dd6b282eece85b1eaa2e32b4f7437412f64a812ff7b02e")
        );
        assert_eq!(
            Scalar::try_from(
                "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
                    .parse::<U256>()
                    .unwrap()
            )
            .unwrap(),
            parse_scalar("0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000")
        );
        assert!(
            Scalar::try_from(
                "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000001"
                    .parse::<U256>()
                    .unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn test_to_u256() {
        assert_eq!(
            parse_scalar("0x0000000000000004000000000000000300000000000000020000000000000001")
                .to_u256(),
            "0x0000000000000004000000000000000300000000000000020000000000000001"
                .parse()
                .unwrap()
        );
        assert_eq!(
            parse_scalar("0x6deb006ce96c1bec03dd6b282eece85b1eaa2e32b4f7437412f64a812ff7b02e")
                .to_u256(),
            "0x6deb006ce96c1bec03dd6b282eece85b1eaa2e32b4f7437412f64a812ff7b02e"
                .parse()
                .unwrap()
        );
        assert_eq!(
            parse_scalar("0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000")
                .to_u256(),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
                .parse()
                .unwrap()
        );
    }

    #[test]
    fn test_from_repr() {
        assert_eq!(
            format_scalar(
                Scalar::from_repr([
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ])
                .unwrap()
            ),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_repr([
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ])
                .unwrap()
            ),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_repr([
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                ])
                .unwrap()
            ),
            "0x201f1e1d1c1b1a191817161514131211100f0e0d0c0b0a090807060504030201"
        );
        assert_eq!(
            format_scalar(
                Scalar::from_repr([
                    0, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255,
                    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
                ])
                .unwrap()
            ),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
        );
        assert!(
            Scalar::from_repr([
                1, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
            ])
            .into_option()
            .is_none()
        );
    }

    #[test]
    fn test_to_repr() {
        assert_eq!(
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000000")
                .to_repr(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]
        );
        assert_eq!(
            parse_scalar("0x0000000000000000000000000000000000000000000000000000000000000001")
                .to_repr(),
            [
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]
        );
        assert_eq!(
            parse_scalar("0x201f1e1d1c1b1a191817161514131211100f0e0d0c0b0a090807060504030201")
                .to_repr(),
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32
            ]
        );
        assert_eq!(
            parse_scalar("0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000")
                .to_repr(),
            [
                0, 0, 0, 0, 0, 0, 0, 192, 71, 85, 155, 158, 242, 221, 115, 6, 254, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127
            ]
        );
    }

    #[test]
    fn test_cmp() {
        let v0 = Scalar::from_const(0);
        let v1 = Scalar::from_const(1);
        let v2 = Scalar::from_const(42);
        let v3 = parse_scalar("0x318c1df8459d125dc54e1fe487bf23e8430221b69660d8ca9427235713f24de1");
        let v4 = Scalar::MAX_MINUS_ONE;
        let v5 = Scalar::MAX;

        assert_eq!(v0.cmp(&v0), Ordering::Equal);
        assert_eq!(v0.cmp(&v1), Ordering::Less);
        assert_eq!(v0.cmp(&v2), Ordering::Less);
        assert_eq!(v0.cmp(&v3), Ordering::Less);
        assert_eq!(v0.cmp(&v4), Ordering::Less);
        assert_eq!(v0.cmp(&v5), Ordering::Less);

        assert_eq!(v1.cmp(&v0), Ordering::Greater);
        assert_eq!(v1.cmp(&v1), Ordering::Equal);
        assert_eq!(v1.cmp(&v2), Ordering::Less);
        assert_eq!(v1.cmp(&v3), Ordering::Less);
        assert_eq!(v1.cmp(&v4), Ordering::Less);
        assert_eq!(v1.cmp(&v5), Ordering::Less);

        assert_eq!(v2.cmp(&v0), Ordering::Greater);
        assert_eq!(v2.cmp(&v1), Ordering::Greater);
        assert_eq!(v2.cmp(&v2), Ordering::Equal);
        assert_eq!(v2.cmp(&v3), Ordering::Less);
        assert_eq!(v2.cmp(&v4), Ordering::Less);
        assert_eq!(v2.cmp(&v5), Ordering::Less);

        assert_eq!(v3.cmp(&v0), Ordering::Greater);
        assert_eq!(v3.cmp(&v1), Ordering::Greater);
        assert_eq!(v3.cmp(&v2), Ordering::Greater);
        assert_eq!(v3.cmp(&v3), Ordering::Equal);
        assert_eq!(v3.cmp(&v4), Ordering::Less);
        assert_eq!(v3.cmp(&v5), Ordering::Less);

        assert_eq!(v4.cmp(&v0), Ordering::Greater);
        assert_eq!(v4.cmp(&v1), Ordering::Greater);
        assert_eq!(v4.cmp(&v2), Ordering::Greater);
        assert_eq!(v4.cmp(&v3), Ordering::Greater);
        assert_eq!(v4.cmp(&v4), Ordering::Equal);
        assert_eq!(v4.cmp(&v5), Ordering::Less);

        assert_eq!(v5.cmp(&v0), Ordering::Greater);
        assert_eq!(v5.cmp(&v1), Ordering::Greater);
        assert_eq!(v5.cmp(&v2), Ordering::Greater);
        assert_eq!(v5.cmp(&v3), Ordering::Greater);
        assert_eq!(v5.cmp(&v4), Ordering::Greater);
        assert_eq!(v5.cmp(&v5), Ordering::Equal);
    }

    #[test]
    fn test_add() {
        let lhs =
            parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        assert_eq!(
            lhs + rhs,
            parse_scalar("0x6447adc64b17816528ee763e0b64ce7ee546304e19dd71176e7b468d8c9a6e7b")
        );
        assert_eq!(
            lhs + &rhs,
            parse_scalar("0x6447adc64b17816528ee763e0b64ce7ee546304e19dd71176e7b468d8c9a6e7b")
        );
    }

    #[test]
    fn test_add_wraparound() {
        let lhs =
            parse_scalar("0x5445e022a3c13a026ec2378170357420280e21d24f537bca42830d1bb5823236");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        assert_eq!(
            lhs + rhs,
            parse_scalar("0x03674752fdab8efaa80c59f2a14e26dc01c3f8a2660c81cd6862b72bc606760b")
        );
        assert_eq!(
            lhs + &rhs,
            parse_scalar("0x03674752fdab8efaa80c59f2a14e26dc01c3f8a2660c81cd6862b72bc606760b")
        );
    }

    #[test]
    fn test_add_assign() {
        let mut lhs =
            parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        lhs += rhs;
        assert_eq!(
            lhs,
            parse_scalar("0x6447adc64b17816528ee763e0b64ce7ee546304e19dd71176e7b468d8c9a6e7b")
        );
    }

    #[test]
    fn test_add_assign_ref() {
        let mut lhs =
            parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        lhs += &rhs;
        assert_eq!(
            lhs,
            parse_scalar("0x6447adc64b17816528ee763e0b64ce7ee546304e19dd71176e7b468d8c9a6e7b")
        );
    }

    #[test]
    fn test_add_assign_wraparound() {
        let mut lhs =
            parse_scalar("0x5445e022a3c13a026ec2378170357420280e21d24f537bca42830d1bb5823236");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        lhs += rhs;
        assert_eq!(
            lhs,
            parse_scalar("0x03674752fdab8efaa80c59f2a14e26dc01c3f8a2660c81cd6862b72bc606760b")
        );
    }

    #[test]
    fn test_add_assign_wraparound_ref() {
        let mut lhs =
            parse_scalar("0x5445e022a3c13a026ec2378170357420280e21d24f537bca42830d1bb5823236");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        lhs += &rhs;
        assert_eq!(
            lhs,
            parse_scalar("0x03674752fdab8efaa80c59f2a14e26dc01c3f8a2660c81cd6862b72bc606760b")
        );
    }

    #[test]
    fn test_sub() {
        let lhs =
            parse_scalar("0x6447adc64b17816528ee763e0b64ce7ee546304e19dd71176e7b468d8c9a6e7b");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        assert_eq!(
            lhs - rhs,
            parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5")
        );
        assert_eq!(
            lhs - &rhs,
            parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5")
        );
    }

    #[test]
    fn test_sub_wraparound() {
        let lhs =
            parse_scalar("0x03674752fdab8efaa80c59f2a14e26dc01c3f8a2660c81cd6862b72bc606760b");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        assert_eq!(
            lhs - rhs,
            parse_scalar("0x5445e022a3c13a026ec2378170357420280e21d24f537bca42830d1bb5823236")
        );
        assert_eq!(
            lhs - &rhs,
            parse_scalar("0x5445e022a3c13a026ec2378170357420280e21d24f537bca42830d1bb5823236")
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut lhs =
            parse_scalar("0x6447adc64b17816528ee763e0b64ce7ee546304e19dd71176e7b468d8c9a6e7b");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        lhs -= rhs;
        assert_eq!(
            lhs,
            parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5")
        );
    }

    #[test]
    fn test_sub_assign_ref() {
        let mut lhs =
            parse_scalar("0x6447adc64b17816528ee763e0b64ce7ee546304e19dd71176e7b468d8c9a6e7b");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        lhs -= &rhs;
        assert_eq!(
            lhs,
            parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5")
        );
    }

    #[test]
    fn test_sub_assign_wraparound() {
        let mut lhs =
            parse_scalar("0x03674752fdab8efaa80c59f2a14e26dc01c3f8a2660c81cd6862b72bc606760b");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        lhs -= rhs;
        assert_eq!(
            lhs,
            parse_scalar("0x5445e022a3c13a026ec2378170357420280e21d24f537bca42830d1bb5823236")
        );
    }

    #[test]
    fn test_sub_assign_wraparound_ref() {
        let mut lhs =
            parse_scalar("0x03674752fdab8efaa80c59f2a14e26dc01c3f8a2660c81cd6862b72bc606760b");
        let rhs =
            parse_scalar("0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6");
        lhs -= &rhs;
        assert_eq!(
            lhs,
            parse_scalar("0x5445e022a3c13a026ec2378170357420280e21d24f537bca42830d1bb5823236")
        );
    }

    fn test_neg_impl(value: Scalar) {
        assert_eq!(-value, Scalar::MAX - value + Scalar::ONE);
    }

    #[test]
    fn test_neg() {
        assert_eq!(-Scalar::ZERO, Scalar::ZERO);
        assert_eq!(-Scalar::ONE, Scalar::MAX);
        assert_eq!(-Scalar::from_const(2), Scalar::MAX_MINUS_ONE);
        test_neg_impl(parse_scalar(
            "0x03674752fdab8efaa80c59f2a14e26dc01c3f8a2660c81cd6862b72bc606760b",
        ));
        test_neg_impl(parse_scalar(
            "0x2f21673059ea54f8394a22713118b2b9e029b4c2b5545b4ae5dfaa10108443d6",
        ));
        test_neg_impl(parse_scalar(
            "0x5445e022a3c13a026ec2378170357420280e21d24f537bca42830d1bb5823236",
        ));
    }

    #[test]
    fn test_mul_by_zero() {
        assert_eq!(Scalar::ZERO * Scalar::from_const(42), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * &Scalar::from_const(42), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * Scalar::from_const(43), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * &Scalar::from_const(43), Scalar::ZERO);
        assert_eq!(Scalar::from_const(42) * Scalar::ZERO, Scalar::ZERO);
        assert_eq!(Scalar::from_const(42) * &Scalar::ZERO, Scalar::ZERO);
        assert_eq!(Scalar::from_const(43) * Scalar::ZERO, Scalar::ZERO);
        assert_eq!(Scalar::from_const(43) * &Scalar::ZERO, Scalar::ZERO);
    }

    #[test]
    fn test_mul_by_one() {
        assert_eq!(Scalar::ONE * Scalar::from_const(42), Scalar::from_const(42));
        assert_eq!(
            Scalar::ONE * &Scalar::from_const(42),
            Scalar::from_const(42)
        );
        assert_eq!(Scalar::ONE * Scalar::from_const(43), Scalar::from_const(43));
        assert_eq!(
            Scalar::ONE * &Scalar::from_const(43),
            Scalar::from_const(43)
        );
        assert_eq!(Scalar::from_const(42) * Scalar::ONE, Scalar::from_const(42));
        assert_eq!(
            Scalar::from_const(42) * &Scalar::ONE,
            Scalar::from_const(42)
        );
        assert_eq!(Scalar::from_const(43) * Scalar::ONE, Scalar::from_const(43));
        assert_eq!(
            Scalar::from_const(43) * &Scalar::ONE,
            Scalar::from_const(43)
        );
    }

    #[test]
    fn test_mul() {
        assert_eq!(
            Scalar::from_const(12) * Scalar::from_const(34),
            Scalar::from_const(408)
        );
        assert_eq!(
            Scalar::from_const(12) * &Scalar::from_const(34),
            Scalar::from_const(408)
        );
        assert_eq!(
            Scalar::from_const(12) * Scalar::from_const(56),
            Scalar::from_const(672)
        );
        assert_eq!(
            Scalar::from_const(12) * &Scalar::from_const(56),
            Scalar::from_const(672)
        );
        assert_eq!(
            Scalar::from_const(34) * Scalar::from_const(12),
            Scalar::from_const(408)
        );
        assert_eq!(
            Scalar::from_const(34) * &Scalar::from_const(12),
            Scalar::from_const(408)
        );
        assert_eq!(
            Scalar::from_const(56) * Scalar::from_const(12),
            Scalar::from_const(672)
        );
        assert_eq!(
            Scalar::from_const(56) * &Scalar::from_const(12),
            Scalar::from_const(672)
        );
    }

    fn test_mul_large_impl(v1: Scalar, v2: Scalar, v3: Scalar) {
        assert_eq!(v1 * v2, v3);
        assert_eq!(v1 * &v2, v3);
        assert_eq!(v2 * v1, v3);
        assert_eq!(v2 * &v1, v3);
    }

    #[test]
    fn test_mul_large() {
        test_mul_large_impl(
            parse_scalar("0x1be5c79927a7c7c2c1057e99b51e26efc2bac5029c6322e20405fc9334c50a9f"),
            parse_scalar("0x395ff9efcaa35d618872a95b7244c4b3b2a7e1d9276d4e88db27217993014628"),
            parse_scalar("0x0f21dcbfb54f62dd8ed757a0a2938c831c5bea612fffa42237f8a0e269b67e02"),
        );
        test_mul_large_impl(
            parse_scalar("0x233f7c593e331b2e1285f17013cd4b692d7219c10bf06adca229780913851577"),
            parse_scalar("0x74b95de3995095f242fa8dc762645eb31dffd6fcda71851456db33ef75365820"),
            parse_scalar("0x53ca68eb0e0389d9950dc014a06f178af3f950222c731e4572d5a04e4708e931"),
        );
    }

    #[test]
    fn test_sum_owned() {
        let values = vec![Scalar::ONE, Scalar::from_const(2), Scalar::from_const(3)];
        assert_eq!(values.into_iter().sum::<Scalar>(), Scalar::from_const(6));
    }

    #[test]
    fn test_sum_refs() {
        let values = vec![Scalar::ONE, Scalar::from_const(2), Scalar::from_const(3)];
        assert_eq!(values.iter().sum::<Scalar>(), Scalar::from_const(6));
    }

    #[test]
    fn test_sum_empty() {
        let values: Vec<Scalar> = vec![];
        assert_eq!(values.into_iter().sum::<Scalar>(), Scalar::ZERO);
    }

    #[test]
    fn test_sum_empty_refs() {
        let values: Vec<Scalar> = vec![];
        assert_eq!(values.iter().sum::<Scalar>(), Scalar::ZERO);
    }

    #[test]
    fn test_sum_single() {
        let values = vec![Scalar::from_const(42)];
        assert_eq!(values.into_iter().sum::<Scalar>(), Scalar::from_const(42));
    }

    #[test]
    fn test_sum_wraps_modulo_p() {
        let values = vec![Scalar::MAX, Scalar::ONE];
        assert_eq!(values.into_iter().sum::<Scalar>(), Scalar::ZERO);
    }

    #[test]
    fn test_product_owned() {
        let values = vec![
            Scalar::from_const(2),
            Scalar::from_const(3),
            Scalar::from_const(4),
        ];
        assert_eq!(
            values.into_iter().product::<Scalar>(),
            Scalar::from_const(24)
        );
    }

    #[test]
    fn test_product_refs() {
        let values = vec![
            Scalar::from_const(2),
            Scalar::from_const(3),
            Scalar::from_const(4),
        ];
        assert_eq!(values.iter().product::<Scalar>(), Scalar::from_const(24));
    }

    #[test]
    fn test_product_empty() {
        let values: Vec<Scalar> = vec![];
        assert_eq!(values.into_iter().product::<Scalar>(), Scalar::ONE);
    }

    #[test]
    fn test_product_empty_refs() {
        let values: Vec<Scalar> = vec![];
        assert_eq!(values.iter().product::<Scalar>(), Scalar::ONE);
    }

    #[test]
    fn test_product_single() {
        let values = vec![Scalar::from_const(42)];
        assert_eq!(
            values.into_iter().product::<Scalar>(),
            Scalar::from_const(42)
        );
    }

    #[test]
    fn test_product_with_zero() {
        let values = vec![Scalar::from_const(5), Scalar::ZERO, Scalar::from_const(7)];
        assert_eq!(values.into_iter().product::<Scalar>(), Scalar::ZERO);
    }

    #[test]
    fn test_product_with_one() {
        let values = vec![Scalar::ONE, Scalar::from_const(5), Scalar::ONE];
        assert_eq!(
            values.into_iter().product::<Scalar>(),
            Scalar::from_const(5)
        );
    }

    #[test]
    fn test_ct_eq() {
        let a = Scalar::from_const(42);
        let b = Scalar::from_const(42);
        let c = Scalar::from_const(43);

        assert_eq!(bool::from(a.ct_eq(&b)), true);
        assert_eq!(bool::from(a.ct_eq(&a)), true);
        assert_eq!(bool::from(a.ct_eq(&c)), false);
        assert_eq!(bool::from(c.ct_eq(&a)), false);

        assert_eq!(bool::from(Scalar::ZERO.ct_eq(&Scalar::ZERO)), true);
        assert_eq!(bool::from(Scalar::ONE.ct_eq(&Scalar::ONE)), true);
        assert_eq!(bool::from(Scalar::MAX.ct_eq(&Scalar::MAX)), true);
        assert_eq!(bool::from(Scalar::ZERO.ct_eq(&Scalar::ONE)), false);
        assert_eq!(bool::from(Scalar::ONE.ct_eq(&Scalar::MAX)), false);

        let v1 = parse_scalar("0x318c1df8459d125dc54e1fe487bf23e8430221b69660d8ca9427235713f24de1");
        let v2 = parse_scalar("0x318c1df8459d125dc54e1fe487bf23e8430221b69660d8ca9427235713f24de2");
        assert_eq!(bool::from(v1.ct_eq(&v2)), false);
        assert_eq!(bool::from(v1.ct_eq(&v1)), true);
    }

    #[test]
    fn test_ct_gt() {
        let v0 = Scalar::from_const(0);
        let v1 = Scalar::from_const(1);
        let v2 = Scalar::from_const(42);
        let v3 = Scalar::MAX_MINUS_ONE;
        let v4 = Scalar::MAX;
        assert_eq!(bool::from(v0.ct_gt(&v0)), false);
        assert_eq!(bool::from(v1.ct_gt(&v1)), false);
        assert_eq!(bool::from(v4.ct_gt(&v4)), false);
        assert_eq!(bool::from(v1.ct_gt(&v0)), true);
        assert_eq!(bool::from(v2.ct_gt(&v0)), true);
        assert_eq!(bool::from(v2.ct_gt(&v1)), true);
        assert_eq!(bool::from(v4.ct_gt(&v3)), true);
        assert_eq!(bool::from(v4.ct_gt(&v0)), true);
        assert_eq!(bool::from(v0.ct_gt(&v1)), false);
        assert_eq!(bool::from(v0.ct_gt(&v4)), false);
        assert_eq!(bool::from(v1.ct_gt(&v2)), false);
        assert_eq!(bool::from(v3.ct_gt(&v4)), false);
    }

    #[test]
    fn test_ct_lt() {
        let v0 = Scalar::from_const(0);
        let v1 = Scalar::from_const(1);
        let v2 = Scalar::from_const(42);
        let v3 = Scalar::MAX_MINUS_ONE;
        let v4 = Scalar::MAX;
        assert_eq!(bool::from(v0.ct_lt(&v0)), false);
        assert_eq!(bool::from(v1.ct_lt(&v1)), false);
        assert_eq!(bool::from(v4.ct_lt(&v4)), false);
        assert_eq!(bool::from(v0.ct_lt(&v1)), true);
        assert_eq!(bool::from(v0.ct_lt(&v4)), true);
        assert_eq!(bool::from(v1.ct_lt(&v2)), true);
        assert_eq!(bool::from(v2.ct_lt(&v3)), true);
        assert_eq!(bool::from(v3.ct_lt(&v4)), true);
        assert_eq!(bool::from(v1.ct_lt(&v0)), false);
        assert_eq!(bool::from(v4.ct_lt(&v3)), false);
        assert_eq!(bool::from(v4.ct_lt(&v0)), false);
    }

    #[test]
    fn test_conditional_select() {
        let a = Scalar::from_const(12);
        let b = Scalar::from_const(34);
        assert_eq!(Scalar::conditional_select(&a, &b, Choice::from(0)), a);
        assert_eq!(Scalar::conditional_select(&a, &b, Choice::from(1)), b);
        assert_eq!(
            Scalar::conditional_select(&Scalar::ZERO, &Scalar::ONE, Choice::from(0)),
            Scalar::ZERO
        );
        assert_eq!(
            Scalar::conditional_select(&Scalar::ZERO, &Scalar::ONE, Choice::from(1)),
            Scalar::ONE
        );
        assert_eq!(Scalar::conditional_select(&a, &a, Choice::from(0)), a);
        assert_eq!(Scalar::conditional_select(&a, &a, Choice::from(1)), a);
    }

    #[test]
    fn test_is_odd() {
        assert_eq!(bool::from(Scalar::ZERO.is_odd()), false);
        assert_eq!(bool::from(Scalar::ONE.is_odd()), true);
        assert_eq!(bool::from(Scalar::from_const(2).is_odd()), false);
        assert_eq!(bool::from(Scalar::from_const(3).is_odd()), true);
        assert_eq!(bool::from(Scalar::from_const(100).is_odd()), false);
        assert_eq!(bool::from(Scalar::from_const(101).is_odd()), true);
        assert_eq!(bool::from(Scalar::MAX_MINUS_ONE.is_odd()), true);
        assert_eq!(bool::from(Scalar::MAX.is_odd()), false);
    }

    struct OsRng;

    impl rand_core::TryRng for OsRng {
        type Error = getrandom::Error;

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
            getrandom::fill(dest)
        }

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            let mut bytes = [0u8; 4];
            getrandom::fill(&mut bytes)?;
            Ok(u32::from_le_bytes(bytes))
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            let mut bytes = [0u8; 8];
            getrandom::fill(&mut bytes)?;
            Ok(u64::from_le_bytes(bytes))
        }
    }

    #[test]
    fn test_random_with_rng() {
        let mut rng = rand_core::UnwrapErr(OsRng);
        assert_ne!(Scalar::random(&mut rng), Scalar::random(&mut rng));
        assert_ne!(Scalar::random(&mut rng), Scalar::random(&mut rng));
        assert_ne!(Scalar::random(&mut rng), Scalar::random(&mut rng));
    }

    #[test]
    fn test_random_default() {
        assert_ne!(Scalar::random_default(), Scalar::random_default());
        assert_ne!(Scalar::random_default(), Scalar::random_default());
        assert_ne!(Scalar::random_default(), Scalar::random_default());
    }

    #[test]
    fn test_square() {
        assert_eq!(Scalar::ZERO.square(), Scalar::ZERO);
        assert_eq!(Scalar::ONE.square(), Scalar::ONE);
        assert_eq!(Scalar::from_const(7).square(), Scalar::from_const(49));
        assert_eq!(Scalar::from_const(12).square(), Scalar::from_const(144));
        let v = parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5");
        assert_eq!(v.square(), v * v);
        let v = parse_scalar("0x1be5c79927a7c7c2c1057e99b51e26efc2bac5029c6322e20405fc9334c50a9f");
        assert_eq!(v.square(), v * v);
    }

    #[test]
    fn test_double() {
        assert_eq!(Scalar::ZERO.double(), Scalar::ZERO);
        assert_eq!(Scalar::ONE.double(), Scalar::from_const(2));
        assert_eq!(Scalar::from_const(21).double(), Scalar::from_const(42));
        let v = parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5");
        assert_eq!(v.double(), v + v);
        assert_eq!(Scalar::MAX.double(), Scalar::MAX_MINUS_ONE);
        assert_eq!(Scalar::MAX_MINUS_ONE.double(), -Scalar::from_const(4));
    }

    #[test]
    fn test_invert() {
        assert!(bool::from(Scalar::ZERO.invert().is_none()));
        assert_eq!(Scalar::ONE.invert().unwrap(), Scalar::ONE);
        assert_eq!(Scalar::MAX.invert().unwrap(), Scalar::MAX);
        assert_eq!(Scalar::from_const(2).invert().unwrap(), Scalar::TWO_INV);
        let v = parse_scalar("0x35264695f12d2c6cefa453ccda4c1bc5051c7b8b648915cc889b9c7d7c162aa5");
        assert_eq!(v * v.invert().unwrap(), Scalar::ONE);
        let v = parse_scalar("0x1be5c79927a7c7c2c1057e99b51e26efc2bac5029c6322e20405fc9334c50a9f");
        assert_eq!(v * v.invert().unwrap(), Scalar::ONE);
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(Scalar::ZERO.sqrt().unwrap(), Scalar::ZERO);
        assert_eq!(Scalar::ONE.sqrt().unwrap(), Scalar::ONE);
        assert_eq!(Scalar::from_const(2).sqrt().unwrap().square(), 2.into());
        assert_eq!(Scalar::from_const(3).sqrt().unwrap().square(), 3.into());
        assert_eq!(Scalar::from_const(4).sqrt().unwrap().square(), 4.into());
        assert!(Scalar::from_const(5).sqrt().into_option().is_none());
        assert!(
            parse_scalar("0x26f9845e46b8d4a46419673fcd8d6a22df74920c2f1149eb60839d7e3dfbe8ec")
                .sqrt()
                .into_option()
                .is_none()
        );
        let v = parse_scalar("0x4dd28128e1cd4cedfb041d4e87476561b8c347877bd4a82687fac6e5d7264156");
        assert_eq!(v.sqrt().unwrap().square(), v);
    }

    #[test]
    fn test_parse_binary() {
        assert_eq!(
            Scalar::from_str_radix("0", 2).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("1", 2).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("00", 2).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("01", 2).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("10", 2).unwrap(),
            Scalar::from_const(2)
        );
        assert_eq!(
            Scalar::from_str_radix("11", 2).unwrap(),
            Scalar::from_const(3)
        );
        assert_eq!(
            Scalar::from_str_radix("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000110011100111101110111110010100111101001101101010101010001111011111111111111111111111111111111111111111111111111111111111111", 2).unwrap(),
            Scalar::MAX - Scalar::ONE
        );
        assert_eq!(
            Scalar::from_str_radix("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000110011100111101110111110010100111101001101101010101010001111100000000000000000000000000000000000000000000000000000000000000", 2).unwrap(),
            Scalar::MAX
        );
        assert!(
            Scalar::from_str_radix("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000110011100111101110111110010100111101001101101010101010001111100000000000000000000000000000000000000000000000000000000000001", 2).is_err(),
        );
    }

    #[test]
    fn test_print_binary() {
        assert_eq!(Scalar::from_const(0).to_str_radix(2, 0, false), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(2, 0, false), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(2, 0, false), "10");
        assert_eq!(Scalar::from_const(3).to_str_radix(2, 0, false), "11");
        assert_eq!(Scalar::from_const(0).to_str_radix(2, 1, false), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(2, 1, false), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(2, 1, false), "10");
        assert_eq!(Scalar::from_const(3).to_str_radix(2, 1, false), "11");
        assert_eq!(Scalar::from_const(0).to_str_radix(2, 2, false), "00");
        assert_eq!(Scalar::from_const(1).to_str_radix(2, 2, false), "01");
        assert_eq!(Scalar::from_const(2).to_str_radix(2, 2, false), "10");
        assert_eq!(Scalar::from_const(3).to_str_radix(2, 2, false), "11");
        assert_eq!(
            (Scalar::MAX - Scalar::ONE).to_str_radix(2, 0, false),
            "111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000110011100111101110111110010100111101001101101010101010001111011111111111111111111111111111111111111111111111111111111111111"
        );
        assert_eq!(
            Scalar::MAX.to_str_radix(2, 0, false),
            "111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000000110011100111101110111110010100111101001101101010101010001111100000000000000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn test_parse_octal() {
        assert_eq!(
            Scalar::from_str_radix("0", 8).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("1", 8).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("2", 8).unwrap(),
            Scalar::from_const(2)
        );
        assert_eq!(
            Scalar::from_str_radix("6", 8).unwrap(),
            Scalar::from_const(6)
        );
        assert_eq!(
            Scalar::from_str_radix("7", 8).unwrap(),
            Scalar::from_const(7)
        );
        assert!(Scalar::from_str_radix("8", 8).is_err());
        assert_eq!(
            Scalar::from_str_radix("00", 8).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("01", 8).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("02", 8).unwrap(),
            Scalar::from_const(2)
        );
        assert_eq!(
            Scalar::from_str_radix("10", 8).unwrap(),
            Scalar::from_const(8)
        );
        assert_eq!(
            Scalar::from_str_radix("11", 8).unwrap(),
            Scalar::from_const(9)
        );
        assert_eq!(
            Scalar::from_str_radix("12", 8).unwrap(),
            Scalar::from_const(10)
        );
        assert_eq!(
            Scalar::from_str_radix("20", 8).unwrap(),
            Scalar::from_const(16)
        );
        assert_eq!(
            Scalar::from_str_radix("21", 8).unwrap(),
            Scalar::from_const(17)
        );
        assert_eq!(
            Scalar::from_str_radix("22", 8).unwrap(),
            Scalar::from_const(18)
        );
        assert_eq!(
            Scalar::from_str_radix("7777777777777777777777777777777777777777770063475676247515525217377777777777777777777", 8).unwrap(),
            Scalar::MAX - Scalar::ONE
        );
        assert_eq!(
            Scalar::from_str_radix("7777777777777777777777777777777777777777770063475676247515525217400000000000000000000", 8).unwrap(),
            Scalar::MAX
        );
        assert!(
            Scalar::from_str_radix("7777777777777777777777777777777777777777770063475676247515525217400000000000000000001", 8).is_err(),
        );
    }

    #[test]
    fn test_print_octal() {
        assert_eq!(Scalar::from_const(0).to_str_radix(8, 0, false), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(8, 0, false), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(8, 0, false), "2");
        assert_eq!(Scalar::from_const(6).to_str_radix(8, 0, false), "6");
        assert_eq!(Scalar::from_const(7).to_str_radix(8, 0, false), "7");
        assert_eq!(Scalar::from_const(8).to_str_radix(8, 0, false), "10");
        assert_eq!(Scalar::from_const(9).to_str_radix(8, 0, false), "11");
        assert_eq!(Scalar::from_const(10).to_str_radix(8, 0, false), "12");
        assert_eq!(Scalar::from_const(0).to_str_radix(8, 1, false), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(8, 1, false), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(8, 1, false), "2");
        assert_eq!(Scalar::from_const(6).to_str_radix(8, 1, false), "6");
        assert_eq!(Scalar::from_const(7).to_str_radix(8, 1, false), "7");
        assert_eq!(Scalar::from_const(8).to_str_radix(8, 1, false), "10");
        assert_eq!(Scalar::from_const(9).to_str_radix(8, 1, false), "11");
        assert_eq!(Scalar::from_const(10).to_str_radix(8, 1, false), "12");
        assert_eq!(Scalar::from_const(0).to_str_radix(8, 2, false), "00");
        assert_eq!(Scalar::from_const(1).to_str_radix(8, 2, false), "01");
        assert_eq!(Scalar::from_const(2).to_str_radix(8, 2, false), "02");
        assert_eq!(Scalar::from_const(6).to_str_radix(8, 2, false), "06");
        assert_eq!(Scalar::from_const(7).to_str_radix(8, 2, false), "07");
        assert_eq!(Scalar::from_const(8).to_str_radix(8, 2, false), "10");
        assert_eq!(Scalar::from_const(9).to_str_radix(8, 2, false), "11");
        assert_eq!(Scalar::from_const(10).to_str_radix(8, 2, false), "12");
        assert_eq!(
            (Scalar::MAX - Scalar::ONE).to_str_radix(8, 0, false),
            "7777777777777777777777777777777777777777770063475676247515525217377777777777777777777"
        );
        assert_eq!(
            Scalar::MAX.to_str_radix(8, 0, false),
            "7777777777777777777777777777777777777777770063475676247515525217400000000000000000000"
        );
    }

    #[test]
    fn test_parse_hexadecimal_lower_case() {
        assert_eq!(
            Scalar::from_str_radix("0", 16).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("1", 16).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("2", 16).unwrap(),
            Scalar::from_const(2)
        );
        assert_eq!(
            Scalar::from_str_radix("9", 16).unwrap(),
            Scalar::from_const(9)
        );
        assert_eq!(
            Scalar::from_str_radix("a", 16).unwrap(),
            Scalar::from_const(10)
        );
        assert_eq!(
            Scalar::from_str_radix("e", 16).unwrap(),
            Scalar::from_const(14)
        );
        assert_eq!(
            Scalar::from_str_radix("f", 16).unwrap(),
            Scalar::from_const(15)
        );
        assert!(Scalar::from_str_radix("8", 8).is_err());
        assert_eq!(
            Scalar::from_str_radix("00", 16).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("01", 16).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("02", 16).unwrap(),
            Scalar::from_const(2)
        );
        assert_eq!(
            Scalar::from_str_radix("09", 16).unwrap(),
            Scalar::from_const(9)
        );
        assert_eq!(
            Scalar::from_str_radix("0a", 16).unwrap(),
            Scalar::from_const(10)
        );
        assert_eq!(
            Scalar::from_str_radix("0e", 16).unwrap(),
            Scalar::from_const(14)
        );
        assert_eq!(
            Scalar::from_str_radix("0f", 16).unwrap(),
            Scalar::from_const(15)
        );
        assert_eq!(
            Scalar::from_str_radix("10", 16).unwrap(),
            Scalar::from_const(16)
        );
        assert_eq!(
            Scalar::from_str_radix("11", 16).unwrap(),
            Scalar::from_const(17)
        );
        assert_eq!(
            Scalar::from_str_radix("12", 16).unwrap(),
            Scalar::from_const(18)
        );
        assert_eq!(
            Scalar::from_str_radix("19", 16).unwrap(),
            Scalar::from_const(25)
        );
        assert_eq!(
            Scalar::from_str_radix("1a", 16).unwrap(),
            Scalar::from_const(26)
        );
        assert_eq!(
            Scalar::from_str_radix("1e", 16).unwrap(),
            Scalar::from_const(30)
        );
        assert_eq!(
            Scalar::from_str_radix("1f", 16).unwrap(),
            Scalar::from_const(31)
        );
        assert_eq!(
            Scalar::from_str_radix("20", 16).unwrap(),
            Scalar::from_const(32)
        );
        assert_eq!(
            Scalar::from_str_radix("21", 16).unwrap(),
            Scalar::from_const(33)
        );
        assert_eq!(
            Scalar::from_str_radix("22", 16).unwrap(),
            Scalar::from_const(34)
        );
        assert_eq!(
            Scalar::from_str_radix(
                "7ffffffffffffffffffffffffffffffe0673ddf29e9b5547bfffffffffffffff",
                16
            )
            .unwrap(),
            Scalar::MAX - Scalar::ONE
        );
        assert_eq!(
            Scalar::from_str_radix(
                "7ffffffffffffffffffffffffffffffe0673ddf29e9b5547C000000000000000",
                16
            )
            .unwrap(),
            Scalar::MAX
        );
        assert!(
            Scalar::from_str_radix(
                "7ffffffffffffffffffffffffffffffe0673ddf29e9b5547C000000000000001",
                8
            )
            .is_err(),
        );
    }

    #[test]
    fn test_parse_hexadecimal_upper_case() {
        assert_eq!(
            Scalar::from_str_radix("0", 16).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("1", 16).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("2", 16).unwrap(),
            Scalar::from_const(2)
        );
        assert_eq!(
            Scalar::from_str_radix("9", 16).unwrap(),
            Scalar::from_const(9)
        );
        assert_eq!(
            Scalar::from_str_radix("a", 16).unwrap(),
            Scalar::from_const(10)
        );
        assert_eq!(
            Scalar::from_str_radix("e", 16).unwrap(),
            Scalar::from_const(14)
        );
        assert_eq!(
            Scalar::from_str_radix("f", 16).unwrap(),
            Scalar::from_const(15)
        );
        assert!(Scalar::from_str_radix("8", 8).is_err());
        assert_eq!(
            Scalar::from_str_radix("00", 16).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("01", 16).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("02", 16).unwrap(),
            Scalar::from_const(2)
        );
        assert_eq!(
            Scalar::from_str_radix("09", 16).unwrap(),
            Scalar::from_const(9)
        );
        assert_eq!(
            Scalar::from_str_radix("0A", 16).unwrap(),
            Scalar::from_const(10)
        );
        assert_eq!(
            Scalar::from_str_radix("0E", 16).unwrap(),
            Scalar::from_const(14)
        );
        assert_eq!(
            Scalar::from_str_radix("0F", 16).unwrap(),
            Scalar::from_const(15)
        );
        assert_eq!(
            Scalar::from_str_radix("10", 16).unwrap(),
            Scalar::from_const(16)
        );
        assert_eq!(
            Scalar::from_str_radix("11", 16).unwrap(),
            Scalar::from_const(17)
        );
        assert_eq!(
            Scalar::from_str_radix("12", 16).unwrap(),
            Scalar::from_const(18)
        );
        assert_eq!(
            Scalar::from_str_radix("19", 16).unwrap(),
            Scalar::from_const(25)
        );
        assert_eq!(
            Scalar::from_str_radix("1A", 16).unwrap(),
            Scalar::from_const(26)
        );
        assert_eq!(
            Scalar::from_str_radix("1E", 16).unwrap(),
            Scalar::from_const(30)
        );
        assert_eq!(
            Scalar::from_str_radix("1F", 16).unwrap(),
            Scalar::from_const(31)
        );
        assert_eq!(
            Scalar::from_str_radix("20", 16).unwrap(),
            Scalar::from_const(32)
        );
        assert_eq!(
            Scalar::from_str_radix("21", 16).unwrap(),
            Scalar::from_const(33)
        );
        assert_eq!(
            Scalar::from_str_radix("22", 16).unwrap(),
            Scalar::from_const(34)
        );
        assert_eq!(
            Scalar::from_str_radix(
                "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE0673DDF29E9B5547BFFFFFFFFFFFFFFF",
                16
            )
            .unwrap(),
            Scalar::MAX - Scalar::ONE
        );
        assert_eq!(
            Scalar::from_str_radix(
                "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE0673DDF29E9B5547C000000000000000",
                16
            )
            .unwrap(),
            Scalar::MAX
        );
        assert!(
            Scalar::from_str_radix(
                "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE0673DDF29E9B5547C000000000000001",
                8
            )
            .is_err(),
        );
    }

    #[test]
    fn test_print_hexadecimal_lower_case() {
        assert_eq!(Scalar::from_const(0).to_str_radix(16, 0, false), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(16, 0, false), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(16, 0, false), "2");
        assert_eq!(Scalar::from_const(9).to_str_radix(16, 0, false), "9");
        assert_eq!(Scalar::from_const(10).to_str_radix(16, 0, false), "a");
        assert_eq!(Scalar::from_const(14).to_str_radix(16, 0, false), "e");
        assert_eq!(Scalar::from_const(15).to_str_radix(16, 0, false), "f");
        assert_eq!(Scalar::from_const(16).to_str_radix(16, 0, false), "10");
        assert_eq!(Scalar::from_const(17).to_str_radix(16, 0, false), "11");
        assert_eq!(Scalar::from_const(18).to_str_radix(16, 0, false), "12");
        assert_eq!(Scalar::from_const(25).to_str_radix(16, 0, false), "19");
        assert_eq!(Scalar::from_const(26).to_str_radix(16, 0, false), "1a");
        assert_eq!(Scalar::from_const(30).to_str_radix(16, 0, false), "1e");
        assert_eq!(Scalar::from_const(31).to_str_radix(16, 0, false), "1f");
        assert_eq!(Scalar::from_const(0).to_str_radix(16, 1, false), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(16, 1, false), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(16, 1, false), "2");
        assert_eq!(Scalar::from_const(9).to_str_radix(16, 1, false), "9");
        assert_eq!(Scalar::from_const(10).to_str_radix(16, 1, false), "a");
        assert_eq!(Scalar::from_const(14).to_str_radix(16, 1, false), "e");
        assert_eq!(Scalar::from_const(15).to_str_radix(16, 1, false), "f");
        assert_eq!(Scalar::from_const(16).to_str_radix(16, 1, false), "10");
        assert_eq!(Scalar::from_const(17).to_str_radix(16, 1, false), "11");
        assert_eq!(Scalar::from_const(18).to_str_radix(16, 1, false), "12");
        assert_eq!(Scalar::from_const(25).to_str_radix(16, 1, false), "19");
        assert_eq!(Scalar::from_const(26).to_str_radix(16, 1, false), "1a");
        assert_eq!(Scalar::from_const(30).to_str_radix(16, 1, false), "1e");
        assert_eq!(Scalar::from_const(31).to_str_radix(16, 1, false), "1f");
        assert_eq!(Scalar::from_const(0).to_str_radix(16, 2, false), "00");
        assert_eq!(Scalar::from_const(1).to_str_radix(16, 2, false), "01");
        assert_eq!(Scalar::from_const(2).to_str_radix(16, 2, false), "02");
        assert_eq!(Scalar::from_const(9).to_str_radix(16, 2, false), "09");
        assert_eq!(Scalar::from_const(10).to_str_radix(16, 2, false), "0a");
        assert_eq!(Scalar::from_const(14).to_str_radix(16, 2, false), "0e");
        assert_eq!(Scalar::from_const(15).to_str_radix(16, 2, false), "0f");
        assert_eq!(Scalar::from_const(16).to_str_radix(16, 2, false), "10");
        assert_eq!(Scalar::from_const(17).to_str_radix(16, 2, false), "11");
        assert_eq!(Scalar::from_const(18).to_str_radix(16, 2, false), "12");
        assert_eq!(Scalar::from_const(25).to_str_radix(16, 2, false), "19");
        assert_eq!(Scalar::from_const(26).to_str_radix(16, 2, false), "1a");
        assert_eq!(Scalar::from_const(30).to_str_radix(16, 2, false), "1e");
        assert_eq!(Scalar::from_const(31).to_str_radix(16, 2, false), "1f");
        assert_eq!(
            (Scalar::MAX - Scalar::ONE).to_str_radix(16, 0, false),
            "7ffffffffffffffffffffffffffffffe0673ddf29e9b5547bfffffffffffffff"
        );
        assert_eq!(
            Scalar::MAX.to_str_radix(16, 0, false),
            "7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
        );
    }

    #[test]
    fn test_print_hexadecimal_upper_case() {
        assert_eq!(Scalar::from_const(0).to_str_radix(16, 0, true), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(16, 0, true), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(16, 0, true), "2");
        assert_eq!(Scalar::from_const(9).to_str_radix(16, 0, true), "9");
        assert_eq!(Scalar::from_const(10).to_str_radix(16, 0, true), "A");
        assert_eq!(Scalar::from_const(14).to_str_radix(16, 0, true), "E");
        assert_eq!(Scalar::from_const(15).to_str_radix(16, 0, true), "F");
        assert_eq!(Scalar::from_const(16).to_str_radix(16, 0, true), "10");
        assert_eq!(Scalar::from_const(17).to_str_radix(16, 0, true), "11");
        assert_eq!(Scalar::from_const(18).to_str_radix(16, 0, true), "12");
        assert_eq!(Scalar::from_const(25).to_str_radix(16, 0, true), "19");
        assert_eq!(Scalar::from_const(26).to_str_radix(16, 0, true), "1A");
        assert_eq!(Scalar::from_const(30).to_str_radix(16, 0, true), "1E");
        assert_eq!(Scalar::from_const(31).to_str_radix(16, 0, true), "1F");
        assert_eq!(Scalar::from_const(0).to_str_radix(16, 1, true), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(16, 1, true), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(16, 1, true), "2");
        assert_eq!(Scalar::from_const(9).to_str_radix(16, 1, true), "9");
        assert_eq!(Scalar::from_const(10).to_str_radix(16, 1, true), "A");
        assert_eq!(Scalar::from_const(14).to_str_radix(16, 1, true), "E");
        assert_eq!(Scalar::from_const(15).to_str_radix(16, 1, true), "F");
        assert_eq!(Scalar::from_const(16).to_str_radix(16, 1, true), "10");
        assert_eq!(Scalar::from_const(17).to_str_radix(16, 1, true), "11");
        assert_eq!(Scalar::from_const(18).to_str_radix(16, 1, true), "12");
        assert_eq!(Scalar::from_const(25).to_str_radix(16, 1, true), "19");
        assert_eq!(Scalar::from_const(26).to_str_radix(16, 1, true), "1A");
        assert_eq!(Scalar::from_const(30).to_str_radix(16, 1, true), "1E");
        assert_eq!(Scalar::from_const(31).to_str_radix(16, 1, true), "1F");
        assert_eq!(Scalar::from_const(0).to_str_radix(16, 2, true), "00");
        assert_eq!(Scalar::from_const(1).to_str_radix(16, 2, true), "01");
        assert_eq!(Scalar::from_const(2).to_str_radix(16, 2, true), "02");
        assert_eq!(Scalar::from_const(9).to_str_radix(16, 2, true), "09");
        assert_eq!(Scalar::from_const(10).to_str_radix(16, 2, true), "0A");
        assert_eq!(Scalar::from_const(14).to_str_radix(16, 2, true), "0E");
        assert_eq!(Scalar::from_const(15).to_str_radix(16, 2, true), "0F");
        assert_eq!(Scalar::from_const(16).to_str_radix(16, 2, true), "10");
        assert_eq!(Scalar::from_const(17).to_str_radix(16, 2, true), "11");
        assert_eq!(Scalar::from_const(18).to_str_radix(16, 2, true), "12");
        assert_eq!(Scalar::from_const(25).to_str_radix(16, 2, true), "19");
        assert_eq!(Scalar::from_const(26).to_str_radix(16, 2, true), "1A");
        assert_eq!(Scalar::from_const(30).to_str_radix(16, 2, true), "1E");
        assert_eq!(Scalar::from_const(31).to_str_radix(16, 2, true), "1F");
        assert_eq!(
            (Scalar::MAX - Scalar::ONE).to_str_radix(16, 0, true),
            "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE0673DDF29E9B5547BFFFFFFFFFFFFFFF"
        );
        assert_eq!(
            Scalar::MAX.to_str_radix(16, 0, true),
            "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE0673DDF29E9B5547C000000000000000"
        );
    }

    #[test]
    fn test_parse_decimal() {
        assert_eq!(
            Scalar::from_str_radix("0", 10).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("1", 10).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("2", 10).unwrap(),
            Scalar::from_const(2)
        );
        assert_eq!(
            Scalar::from_str_radix("00", 10).unwrap(),
            Scalar::from_const(0)
        );
        assert_eq!(
            Scalar::from_str_radix("01", 10).unwrap(),
            Scalar::from_const(1)
        );
        assert_eq!(
            Scalar::from_str_radix("02", 10).unwrap(),
            Scalar::from_const(2)
        );
        assert_eq!(
            Scalar::from_str_radix("10", 10).unwrap(),
            Scalar::from_const(10)
        );
        assert_eq!(
            Scalar::from_str_radix("11", 10).unwrap(),
            Scalar::from_const(11)
        );
        assert_eq!(
            Scalar::from_str_radix("12", 10).unwrap(),
            Scalar::from_const(12)
        );
        assert_eq!(
            Scalar::from_str_radix("20", 10).unwrap(),
            Scalar::from_const(20)
        );
        assert_eq!(
            Scalar::from_str_radix("21", 10).unwrap(),
            Scalar::from_const(21)
        );
        assert_eq!(
            Scalar::from_str_radix("22", 10).unwrap(),
            Scalar::from_const(22)
        );
        assert_eq!(
            Scalar::from_str_radix(
                "57896044618658097711785492504343953925963004582726670246466397748101839323135",
                10
            )
            .unwrap(),
            Scalar::MAX - Scalar::ONE
        );
        assert_eq!(
            Scalar::from_str_radix(
                "57896044618658097711785492504343953925963004582726670246466397748101839323136",
                10
            )
            .unwrap(),
            Scalar::MAX
        );
        assert!(
            Scalar::from_str_radix(
                "57896044618658097711785492504343953925963004582726670246466397748101839323137",
                10
            )
            .is_err(),
        );
    }

    #[test]
    fn test_print_decimal() {
        assert_eq!(Scalar::from_const(0).to_str_radix(10, 0, false), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(10, 0, false), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(10, 0, false), "2");
        assert_eq!(Scalar::from_const(9).to_str_radix(10, 0, false), "9");
        assert_eq!(Scalar::from_const(10).to_str_radix(10, 0, false), "10");
        assert_eq!(Scalar::from_const(11).to_str_radix(10, 0, false), "11");
        assert_eq!(Scalar::from_const(0).to_str_radix(10, 1, false), "0");
        assert_eq!(Scalar::from_const(1).to_str_radix(10, 1, false), "1");
        assert_eq!(Scalar::from_const(2).to_str_radix(10, 1, false), "2");
        assert_eq!(Scalar::from_const(9).to_str_radix(10, 1, false), "9");
        assert_eq!(Scalar::from_const(10).to_str_radix(10, 1, false), "10");
        assert_eq!(Scalar::from_const(11).to_str_radix(10, 1, false), "11");
        assert_eq!(Scalar::from_const(0).to_str_radix(10, 2, false), "00");
        assert_eq!(Scalar::from_const(1).to_str_radix(10, 2, false), "01");
        assert_eq!(Scalar::from_const(2).to_str_radix(10, 2, false), "02");
        assert_eq!(Scalar::from_const(9).to_str_radix(10, 2, false), "09");
        assert_eq!(Scalar::from_const(10).to_str_radix(10, 2, false), "10");
        assert_eq!(Scalar::from_const(11).to_str_radix(10, 2, false), "11");
        assert_eq!(Scalar::from_const(0).to_str_radix(10, 3, false), "000");
        assert_eq!(Scalar::from_const(1).to_str_radix(10, 3, false), "001");
        assert_eq!(Scalar::from_const(2).to_str_radix(10, 3, false), "002");
        assert_eq!(Scalar::from_const(9).to_str_radix(10, 3, false), "009");
        assert_eq!(Scalar::from_const(10).to_str_radix(10, 3, false), "010");
        assert_eq!(Scalar::from_const(11).to_str_radix(10, 3, false), "011");
        assert_eq!(
            (Scalar::MAX - Scalar::ONE).to_str_radix(10, 0, false),
            "57896044618658097711785492504343953925963004582726670246466397748101839323135"
        );
        assert_eq!(
            Scalar::MAX.to_str_radix(10, 0, false),
            "57896044618658097711785492504343953925963004582726670246466397748101839323136"
        );
    }
}
