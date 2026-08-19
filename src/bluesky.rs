use anyhow::{self, Context};
use getrandom;
use primitive_types::{H512, U256, U512};
use starkom_ff::{
    Field, Field256, PrimeField, PrimeField256,
    helpers::{adc, add, mac, mul, sbb, sub},
};
use std::cmp::Ordering;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str::FromStr;
use std::sync::LazyLock;
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
    CtOption,
};

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

/// Upper-case characters used in textual representations.
static CHARACTERS_UPPER_CASE: &'static [u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Lower-case characters used in textual representations.
static CHARACTERS_LOWER_CASE: &'static [u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

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
    /// The raw (non-Montgomery) little-endian representation of `MAX`.
    const MAX_RAW: Self = Self(
        0xc000000000000000u64,
        0x0673ddf29e9b5547u64,
        0xfffffffffffffffeu64,
        0x7fffffffffffffffu64,
    );

    /// R in raw form, ie. the four limbs of `2^256 mod p` in little-endian order.
    const R: Self = Self(
        0x7ffffffffffffffeu64,
        0xf318441ac2c95570u64,
        0x0000000000000003u64,
        0x0000000000000000u64,
    );

    /// R in Montgomery form, ie. R^2 mod p.
    const R2: Self = Self(
        0xbfffffffffffffe5u64,
        0xab970f33c00b568du64,
        0xb296e2afc92ce69du64,
        0x1968ac3835a4f8ddu64,
    );

    const P: [u64; 4] = MODULUS;

    const P_INV: u64 = 0xbfffffffffffffffu64;

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

    const fn modp(self) -> Self {
        let subp = self.subp();
        match self.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => subp,
            _ => self,
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

        Self(t0, t1, t2, t3).modp()
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

        Self(t0, t1, t2, t3).modp()
    }

    /// Constructs scalars at compile time.
    pub const fn from_const(value: u64) -> Scalar {
        let raw = Self(value, 0, 0, 0);
        Self::mont_mul(&raw, &Self::R2)
    }

    fn to_string_impl(&self, radix: usize, pad_to: usize, upper_case: bool) -> String {
        let characters = if upper_case {
            CHARACTERS_UPPER_CASE
        } else {
            CHARACTERS_LOWER_CASE
        };
        let mut value = self.to_u256();
        let mut s = String::default();
        let radix = U256::from(radix);
        while !value.is_zero() {
            let digit = (value % radix).as_usize();
            s.push(characters[digit] as char);
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

    fn to_string_impl_log2(&self, radix_log2: u32, pad_to: usize, upper_case: bool) -> String {
        assert!(radix_log2 < 6);
        let characters = if upper_case {
            CHARACTERS_UPPER_CASE
        } else {
            CHARACTERS_LOWER_CASE
        };
        let mut value = self.to_u256();
        let mut s = String::default();
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
}

/// Alias for [`Scalar::from_const`].
pub const fn from_const(value: u64) -> Scalar {
    Scalar::from_const(value)
}

/// Parses a scalar from a string using the [`FromStr`] trait and unwrapping the result.
///
/// REQUIRES: the input string must be a static one known to have a valid scalar.
pub fn parse_scalar(s: &'static str) -> Scalar {
    s.parse().unwrap()
}

impl std::fmt::Debug for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scalar({:#066x})", self)
    }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#066x}", self)
    }
}

impl std::fmt::Binary for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if f.alternate() { "0b" } else { "" };
        f.pad_integral(true, prefix, &self.to_str_radix(2, 0, false))
    }
}

impl std::fmt::Octal for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if f.alternate() { "0o" } else { "" };
        f.pad_integral(true, prefix, &self.to_str_radix(8, 0, false))
    }
}

impl std::fmt::LowerHex for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if f.alternate() { "0x" } else { "" };
        f.pad_integral(true, prefix, &self.to_str_radix(16, 0, false))
    }
}

impl std::fmt::UpperHex for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if f.alternate() { "0x" } else { "" };
        f.pad_integral(true, prefix, &self.to_str_radix(16, 0, true))
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
        let lhs = self.to_raw();
        let rhs = other.to_raw();
        let gt3 = lhs.3.ct_gt(&rhs.3);
        let gt2 = lhs.2.ct_gt(&rhs.2);
        let gt1 = lhs.1.ct_gt(&rhs.1);
        let gt0 = lhs.0.ct_gt(&rhs.0);
        let eq3 = lhs.3.ct_eq(&rhs.3);
        let eq2 = lhs.2.ct_eq(&rhs.2);
        let eq1 = lhs.1.ct_eq(&rhs.1);
        gt3 | eq3 & gt2 | eq3 & eq2 & gt1 | eq3 & eq2 & eq1 & gt0
    }
}

impl ConstantTimeLess for Scalar {}

impl ConditionallySelectable for Scalar {
    fn conditional_select(a: &Self, b: &Self, choice: subtle::Choice) -> Self {
        Scalar(
            u64::conditional_select(&a.0, &b.0, choice),
            u64::conditional_select(&a.1, &b.1, choice),
            u64::conditional_select(&a.2, &b.2, choice),
            u64::conditional_select(&a.3, &b.3, choice),
        )
    }
}

impl Add<&Scalar> for Scalar {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let (r0, c0) = add(self.0, rhs.0);
        let (r1, c1) = adc(self.1, rhs.1, c0);
        let (r2, c2) = adc(self.2, rhs.2, c1);
        let (r3, _) = adc(self.3, rhs.3, c2);
        Self(r0, r1, r2, r3).modp()
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

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero().into() {
            return self;
        }
        let (r0, b0) = sub(Self::P[0], self.0);
        let (r1, b1) = sbb(Self::P[1], self.1, b0);
        let (r2, b2) = sbb(Self::P[2], self.2, b1);
        let (r3, _) = sbb(Self::P[3], self.3, b2);
        Self(r0, r1, r2, r3)
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

impl Div<&Scalar> for Scalar {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        assert!(!bool::from(rhs.is_zero()), "division by zero");
        Self::mont_mul(&self, &rhs.invert_unwrap())
    }
}

impl Div for Scalar {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(!bool::from(rhs.is_zero()), "division by zero");
        Self::mont_mul(&self, &rhs.invert_unwrap())
    }
}

impl DivAssign<&Scalar> for Scalar {
    fn div_assign(&mut self, rhs: &Self) {
        assert!(!bool::from(rhs.is_zero()), "division by zero");
        *self = Self::mont_mul(self, &rhs.invert_unwrap());
    }
}

impl DivAssign for Scalar {
    fn div_assign(&mut self, rhs: Self) {
        assert!(!bool::from(rhs.is_zero()), "division by zero");
        *self = Self::mont_mul(self, &rhs.invert_unwrap());
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

impl FromStr for Scalar {
    type Err = std::fmt::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with("0x") || s.starts_with("0X") {
            Self::from_str_radix(&s[2..], 16)
        } else if s.starts_with("0b") || s.starts_with("0B") {
            Self::from_str_radix(&s[2..], 2)
        } else if s.starts_with("0o") || s.starts_with("0O") {
            Self::from_str_radix(&s[2..], 8)
        } else if s.starts_with("0") {
            Self::from_str_radix(s, 8)
        } else {
            Self::from_str_radix(s, 10)
        }
    }
}

impl From<u8> for Scalar {
    fn from(value: u8) -> Self {
        Self::mont_mul(&Self(value as u64, 0, 0, 0), &Self::R2)
    }
}

impl From<u16> for Scalar {
    fn from(value: u16) -> Self {
        Self::mont_mul(&Self(value as u64, 0, 0, 0), &Self::R2)
    }
}

impl From<u32> for Scalar {
    fn from(value: u32) -> Self {
        Self::mont_mul(&Self(value as u64, 0, 0, 0), &Self::R2)
    }
}

impl From<u64> for Scalar {
    fn from(value: u64) -> Self {
        Self::mont_mul(&Self(value, 0, 0, 0), &Self::R2)
    }
}

impl From<u128> for Scalar {
    fn from(value: u128) -> Self {
        Self::mont_mul(&Self(value as u64, (value >> 64) as u64, 0, 0), &Self::R2)
    }
}

impl TryFrom<U256> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: U256) -> Result<Self, Self::Error> {
        Self::try_from_le_bytes(&value.to_little_endian())
            .into_option()
            .context("overflow")
    }
}

impl TryFrom<usize> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self::mont_mul(&Self(value as u64, 0, 0, 0), &Self::R2))
    }
}

impl Field for Scalar {
    const LEN: usize = 32;

    const ZERO: Self = Self(0, 0, 0, 0);

    const ONE: Self = Self::R;

    const MAX: Self = Self(
        0x4000000000000003u64,
        0x135b99d7dbd1ffd7u64,
        0xfffffffffffffffau64,
        0x7fffffffffffffffu64,
    );

    fn is_odd(&self) -> Choice {
        (self.to_le_bytes()[0] & 1).into()
    }

    fn try_random<R: rand_core::TryCryptoRng>(rng: &mut R) -> Result<Self, R::Error> {
        let mut bytes = [0u8; 64];
        rng.try_fill_bytes(&mut bytes)?;
        Ok(Self::from_u512_mod_n(U512::from_little_endian(&bytes)))
    }

    fn random<R: rand_core::CryptoRng>(rng: &mut R) -> Self {
        let mut bytes = [0u8; 64];
        rng.fill_bytes(&mut bytes);
        Self::from_u512_mod_n(U512::from_little_endian(&bytes))
    }

    fn random_default() -> Self {
        let mut bytes = [0u8; 64];
        getrandom::fill(&mut bytes).unwrap();
        Self::from_h512(H512::from_slice(&bytes))
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
        value.modp()
    }

    fn invert(&self) -> CtOption<Self> {
        CtOption::new(self.pow(Scalar::MINUS_TWO), !self.is_zero())
    }

    fn invert_vartime(&self) -> Option<Self> {
        if self.is_zero().into() {
            None
        } else {
            Some(self.pow(Scalar::MINUS_TWO))
        }
    }

    fn pow(mut self, exp: Self) -> Self {
        static ONE: U256 = U256::one();
        let mut exp = exp.to_u256();
        let mut result = Self::ONE;
        for _ in 0..256 {
            let product = result * self;
            result = Scalar::conditional_select(
                &result,
                &product,
                ((!(exp & ONE).is_zero()) as u8).into(),
            );
            exp >>= 1;
            self = self.square();
        }
        result
    }

    fn pow_vartime(mut self, exp: Self) -> Self {
        static ONE: U256 = U256::one();
        let mut exp = exp.to_u256();
        let mut result = Self::ONE;
        while !exp.is_zero() {
            if !(exp & ONE).is_zero() {
                result *= self;
            }
            exp >>= 1;
            self = self.square();
        }
        result
    }

    fn div_int(&self, rhs: &Self) -> (Self, Self) {
        assert!(!bool::from(rhs.is_zero()));
        let lhs = self.to_u256();
        let rhs = rhs.to_u256();
        let (quotient, remainder) = lhs.div_mod(rhs);
        (quotient.try_into().unwrap(), remainder.try_into().unwrap())
    }

    fn try_from_le_bytes(bytes: &[u8]) -> CtOption<Self> {
        assert!(bytes.len() == 32);
        let raw = Self(
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        );
        CtOption::new(
            Self::mont_mul(&raw, &Self::R2),
            Choice::from((raw.cmp_raw(&Self::MAX_RAW) != Ordering::Greater) as u8),
        )
    }

    fn try_from_be_bytes(bytes: &[u8]) -> CtOption<Self> {
        assert!(bytes.len() == 32);
        let raw = Self(
            u64::from_be_bytes(bytes[24..32].try_into().unwrap()),
            u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_be_bytes(bytes[0..8].try_into().unwrap()),
        );
        CtOption::new(
            Self::mont_mul(&raw, &Self::R2),
            Choice::from((raw.cmp_raw(&Self::MAX_RAW) != Ordering::Greater) as u8),
        )
    }

    fn from_str_radix(s: &str, radix: usize) -> Result<Self, std::fmt::Error> {
        assert!(radix >= 2 && radix <= 36);
        if s.is_empty() {
            return Err(std::fmt::Error);
        }
        let radix_u256: U256 = radix.into();
        let mut value = U256::zero();
        for byte in s.bytes() {
            let digit = CHARACTERS_UPPER_CASE[..radix]
                .iter()
                .position(|&c| c == byte)
                .or_else(|| {
                    CHARACTERS_LOWER_CASE[..radix]
                        .iter()
                        .position(|&c| c == byte)
                })
                .ok_or(std::fmt::Error)?;
            value = value
                .checked_mul(radix_u256)
                .ok_or(std::fmt::Error)?
                .checked_add(digit.into())
                .ok_or(std::fmt::Error)?;
        }
        Scalar::try_from(value).map_err(|_| std::fmt::Error)
    }

    fn to_str_radix(&self, radix: usize, pad_to: usize, upper_case: bool) -> String {
        assert!(radix >= 2 && radix <= 36);
        match radix {
            2 | 4 | 8 | 16 | 32 => self.to_string_impl_log2(radix.ilog2(), pad_to, upper_case),
            _ => self.to_string_impl(radix, pad_to, upper_case),
        }
    }

    fn try_to_u8(&self) -> Option<u8> {
        let raw = self.to_raw();
        if (raw.1, raw.2, raw.3) != (0, 0, 0) {
            return None;
        }
        if raw.0 > u8::MAX as u64 {
            return None;
        }
        Some(raw.0 as u8)
    }

    fn try_to_u16(&self) -> Option<u16> {
        let raw = self.to_raw();
        if (raw.1, raw.2, raw.3) != (0, 0, 0) {
            return None;
        }
        if raw.0 > u16::MAX as u64 {
            return None;
        }
        Some(raw.0 as u16)
    }
}

impl Field256 for Scalar {
    fn to_le_bytes(&self) -> [u8; 32] {
        let raw = self.to_raw();
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&raw.0.to_le_bytes());
        bytes[8..16].copy_from_slice(&raw.1.to_le_bytes());
        bytes[16..24].copy_from_slice(&raw.2.to_le_bytes());
        bytes[24..32].copy_from_slice(&raw.3.to_le_bytes());
        bytes
    }

    fn to_be_bytes(&self) -> [u8; 32] {
        let raw = self.to_raw();
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&raw.3.to_be_bytes());
        bytes[8..16].copy_from_slice(&raw.2.to_be_bytes());
        bytes[16..24].copy_from_slice(&raw.1.to_be_bytes());
        bytes[24..32].copy_from_slice(&raw.0.to_be_bytes());
        bytes
    }

    fn from_u512_mod_n(u512: U512) -> Self {
        static P: LazyLock<U512> = LazyLock::new(|| Scalar::MODULUS.parse().unwrap());
        let value = u512 % *P;
        let bytes = value.to_little_endian();
        Scalar::try_from_le_bytes(&bytes[0..32]).unwrap()
    }

    fn from_h512(h512: H512) -> Self {
        let u512 = U512::from_little_endian(h512.as_bytes());
        Self::from_u512_mod_n(u512)
    }

    fn try_to_u32(&self) -> CtOption<u32> {
        let raw = self.to_raw();
        CtOption::new(
            raw.0 as u32,
            Choice::from((raw.0 <= u32::MAX as u64) as u8)
                & Choice::from((raw.1 == 0) as u8)
                & Choice::from((raw.2 == 0) as u8)
                & Choice::from((raw.3 == 0) as u8),
        )
    }

    fn try_to_u64(&self) -> CtOption<u64> {
        let raw = self.to_raw();
        CtOption::new(
            raw.0 as u64,
            Choice::from((raw.1 == 0) as u8)
                & Choice::from((raw.2 == 0) as u8)
                & Choice::from((raw.3 == 0) as u8),
        )
    }

    fn try_to_u128(&self) -> CtOption<u128> {
        let raw = self.to_raw();
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&raw.0.to_le_bytes());
        bytes[8..16].copy_from_slice(&raw.1.to_le_bytes());
        CtOption::new(
            u128::from_le_bytes(bytes),
            Choice::from((raw.2 == 0) as u8) & Choice::from((raw.3 == 0) as u8),
        )
    }

    fn to_u256(&self) -> U256 {
        U256::from_little_endian(&self.to_le_bytes())
    }

    fn to_u512(&self) -> U512 {
        let mut bytes = [0u8; 64];
        bytes[0..32].copy_from_slice(&self.to_le_bytes());
        U512::from_little_endian(&bytes)
    }
}

impl PrimeField for Scalar {
    const MODULUS: &'static str =
        "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000001";

    const S: usize = 62;

    const MULTIPLICATIVE_GENERATOR: Self = Self(
        0x7ffffffffffffff6u64,
        0xbf795485cdeeab32u64,
        0x0000000000000013u64,
        0x0000000000000000u64,
    );

    const MINUS_TWO: Self = Self(
        0xc000000000000005u64,
        0x204355bd1908aa66u64,
        0xfffffffffffffff6u64,
        0x7fffffffffffffffu64,
    );

    const TWO_INV: Self = Self(
        0x3fffffffffffffffu64,
        0xf98c220d6164aab8u64,
        0x0000000000000001u64,
        0x0000000000000000u64,
    );

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

impl PrimeField256 for Scalar {}

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

    #[test]
    fn test_from_const() {
        assert_eq!(from_const(0), Scalar::ZERO);
        assert_eq!(from_const(1), Scalar::ONE);
        assert_eq!(
            from_const(0).to_string(),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            from_const(1).to_string(),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(
            from_const(2).to_string(),
            "0x0000000000000000000000000000000000000000000000000000000000000002"
        );
        assert_eq!(
            from_const(15).to_string(),
            "0x000000000000000000000000000000000000000000000000000000000000000f"
        );
        assert_eq!(
            from_const(16).to_string(),
            "0x0000000000000000000000000000000000000000000000000000000000000010"
        );
        assert_eq!(
            from_const(17).to_string(),
            "0x0000000000000000000000000000000000000000000000000000000000000011"
        );
        assert_eq!(
            from_const(u64::MAX - 1).to_string(),
            "0x000000000000000000000000000000000000000000000000fffffffffffffffe"
        );
        assert_eq!(
            from_const(u64::MAX).to_string(),
            "0x000000000000000000000000000000000000000000000000ffffffffffffffff"
        );
    }

    #[test]
    fn test_modulus() {
        assert_eq!(
            format!("{:#066x}", Scalar::MAX),
            format!("{:#066x}", Scalar::MODULUS.parse::<U256>().unwrap() - 1)
        );
    }

    #[test]
    fn test_zero() {
        assert_eq!(Scalar::ZERO, Scalar::zero());
        assert_eq!(Scalar::ZERO, from_const(0));
        assert_eq!(Scalar::ZERO + from_const(0), from_const(0));
        assert_eq!(Scalar::ZERO + from_const(1), from_const(1));
        assert_eq!(Scalar::ZERO + from_const(2), from_const(2));
        assert_eq!(Scalar::ZERO + from_const(3), from_const(3));
        assert_eq!(Scalar::ZERO * from_const(0), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * from_const(1), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * from_const(2), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * from_const(3), Scalar::ZERO);
    }

    #[test]
    fn test_one() {
        assert_eq!(Scalar::ONE, Scalar::R);
        assert_eq!(Scalar::ONE, Scalar::one());
        assert_eq!(Scalar::ONE, from_const(1));
        assert_eq!(Scalar::ONE + from_const(0), from_const(1));
        assert_eq!(Scalar::ONE + from_const(1), from_const(2));
        assert_eq!(Scalar::ONE + from_const(2), from_const(3));
        assert_eq!(Scalar::ONE + from_const(3), from_const(4));
        assert_eq!(Scalar::ONE * from_const(0), from_const(0));
        assert_eq!(Scalar::ONE * from_const(1), from_const(1));
        assert_eq!(Scalar::ONE * from_const(2), from_const(2));
        assert_eq!(Scalar::ONE * from_const(3), from_const(3));
    }

    #[test]
    fn test_max() {
        assert_eq!(Scalar::MAX, -Scalar::ONE);
    }

    #[test]
    fn test_fmt_display() {
        assert_eq!(
            format!("{}", from_const(0)),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            format!("{}", from_const(1)),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(
            format!("{}", from_const(2)),
            "0x0000000000000000000000000000000000000000000000000000000000000002"
        );
        assert_eq!(
            format!(
                "{}",
                parse_scalar("0x17386c7200968ccab11e0a32e9b8c520b89637cc9b71975efe17b59138fe9c7b")
            ),
            "0x17386c7200968ccab11e0a32e9b8c520b89637cc9b71975efe17b59138fe9c7b"
        );
        assert_eq!(
            format!("{}", Scalar::MAX - Scalar::ONE),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547bfffffffffffffff"
        );
        assert_eq!(
            format!("{}", Scalar::MAX),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
        );
    }

    #[test]
    fn test_fmt_debug() {
        assert_eq!(
            format!("{:?}", from_const(0)),
            "Scalar(0x0000000000000000000000000000000000000000000000000000000000000000)"
        );
        assert_eq!(
            format!("{:?}", from_const(1)),
            "Scalar(0x0000000000000000000000000000000000000000000000000000000000000001)"
        );
        assert_eq!(
            format!("{:?}", from_const(2)),
            "Scalar(0x0000000000000000000000000000000000000000000000000000000000000002)"
        );
        assert_eq!(
            format!(
                "{:?}",
                parse_scalar("0x17386c7200968ccab11e0a32e9b8c520b89637cc9b71975efe17b59138fe9c7b")
            ),
            "Scalar(0x17386c7200968ccab11e0a32e9b8c520b89637cc9b71975efe17b59138fe9c7b)"
        );
        assert_eq!(
            format!("{:?}", Scalar::MAX - Scalar::ONE),
            "Scalar(0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547bfffffffffffffff)"
        );
        assert_eq!(
            format!("{:?}", Scalar::MAX),
            "Scalar(0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000)"
        );
    }

    #[test]
    fn test_fmt_lower_hex() {
        assert_eq!(format!("{:x}", from_const(0)), "0");
        assert_eq!(format!("{:x}", from_const(1)), "1");
        assert_eq!(format!("{:x}", from_const(0xdeadbeef)), "deadbeef");
        assert_eq!(format!("{:#x}", from_const(0)), "0x0");
        assert_eq!(format!("{:#x}", from_const(0xdeadbeef)), "0xdeadbeef");
        assert_eq!(format!("{:10x}", from_const(0xdeadbeef)), "  deadbeef");
        assert_eq!(format!("{:010x}", from_const(0xdeadbeef)), "00deadbeef");
        assert_eq!(format!("{:#012x}", from_const(0xdeadbeef)), "0x00deadbeef");
        assert_eq!(format!("{:<10x}", from_const(0xdeadbeef)), "deadbeef  ");
        assert_eq!(format!("{:_<10x}", from_const(0xdeadbeef)), "deadbeef__");
    }

    #[test]
    fn test_fmt_upper_hex() {
        assert_eq!(format!("{:X}", from_const(0)), "0");
        assert_eq!(format!("{:X}", from_const(0xdeadbeef)), "DEADBEEF");
        assert_eq!(format!("{:#X}", from_const(0xdeadbeef)), "0xDEADBEEF");
        assert_eq!(format!("{:010X}", from_const(0xdeadbeef)), "00DEADBEEF");
        assert_eq!(format!("{:#012X}", from_const(0xdeadbeef)), "0x00DEADBEEF");
        assert_eq!(format!("{:<10X}", from_const(0xdeadbeef)), "DEADBEEF  ");
    }

    #[test]
    fn test_fmt_binary() {
        assert_eq!(format!("{:b}", from_const(0)), "0");
        assert_eq!(format!("{:b}", from_const(1)), "1");
        assert_eq!(format!("{:b}", from_const(0b1010)), "1010");
        assert_eq!(format!("{:#b}", from_const(0b1010)), "0b1010");
        assert_eq!(format!("{:10b}", from_const(0b1010)), "      1010");
        assert_eq!(format!("{:010b}", from_const(0b1010)), "0000001010");
        assert_eq!(format!("{:#012b}", from_const(0b1010)), "0b0000001010");
        assert_eq!(format!("{:<10b}", from_const(0b1010)), "1010      ");
    }

    #[test]
    fn test_fmt_octal() {
        assert_eq!(format!("{:o}", from_const(0)), "0");
        assert_eq!(format!("{:o}", from_const(1)), "1");
        assert_eq!(format!("{:o}", from_const(0o755)), "755");
        assert_eq!(format!("{:#o}", from_const(0o755)), "0o755");
        assert_eq!(format!("{:10o}", from_const(0o755)), "       755");
        assert_eq!(format!("{:010o}", from_const(0o755)), "0000000755");
        assert_eq!(format!("{:#012o}", from_const(0o755)), "0o0000000755");
        assert_eq!(format!("{:<10o}", from_const(0o755)), "755       ");
    }

    #[test]
    fn test_equality() {
        assert!(from_const(0) == from_const(0));
        assert!(from_const(0) != from_const(1));
        assert!(from_const(0) != from_const(2));
        assert!(from_const(0) != Scalar::MAX - Scalar::ONE);
        assert!(from_const(0) != Scalar::MAX);
        assert!(from_const(1) != from_const(0));
        assert!(from_const(1) == from_const(1));
        assert!(from_const(1) != from_const(2));
        assert!(from_const(0) != Scalar::MAX - Scalar::ONE);
        assert!(from_const(0) != Scalar::MAX);
        assert!(from_const(2) != from_const(0));
        assert!(from_const(2) != from_const(1));
        assert!(from_const(2) == from_const(2));
        assert!(from_const(0) != Scalar::MAX - Scalar::ONE);
        assert!(from_const(0) != Scalar::MAX);
        assert!(Scalar::MAX - Scalar::ONE != from_const(0));
        assert!(Scalar::MAX - Scalar::ONE != from_const(1));
        assert!(Scalar::MAX - Scalar::ONE != from_const(2));
        assert!(Scalar::MAX - Scalar::ONE == Scalar::MAX - Scalar::ONE);
        assert!(Scalar::MAX - Scalar::ONE != Scalar::MAX);
        assert!(Scalar::MAX != from_const(0));
        assert!(Scalar::MAX != from_const(1));
        assert!(Scalar::MAX != from_const(2));
        assert!(Scalar::MAX != Scalar::MAX - Scalar::ONE);
        assert!(Scalar::MAX == Scalar::MAX);
    }

    #[test]
    fn test_total_order() {
        let v0 = from_const(0);
        let v1 = from_const(1);
        let v2 = from_const(42);
        let v3 = parse_scalar("0x318c1df8459d125dc54e1fe487bf23e8430221b69660d8ca9427235713f24de1");
        let v4 = Scalar::MAX - Scalar::ONE;
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
    fn test_ct_eq() {
        let a = from_const(42);
        let b = from_const(42);
        let c = from_const(43);

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
        let v0 = from_const(0);
        let v1 = from_const(1);
        let v2 = from_const(42);
        let v3 = Scalar::MAX - Scalar::ONE;
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
        let v0 = from_const(0);
        let v1 = from_const(1);
        let v2 = from_const(42);
        let v3 = Scalar::MAX - Scalar::ONE;
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
        let a = from_const(12);
        let b = from_const(34);
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

    fn test_neg_impl(value: Scalar) {
        assert_eq!(-value, Scalar::MAX - value + Scalar::ONE);
    }

    #[test]
    fn test_neg() {
        assert_eq!(-Scalar::ZERO, Scalar::ZERO);
        assert_eq!(-Scalar::ONE, Scalar::MAX);
        assert_eq!(-from_const(2), Scalar::MAX - Scalar::ONE);
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

    #[test]
    fn test_mul_by_zero() {
        assert_eq!(Scalar::ZERO * from_const(42), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * &from_const(42), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * from_const(43), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * &from_const(43), Scalar::ZERO);
        assert_eq!(from_const(42) * Scalar::ZERO, Scalar::ZERO);
        assert_eq!(from_const(42) * &Scalar::ZERO, Scalar::ZERO);
        assert_eq!(from_const(43) * Scalar::ZERO, Scalar::ZERO);
        assert_eq!(from_const(43) * &Scalar::ZERO, Scalar::ZERO);
    }

    #[test]
    fn test_mul_by_one() {
        assert_eq!(Scalar::ONE * from_const(42), from_const(42));
        assert_eq!(Scalar::ONE * &from_const(42), from_const(42));
        assert_eq!(Scalar::ONE * from_const(43), from_const(43));
        assert_eq!(Scalar::ONE * &from_const(43), from_const(43));
        assert_eq!(from_const(42) * Scalar::ONE, from_const(42));
        assert_eq!(from_const(42) * &Scalar::ONE, from_const(42));
        assert_eq!(from_const(43) * Scalar::ONE, from_const(43));
        assert_eq!(from_const(43) * &Scalar::ONE, from_const(43));
    }

    #[test]
    fn test_mul() {
        assert_eq!(from_const(12) * from_const(34), from_const(408));
        assert_eq!(from_const(12) * &from_const(34), from_const(408));
        assert_eq!(from_const(12) * from_const(56), from_const(672));
        assert_eq!(from_const(12) * &from_const(56), from_const(672));
        assert_eq!(from_const(34) * from_const(12), from_const(408));
        assert_eq!(from_const(34) * &from_const(12), from_const(408));
        assert_eq!(from_const(56) * from_const(12), from_const(672));
        assert_eq!(from_const(56) * &from_const(12), from_const(672));
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
            parse_scalar("0x4433ff6315d939cda16f055756432036cab445af8186bc60b243127905b84c73"),
            parse_scalar("0x6bb08da54b6c7a0760e3a845a73c45dd643d18fd594ddee7c1ab58c4af38310b"),
        );
    }

    #[test]
    fn test_div_by_one() {
        assert_eq!(Scalar::ONE / from_const(42), from_const(42).invert_unwrap());
        assert_eq!(
            Scalar::ONE / &from_const(42),
            from_const(42).invert_unwrap()
        );
        assert_eq!(Scalar::ONE / from_const(43), from_const(43).invert_unwrap());
        assert_eq!(
            Scalar::ONE / &from_const(43),
            from_const(43).invert_unwrap()
        );
        assert_eq!(from_const(42) / Scalar::ONE, from_const(42));
        assert_eq!(from_const(42) / &Scalar::ONE, from_const(42));
        assert_eq!(from_const(43) / Scalar::ONE, from_const(43));
        assert_eq!(from_const(43) / &Scalar::ONE, from_const(43));
    }

    #[test]
    fn test_div() {
        assert_eq!(from_const(408) / from_const(34), from_const(12));
        assert_eq!(from_const(408) / &from_const(34), from_const(12));
        assert_eq!(from_const(672) / from_const(56), from_const(12));
        assert_eq!(from_const(672) / &from_const(56), from_const(12));
        assert_eq!(from_const(408) / from_const(12), from_const(34));
        assert_eq!(from_const(408) / &from_const(12), from_const(34));
        assert_eq!(from_const(672) / from_const(12), from_const(56));
        assert_eq!(from_const(672) / &from_const(12), from_const(56));
    }

    #[test]
    fn test_sum_owned() {
        let values = vec![Scalar::ONE, from_const(2), from_const(3)];
        assert_eq!(values.into_iter().sum::<Scalar>(), from_const(6));
    }

    #[test]
    fn test_sum_refs() {
        let values = vec![Scalar::ONE, from_const(2), from_const(3)];
        assert_eq!(values.iter().sum::<Scalar>(), from_const(6));
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
        let values = vec![from_const(42)];
        assert_eq!(values.into_iter().sum::<Scalar>(), from_const(42));
    }

    #[test]
    fn test_sum_wraps_modulo_p() {
        let values = vec![Scalar::MAX, Scalar::ONE];
        assert_eq!(values.into_iter().sum::<Scalar>(), Scalar::ZERO);
    }

    #[test]
    fn test_product_owned() {
        let values = vec![from_const(2), from_const(3), from_const(4)];
        assert_eq!(values.into_iter().product::<Scalar>(), from_const(24));
    }

    #[test]
    fn test_product_refs() {
        let values = vec![from_const(2), from_const(3), from_const(4)];
        assert_eq!(values.iter().product::<Scalar>(), from_const(24));
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
        let values = vec![from_const(42)];
        assert_eq!(values.into_iter().product::<Scalar>(), from_const(42));
    }

    #[test]
    fn test_product_with_zero() {
        let values = vec![from_const(5), Scalar::ZERO, from_const(7)];
        assert_eq!(values.into_iter().product::<Scalar>(), Scalar::ZERO);
    }

    #[test]
    fn test_product_with_one() {
        let values = vec![Scalar::ONE, from_const(5), Scalar::ONE];
        assert_eq!(values.into_iter().product::<Scalar>(), from_const(5));
    }

    #[test]
    fn test_from_u8() {
        assert_eq!(Scalar::from(0u8), from_const(0));
        assert_eq!(Scalar::from(1u8), from_const(1));
        assert_eq!(Scalar::from(2u8), from_const(2));
        assert_eq!(Scalar::from(u8::MAX - 1), from_const((u8::MAX - 1) as u64));
        assert_eq!(Scalar::from(u8::MAX), from_const(u8::MAX as u64));
    }

    #[test]
    fn test_from_u16() {
        assert_eq!(Scalar::from(0u16), from_const(0));
        assert_eq!(Scalar::from(1u16), from_const(1));
        assert_eq!(Scalar::from(2u16), from_const(2));
        assert_eq!(
            Scalar::from(u16::MAX - 1),
            from_const((u16::MAX - 1) as u64)
        );
        assert_eq!(Scalar::from(u16::MAX), from_const(u16::MAX as u64));
    }

    #[test]
    fn test_from_u32() {
        assert_eq!(Scalar::from(0u32), from_const(0));
        assert_eq!(Scalar::from(1u32), from_const(1));
        assert_eq!(Scalar::from(2u32), from_const(2));
        assert_eq!(
            Scalar::from(u32::MAX - 1),
            from_const((u32::MAX - 1) as u64)
        );
        assert_eq!(Scalar::from(u32::MAX), from_const(u32::MAX as u64));
    }

    #[test]
    fn test_from_u64() {
        assert_eq!(Scalar::from(0u64), from_const(0));
        assert_eq!(Scalar::from(1u64), from_const(1));
        assert_eq!(Scalar::from(2u64), from_const(2));
        assert_eq!(Scalar::from(u64::MAX - 1), from_const(u64::MAX - 1));
        assert_eq!(Scalar::from(u64::MAX), from_const(u64::MAX));
    }

    #[test]
    fn test_from_u128() {
        assert_eq!(Scalar::from(0u128), from_const(0));
        assert_eq!(Scalar::from(1u128), from_const(1));
        assert_eq!(Scalar::from(2u128), from_const(2));
        assert_eq!(
            Scalar::from(u128::MAX - 1),
            parse_scalar("0x00000000000000000000000000000000fffffffffffffffffffffffffffffffe")
        );
        assert_eq!(
            Scalar::from(u128::MAX),
            parse_scalar("0x00000000000000000000000000000000ffffffffffffffffffffffffffffffff")
        );
    }

    #[test]
    fn test_try_from_u256() {
        assert_eq!(
            <Scalar as TryFrom<U256>>::try_from(0.into()).unwrap(),
            from_const(0)
        );
        assert_eq!(
            <Scalar as TryFrom<U256>>::try_from(1.into()).unwrap(),
            from_const(1)
        );
        assert_eq!(
            <Scalar as TryFrom<U256>>::try_from(2.into()).unwrap(),
            from_const(2)
        );
        let modulus: U256 = Scalar::MODULUS.parse().unwrap();
        assert_eq!(
            <Scalar as TryFrom<U256>>::try_from(modulus - 2).unwrap(),
            -from_const(2)
        );
        assert_eq!(
            <Scalar as TryFrom<U256>>::try_from(modulus - 1).unwrap(),
            -from_const(1)
        );
        assert!(<Scalar as TryFrom<U256>>::try_from(modulus).is_err());
        assert!(<Scalar as TryFrom<U256>>::try_from(modulus + 1).is_err());
    }

    #[test]
    fn test_try_from_usize() {
        assert_eq!(Scalar::try_from(0usize).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(1usize).unwrap(), from_const(1));
        assert_eq!(Scalar::try_from(2usize).unwrap(), from_const(2));
        assert_eq!(
            Scalar::try_from(usize::MAX - 1).unwrap(),
            from_const(usize::MAX as u64 - 1)
        );
        assert_eq!(
            Scalar::try_from(usize::MAX).unwrap(),
            from_const(usize::MAX as u64)
        );
    }

    #[test]
    fn test_is_zero() {
        assert!(bool::from(from_const(0).is_zero()));
        assert!(!bool::from(from_const(1).is_zero()));
        assert!(!bool::from(from_const(2).is_zero()));
        assert!(!bool::from((Scalar::MAX - Scalar::ONE).is_zero()));
        assert!(!bool::from(Scalar::MAX.is_zero()));
    }

    #[test]
    fn test_is_even() {
        assert!(bool::from(from_const(0).is_even()));
        assert!(!bool::from(from_const(1).is_even()));
        assert!(bool::from(from_const(2).is_even()));
        assert!(!bool::from(from_const(3).is_even()));
        assert!(bool::from(from_const(100).is_even()));
        assert!(!bool::from(from_const(101).is_even()));
        assert!(!bool::from((Scalar::MAX - Scalar::ONE).is_even()));
        assert!(bool::from(Scalar::MAX.is_even()));
    }

    #[test]
    fn test_is_odd() {
        assert!(!bool::from(from_const(0).is_odd()));
        assert!(bool::from(from_const(1).is_odd()));
        assert!(!bool::from(from_const(2).is_odd()));
        assert!(bool::from(from_const(3).is_odd()));
        assert!(!bool::from(from_const(100).is_odd()));
        assert!(bool::from(from_const(101).is_odd()));
        assert!(bool::from((Scalar::MAX - Scalar::ONE).is_odd()));
        assert!(!bool::from(Scalar::MAX.is_odd()));
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

    impl rand_core::TryCryptoRng for OsRng {}

    #[test]
    fn test_try_random() {
        let mut rng = OsRng;
        assert_ne!(
            Scalar::try_random(&mut rng).unwrap(),
            Scalar::try_random(&mut rng).unwrap()
        );
        assert_ne!(
            Scalar::try_random(&mut rng).unwrap(),
            Scalar::try_random(&mut rng).unwrap()
        );
        assert_ne!(
            Scalar::try_random(&mut rng).unwrap(),
            Scalar::try_random(&mut rng).unwrap()
        );
    }

    #[test]
    fn test_random() {
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
    fn test_double() {
        assert_eq!(from_const(0).double(), from_const(0));
        assert_eq!(from_const(1).double(), from_const(2));
        assert_eq!(from_const(2).double(), from_const(4));
        assert_eq!((Scalar::MAX - from_const(2)).double(), -from_const(6));
        assert_eq!((Scalar::MAX - from_const(1)).double(), -from_const(4));
        assert_eq!((Scalar::MAX).double(), -from_const(2));
    }

    #[test]
    fn test_square() {
        assert_eq!(from_const(0).square(), from_const(0));
        assert_eq!(from_const(1).square(), from_const(1));
        assert_eq!(from_const(2).square(), from_const(4));
        assert_eq!((Scalar::MAX - from_const(2)).square(), from_const(9));
        assert_eq!((Scalar::MAX - from_const(1)).square(), from_const(4));
        assert_eq!((Scalar::MAX).square(), from_const(1));
    }

    #[test]
    fn test_cube() {
        assert_eq!(from_const(0).cube(), from_const(0));
        assert_eq!(from_const(1).cube(), from_const(1));
        assert_eq!(from_const(2).cube(), from_const(8));
        assert_eq!((Scalar::MAX - from_const(2)).cube(), -from_const(27));
        assert_eq!((Scalar::MAX - from_const(1)).cube(), -from_const(8));
        assert_eq!((Scalar::MAX).cube(), -from_const(1));
    }

    fn test_inversion_impl(value: Scalar) {
        assert_ne!(value, Scalar::ZERO);
        assert_eq!(value * value.invert().unwrap(), Scalar::ONE);
        assert_eq!(value * value.invert_unwrap(), Scalar::ONE);
        assert_eq!(value * value.invert_or_zero(), Scalar::ONE);
        assert_eq!(value * value.invert_vartime().unwrap(), Scalar::ONE);
    }

    #[test]
    fn test_inversion() {
        assert!(from_const(0).invert_vartime().is_none());
        assert_eq!(from_const(0).invert_or_zero(), Scalar::ZERO);
        assert!(bool::from(from_const(0).invert().is_none()));
        test_inversion_impl(1u64.into());
        test_inversion_impl(2u64.into());
        test_inversion_impl(42u64.into());
        test_inversion_impl(Scalar::MAX);
    }

    fn test_invert_batch_impl(values: &[Scalar]) {
        let expected: Vec<Scalar> = values
            .iter()
            .map(|value| value.invert_vartime().unwrap())
            .collect();

        let mut batch = values.to_vec();
        Scalar::invert_batch(&mut batch);
        assert_eq!(batch, expected);

        batch = values.to_vec();
        Scalar::invert_batch_vartime(&mut batch);
        assert_eq!(batch, expected);
    }

    #[test]
    fn test_invert_batch() {
        test_invert_batch_impl(&[]);
        test_invert_batch_impl(&[Scalar::ONE]);
        test_invert_batch_impl(&[from_const(42)]);
        test_invert_batch_impl(&[Scalar::MAX]);
        test_invert_batch_impl(&[from_const(1), from_const(2), from_const(3)]);
        test_invert_batch_impl(&[
            from_const(42),
            Scalar::ONE,
            Scalar::MAX,
            from_const(1),
            from_const(2),
            Scalar::MAX - from_const(1),
        ]);
    }

    #[test]
    fn test_power() {
        assert_eq!(from_const(0).pow(from_const(0)), from_const(1));
        assert_eq!(from_const(0).pow(from_const(1)), from_const(0));
        assert_eq!(from_const(0).pow(from_const(2)), from_const(0));
        assert_eq!(from_const(1).pow(from_const(0)), from_const(1));
        assert_eq!(from_const(1).pow(from_const(1)), from_const(1));
        assert_eq!(from_const(1).pow(from_const(2)), from_const(1));
        assert_eq!(from_const(2).pow(from_const(0)), from_const(1));
        assert_eq!(from_const(2).pow(from_const(1)), from_const(2));
        assert_eq!(from_const(2).pow(from_const(2)), from_const(4));
        assert_eq!(from_const(2).pow(from_const(3)), from_const(8));
    }

    #[test]
    fn test_small_power() {
        assert_eq!(from_const(0).pow_small(0), from_const(1));
        assert_eq!(from_const(0).pow_small(1), from_const(0));
        assert_eq!(from_const(0).pow_small(2), from_const(0));
        assert_eq!(from_const(1).pow_small(0), from_const(1));
        assert_eq!(from_const(1).pow_small(1), from_const(1));
        assert_eq!(from_const(1).pow_small(2), from_const(1));
        assert_eq!(from_const(2).pow_small(0), from_const(1));
        assert_eq!(from_const(2).pow_small(1), from_const(2));
        assert_eq!(from_const(2).pow_small(2), from_const(4));
        assert_eq!(from_const(2).pow_small(3), from_const(8));
    }

    #[test]
    fn test_power_vartime() {
        assert_eq!(from_const(0).pow_vartime(from_const(0)), from_const(1));
        assert_eq!(from_const(0).pow_vartime(from_const(1)), from_const(0));
        assert_eq!(from_const(0).pow_vartime(from_const(2)), from_const(0));
        assert_eq!(from_const(1).pow_vartime(from_const(0)), from_const(1));
        assert_eq!(from_const(1).pow_vartime(from_const(1)), from_const(1));
        assert_eq!(from_const(1).pow_vartime(from_const(2)), from_const(1));
        assert_eq!(from_const(2).pow_vartime(from_const(0)), from_const(1));
        assert_eq!(from_const(2).pow_vartime(from_const(1)), from_const(2));
        assert_eq!(from_const(2).pow_vartime(from_const(2)), from_const(4));
        assert_eq!(from_const(2).pow_vartime(from_const(3)), from_const(8));
    }

    #[test]
    fn test_small_power_vartime() {
        assert_eq!(from_const(0).pow_small_vartime(0), from_const(1));
        assert_eq!(from_const(0).pow_small_vartime(1), from_const(0));
        assert_eq!(from_const(0).pow_small_vartime(2), from_const(0));
        assert_eq!(from_const(1).pow_small_vartime(0), from_const(1));
        assert_eq!(from_const(1).pow_small_vartime(1), from_const(1));
        assert_eq!(from_const(1).pow_small_vartime(2), from_const(1));
        assert_eq!(from_const(2).pow_small_vartime(0), from_const(1));
        assert_eq!(from_const(2).pow_small_vartime(1), from_const(2));
        assert_eq!(from_const(2).pow_small_vartime(2), from_const(4));
        assert_eq!(from_const(2).pow_small_vartime(3), from_const(8));
    }

    #[test]
    fn test_integer_division() {
        assert_eq!(
            from_const(13).div_int(&from_const(5)),
            (from_const(2), from_const(3))
        );
        assert_eq!(
            from_const(61).div_int(&from_const(7)),
            (from_const(8), from_const(5))
        );
    }

    #[test]
    fn test_try_from_le_bytes() {
        assert_eq!(
            Scalar::try_from_le_bytes(&[
                134, 217, 203, 162, 4, 73, 55, 251, 211, 179, 190, 229, 147, 65, 246, 233, 246, 34,
                124, 231, 166, 122, 247, 92, 185, 41, 60, 53, 21, 52, 225, 38
            ])
            .unwrap(),
            parse_scalar("0x26e13415353c29b95cf77aa6e77c22f6e9f64193e5beb3d3fb374904a2cbd986")
        );
        assert_eq!(
            Scalar::try_from_le_bytes(&[
                94, 15, 32, 74, 182, 189, 242, 78, 168, 143, 91, 154, 184, 98, 85, 163, 142, 220,
                154, 67, 53, 216, 247, 158, 226, 97, 86, 13, 82, 137, 175, 54
            ])
            .unwrap(),
            parse_scalar("0x36af89520d5661e29ef7d835439adc8ea35562b89a5b8fa84ef2bdb64a200f5e")
        );
    }

    #[test]
    fn test_try_from_be_bytes() {
        assert_eq!(
            Scalar::try_from_be_bytes(&[
                79, 234, 193, 25, 67, 246, 138, 164, 3, 189, 35, 158, 1, 117, 190, 191, 241, 43,
                207, 155, 72, 12, 169, 119, 131, 204, 73, 22, 224, 246, 241, 210
            ])
            .unwrap(),
            parse_scalar("0x4feac11943f68aa403bd239e0175bebff12bcf9b480ca97783cc4916e0f6f1d2")
        );
        assert_eq!(
            Scalar::try_from_be_bytes(&[
                1, 116, 38, 26, 115, 42, 57, 217, 140, 177, 67, 128, 123, 20, 67, 224, 47, 204,
                201, 64, 41, 58, 242, 162, 99, 50, 143, 217, 160, 22, 179, 45
            ])
            .unwrap(),
            parse_scalar("0x0174261a732a39d98cb143807b1443e02fccc940293af2a263328fd9a016b32d")
        );
    }

    #[test]
    fn test_parse_binary() {
        assert_eq!(Scalar::from_str_radix("0", 2).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("1", 2).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("00", 2).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("01", 2).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("10", 2).unwrap(), from_const(2));
        assert_eq!(Scalar::from_str_radix("11", 2).unwrap(), from_const(3));
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
        assert_eq!(from_const(0).to_str_radix(2, 0, false), "0");
        assert_eq!(from_const(1).to_str_radix(2, 0, false), "1");
        assert_eq!(from_const(2).to_str_radix(2, 0, false), "10");
        assert_eq!(from_const(3).to_str_radix(2, 0, false), "11");
        assert_eq!(from_const(0).to_str_radix(2, 1, false), "0");
        assert_eq!(from_const(1).to_str_radix(2, 1, false), "1");
        assert_eq!(from_const(2).to_str_radix(2, 1, false), "10");
        assert_eq!(from_const(3).to_str_radix(2, 1, false), "11");
        assert_eq!(from_const(0).to_str_radix(2, 2, false), "00");
        assert_eq!(from_const(1).to_str_radix(2, 2, false), "01");
        assert_eq!(from_const(2).to_str_radix(2, 2, false), "10");
        assert_eq!(from_const(3).to_str_radix(2, 2, false), "11");
        assert_eq!(from_const(0).to_str_radix(2, 3, false), "000");
        assert_eq!(from_const(1).to_str_radix(2, 3, false), "001");
        assert_eq!(from_const(2).to_str_radix(2, 3, false), "010");
        assert_eq!(from_const(3).to_str_radix(2, 3, false), "011");
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
        assert_eq!(Scalar::from_str_radix("0", 8).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("1", 8).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("2", 8).unwrap(), from_const(2));
        assert_eq!(Scalar::from_str_radix("6", 8).unwrap(), from_const(6));
        assert_eq!(Scalar::from_str_radix("7", 8).unwrap(), from_const(7));
        assert!(Scalar::from_str_radix("8", 8).is_err());
        assert_eq!(Scalar::from_str_radix("00", 8).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("01", 8).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("02", 8).unwrap(), from_const(2));
        assert_eq!(Scalar::from_str_radix("10", 8).unwrap(), from_const(8));
        assert_eq!(Scalar::from_str_radix("11", 8).unwrap(), from_const(9));
        assert_eq!(Scalar::from_str_radix("12", 8).unwrap(), from_const(10));
        assert_eq!(Scalar::from_str_radix("20", 8).unwrap(), from_const(16));
        assert_eq!(Scalar::from_str_radix("21", 8).unwrap(), from_const(17));
        assert_eq!(Scalar::from_str_radix("22", 8).unwrap(), from_const(18));
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
        assert_eq!(from_const(0).to_str_radix(8, 0, false), "0");
        assert_eq!(from_const(1).to_str_radix(8, 0, false), "1");
        assert_eq!(from_const(2).to_str_radix(8, 0, false), "2");
        assert_eq!(from_const(6).to_str_radix(8, 0, false), "6");
        assert_eq!(from_const(7).to_str_radix(8, 0, false), "7");
        assert_eq!(from_const(8).to_str_radix(8, 0, false), "10");
        assert_eq!(from_const(9).to_str_radix(8, 0, false), "11");
        assert_eq!(from_const(10).to_str_radix(8, 0, false), "12");
        assert_eq!(from_const(0).to_str_radix(8, 1, false), "0");
        assert_eq!(from_const(1).to_str_radix(8, 1, false), "1");
        assert_eq!(from_const(2).to_str_radix(8, 1, false), "2");
        assert_eq!(from_const(6).to_str_radix(8, 1, false), "6");
        assert_eq!(from_const(7).to_str_radix(8, 1, false), "7");
        assert_eq!(from_const(8).to_str_radix(8, 1, false), "10");
        assert_eq!(from_const(9).to_str_radix(8, 1, false), "11");
        assert_eq!(from_const(10).to_str_radix(8, 1, false), "12");
        assert_eq!(from_const(0).to_str_radix(8, 2, false), "00");
        assert_eq!(from_const(1).to_str_radix(8, 2, false), "01");
        assert_eq!(from_const(2).to_str_radix(8, 2, false), "02");
        assert_eq!(from_const(6).to_str_radix(8, 2, false), "06");
        assert_eq!(from_const(7).to_str_radix(8, 2, false), "07");
        assert_eq!(from_const(8).to_str_radix(8, 2, false), "10");
        assert_eq!(from_const(9).to_str_radix(8, 2, false), "11");
        assert_eq!(from_const(10).to_str_radix(8, 2, false), "12");
        assert_eq!(from_const(0).to_str_radix(8, 3, false), "000");
        assert_eq!(from_const(1).to_str_radix(8, 3, false), "001");
        assert_eq!(from_const(2).to_str_radix(8, 3, false), "002");
        assert_eq!(from_const(6).to_str_radix(8, 3, false), "006");
        assert_eq!(from_const(7).to_str_radix(8, 3, false), "007");
        assert_eq!(from_const(8).to_str_radix(8, 3, false), "010");
        assert_eq!(from_const(9).to_str_radix(8, 3, false), "011");
        assert_eq!(from_const(10).to_str_radix(8, 3, false), "012");
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
    fn test_parse_decimal() {
        assert_eq!(Scalar::from_str_radix("0", 10).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("1", 10).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("2", 10).unwrap(), from_const(2));
        assert_eq!(Scalar::from_str_radix("00", 10).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("01", 10).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("02", 10).unwrap(), from_const(2));
        assert_eq!(Scalar::from_str_radix("10", 10).unwrap(), from_const(10));
        assert_eq!(Scalar::from_str_radix("11", 10).unwrap(), from_const(11));
        assert_eq!(Scalar::from_str_radix("12", 10).unwrap(), from_const(12));
        assert_eq!(Scalar::from_str_radix("20", 10).unwrap(), from_const(20));
        assert_eq!(Scalar::from_str_radix("21", 10).unwrap(), from_const(21));
        assert_eq!(Scalar::from_str_radix("22", 10).unwrap(), from_const(22));
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
        assert_eq!(from_const(0).to_str_radix(10, 0, false), "0");
        assert_eq!(from_const(1).to_str_radix(10, 0, false), "1");
        assert_eq!(from_const(2).to_str_radix(10, 0, false), "2");
        assert_eq!(from_const(9).to_str_radix(10, 0, false), "9");
        assert_eq!(from_const(10).to_str_radix(10, 0, false), "10");
        assert_eq!(from_const(11).to_str_radix(10, 0, false), "11");
        assert_eq!(from_const(0).to_str_radix(10, 1, false), "0");
        assert_eq!(from_const(1).to_str_radix(10, 1, false), "1");
        assert_eq!(from_const(2).to_str_radix(10, 1, false), "2");
        assert_eq!(from_const(9).to_str_radix(10, 1, false), "9");
        assert_eq!(from_const(10).to_str_radix(10, 1, false), "10");
        assert_eq!(from_const(11).to_str_radix(10, 1, false), "11");
        assert_eq!(from_const(0).to_str_radix(10, 2, false), "00");
        assert_eq!(from_const(1).to_str_radix(10, 2, false), "01");
        assert_eq!(from_const(2).to_str_radix(10, 2, false), "02");
        assert_eq!(from_const(9).to_str_radix(10, 2, false), "09");
        assert_eq!(from_const(10).to_str_radix(10, 2, false), "10");
        assert_eq!(from_const(11).to_str_radix(10, 2, false), "11");
        assert_eq!(from_const(0).to_str_radix(10, 3, false), "000");
        assert_eq!(from_const(1).to_str_radix(10, 3, false), "001");
        assert_eq!(from_const(2).to_str_radix(10, 3, false), "002");
        assert_eq!(from_const(9).to_str_radix(10, 3, false), "009");
        assert_eq!(from_const(10).to_str_radix(10, 3, false), "010");
        assert_eq!(from_const(11).to_str_radix(10, 3, false), "011");
        assert_eq!(
            (Scalar::MAX - Scalar::ONE).to_str_radix(10, 0, false),
            "57896044618658097711785492504343953925963004582726670246466397748101839323135"
        );
        assert_eq!(
            Scalar::MAX.to_str_radix(10, 0, false),
            "57896044618658097711785492504343953925963004582726670246466397748101839323136"
        );
    }

    #[test]
    fn test_parse_hexadecimal_lower_case() {
        assert_eq!(Scalar::from_str_radix("0", 16).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("1", 16).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("2", 16).unwrap(), from_const(2));
        assert_eq!(Scalar::from_str_radix("9", 16).unwrap(), from_const(9));
        assert_eq!(Scalar::from_str_radix("a", 16).unwrap(), from_const(10));
        assert_eq!(Scalar::from_str_radix("e", 16).unwrap(), from_const(14));
        assert_eq!(Scalar::from_str_radix("f", 16).unwrap(), from_const(15));
        assert!(Scalar::from_str_radix("8", 8).is_err());
        assert_eq!(Scalar::from_str_radix("00", 16).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("01", 16).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("02", 16).unwrap(), from_const(2));
        assert_eq!(Scalar::from_str_radix("09", 16).unwrap(), from_const(9));
        assert_eq!(Scalar::from_str_radix("0a", 16).unwrap(), from_const(10));
        assert_eq!(Scalar::from_str_radix("0e", 16).unwrap(), from_const(14));
        assert_eq!(Scalar::from_str_radix("0f", 16).unwrap(), from_const(15));
        assert_eq!(Scalar::from_str_radix("10", 16).unwrap(), from_const(16));
        assert_eq!(Scalar::from_str_radix("11", 16).unwrap(), from_const(17));
        assert_eq!(Scalar::from_str_radix("12", 16).unwrap(), from_const(18));
        assert_eq!(Scalar::from_str_radix("19", 16).unwrap(), from_const(25));
        assert_eq!(Scalar::from_str_radix("1a", 16).unwrap(), from_const(26));
        assert_eq!(Scalar::from_str_radix("1e", 16).unwrap(), from_const(30));
        assert_eq!(Scalar::from_str_radix("1f", 16).unwrap(), from_const(31));
        assert_eq!(Scalar::from_str_radix("20", 16).unwrap(), from_const(32));
        assert_eq!(Scalar::from_str_radix("21", 16).unwrap(), from_const(33));
        assert_eq!(Scalar::from_str_radix("22", 16).unwrap(), from_const(34));
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
                "7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000",
                16
            )
            .unwrap(),
            Scalar::MAX
        );
        assert!(
            Scalar::from_str_radix(
                "7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000001",
                16
            )
            .is_err(),
        );
    }

    #[test]
    fn test_print_hexadecimal_lower_case() {
        assert_eq!(from_const(0).to_str_radix(16, 0, false), "0");
        assert_eq!(from_const(1).to_str_radix(16, 0, false), "1");
        assert_eq!(from_const(2).to_str_radix(16, 0, false), "2");
        assert_eq!(from_const(9).to_str_radix(16, 0, false), "9");
        assert_eq!(from_const(10).to_str_radix(16, 0, false), "a");
        assert_eq!(from_const(14).to_str_radix(16, 0, false), "e");
        assert_eq!(from_const(15).to_str_radix(16, 0, false), "f");
        assert_eq!(from_const(16).to_str_radix(16, 0, false), "10");
        assert_eq!(from_const(17).to_str_radix(16, 0, false), "11");
        assert_eq!(from_const(18).to_str_radix(16, 0, false), "12");
        assert_eq!(from_const(25).to_str_radix(16, 0, false), "19");
        assert_eq!(from_const(26).to_str_radix(16, 0, false), "1a");
        assert_eq!(from_const(30).to_str_radix(16, 0, false), "1e");
        assert_eq!(from_const(31).to_str_radix(16, 0, false), "1f");
        assert_eq!(from_const(0).to_str_radix(16, 1, false), "0");
        assert_eq!(from_const(1).to_str_radix(16, 1, false), "1");
        assert_eq!(from_const(2).to_str_radix(16, 1, false), "2");
        assert_eq!(from_const(9).to_str_radix(16, 1, false), "9");
        assert_eq!(from_const(10).to_str_radix(16, 1, false), "a");
        assert_eq!(from_const(14).to_str_radix(16, 1, false), "e");
        assert_eq!(from_const(15).to_str_radix(16, 1, false), "f");
        assert_eq!(from_const(16).to_str_radix(16, 1, false), "10");
        assert_eq!(from_const(17).to_str_radix(16, 1, false), "11");
        assert_eq!(from_const(18).to_str_radix(16, 1, false), "12");
        assert_eq!(from_const(25).to_str_radix(16, 1, false), "19");
        assert_eq!(from_const(26).to_str_radix(16, 1, false), "1a");
        assert_eq!(from_const(30).to_str_radix(16, 1, false), "1e");
        assert_eq!(from_const(31).to_str_radix(16, 1, false), "1f");
        assert_eq!(from_const(0).to_str_radix(16, 2, false), "00");
        assert_eq!(from_const(1).to_str_radix(16, 2, false), "01");
        assert_eq!(from_const(2).to_str_radix(16, 2, false), "02");
        assert_eq!(from_const(9).to_str_radix(16, 2, false), "09");
        assert_eq!(from_const(10).to_str_radix(16, 2, false), "0a");
        assert_eq!(from_const(14).to_str_radix(16, 2, false), "0e");
        assert_eq!(from_const(15).to_str_radix(16, 2, false), "0f");
        assert_eq!(from_const(16).to_str_radix(16, 2, false), "10");
        assert_eq!(from_const(17).to_str_radix(16, 2, false), "11");
        assert_eq!(from_const(18).to_str_radix(16, 2, false), "12");
        assert_eq!(from_const(25).to_str_radix(16, 2, false), "19");
        assert_eq!(from_const(26).to_str_radix(16, 2, false), "1a");
        assert_eq!(from_const(30).to_str_radix(16, 2, false), "1e");
        assert_eq!(from_const(31).to_str_radix(16, 2, false), "1f");
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
    fn test_parse_hexadecimal_upper_case() {
        assert_eq!(Scalar::from_str_radix("0", 16).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("1", 16).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("2", 16).unwrap(), from_const(2));
        assert_eq!(Scalar::from_str_radix("9", 16).unwrap(), from_const(9));
        assert_eq!(Scalar::from_str_radix("a", 16).unwrap(), from_const(10));
        assert_eq!(Scalar::from_str_radix("e", 16).unwrap(), from_const(14));
        assert_eq!(Scalar::from_str_radix("f", 16).unwrap(), from_const(15));
        assert!(Scalar::from_str_radix("8", 8).is_err());
        assert_eq!(Scalar::from_str_radix("00", 16).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("01", 16).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("02", 16).unwrap(), from_const(2));
        assert_eq!(Scalar::from_str_radix("09", 16).unwrap(), from_const(9));
        assert_eq!(Scalar::from_str_radix("0a", 16).unwrap(), from_const(10));
        assert_eq!(Scalar::from_str_radix("0e", 16).unwrap(), from_const(14));
        assert_eq!(Scalar::from_str_radix("0f", 16).unwrap(), from_const(15));
        assert_eq!(Scalar::from_str_radix("10", 16).unwrap(), from_const(16));
        assert_eq!(Scalar::from_str_radix("11", 16).unwrap(), from_const(17));
        assert_eq!(Scalar::from_str_radix("12", 16).unwrap(), from_const(18));
        assert_eq!(Scalar::from_str_radix("19", 16).unwrap(), from_const(25));
        assert_eq!(Scalar::from_str_radix("1a", 16).unwrap(), from_const(26));
        assert_eq!(Scalar::from_str_radix("1e", 16).unwrap(), from_const(30));
        assert_eq!(Scalar::from_str_radix("1f", 16).unwrap(), from_const(31));
        assert_eq!(Scalar::from_str_radix("20", 16).unwrap(), from_const(32));
        assert_eq!(Scalar::from_str_radix("21", 16).unwrap(), from_const(33));
        assert_eq!(Scalar::from_str_radix("22", 16).unwrap(), from_const(34));
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
                16
            )
            .is_err(),
        );
    }

    #[test]
    fn test_print_hexadecimal_upper_case() {
        assert_eq!(from_const(0).to_str_radix(16, 0, true), "0");
        assert_eq!(from_const(1).to_str_radix(16, 0, true), "1");
        assert_eq!(from_const(2).to_str_radix(16, 0, true), "2");
        assert_eq!(from_const(9).to_str_radix(16, 0, true), "9");
        assert_eq!(from_const(10).to_str_radix(16, 0, true), "A");
        assert_eq!(from_const(14).to_str_radix(16, 0, true), "E");
        assert_eq!(from_const(15).to_str_radix(16, 0, true), "F");
        assert_eq!(from_const(16).to_str_radix(16, 0, true), "10");
        assert_eq!(from_const(17).to_str_radix(16, 0, true), "11");
        assert_eq!(from_const(18).to_str_radix(16, 0, true), "12");
        assert_eq!(from_const(25).to_str_radix(16, 0, true), "19");
        assert_eq!(from_const(26).to_str_radix(16, 0, true), "1A");
        assert_eq!(from_const(30).to_str_radix(16, 0, true), "1E");
        assert_eq!(from_const(31).to_str_radix(16, 0, true), "1F");
        assert_eq!(from_const(0).to_str_radix(16, 1, true), "0");
        assert_eq!(from_const(1).to_str_radix(16, 1, true), "1");
        assert_eq!(from_const(2).to_str_radix(16, 1, true), "2");
        assert_eq!(from_const(9).to_str_radix(16, 1, true), "9");
        assert_eq!(from_const(10).to_str_radix(16, 1, true), "A");
        assert_eq!(from_const(14).to_str_radix(16, 1, true), "E");
        assert_eq!(from_const(15).to_str_radix(16, 1, true), "F");
        assert_eq!(from_const(16).to_str_radix(16, 1, true), "10");
        assert_eq!(from_const(17).to_str_radix(16, 1, true), "11");
        assert_eq!(from_const(18).to_str_radix(16, 1, true), "12");
        assert_eq!(from_const(25).to_str_radix(16, 1, true), "19");
        assert_eq!(from_const(26).to_str_radix(16, 1, true), "1A");
        assert_eq!(from_const(30).to_str_radix(16, 1, true), "1E");
        assert_eq!(from_const(31).to_str_radix(16, 1, true), "1F");
        assert_eq!(from_const(0).to_str_radix(16, 2, true), "00");
        assert_eq!(from_const(1).to_str_radix(16, 2, true), "01");
        assert_eq!(from_const(2).to_str_radix(16, 2, true), "02");
        assert_eq!(from_const(9).to_str_radix(16, 2, true), "09");
        assert_eq!(from_const(10).to_str_radix(16, 2, true), "0A");
        assert_eq!(from_const(14).to_str_radix(16, 2, true), "0E");
        assert_eq!(from_const(15).to_str_radix(16, 2, true), "0F");
        assert_eq!(from_const(16).to_str_radix(16, 2, true), "10");
        assert_eq!(from_const(17).to_str_radix(16, 2, true), "11");
        assert_eq!(from_const(18).to_str_radix(16, 2, true), "12");
        assert_eq!(from_const(25).to_str_radix(16, 2, true), "19");
        assert_eq!(from_const(26).to_str_radix(16, 2, true), "1A");
        assert_eq!(from_const(30).to_str_radix(16, 2, true), "1E");
        assert_eq!(from_const(31).to_str_radix(16, 2, true), "1F");
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
    fn test_try_to_u8() {
        assert_eq!(from_const(0).try_to_u8().unwrap(), 0);
        assert_eq!(from_const(1).try_to_u8().unwrap(), 1);
        assert_eq!(from_const(2).try_to_u8().unwrap(), 2);
        assert_eq!(
            from_const(u8::MAX as u64 - 1).try_to_u8().unwrap(),
            u8::MAX - 1
        );
        assert_eq!(from_const(u8::MAX as u64).try_to_u8().unwrap(), u8::MAX);
        assert!(from_const(u8::MAX as u64 + 1).try_to_u8().is_none());
        assert!(from_const(u8::MAX as u64 + 2).try_to_u8().is_none());
    }

    #[test]
    fn test_try_to_u16() {
        assert_eq!(from_const(0).try_to_u16().unwrap(), 0);
        assert_eq!(from_const(1).try_to_u16().unwrap(), 1);
        assert_eq!(from_const(2).try_to_u16().unwrap(), 2);
        assert_eq!(
            from_const(u16::MAX as u64 - 1).try_to_u16().unwrap(),
            u16::MAX - 1
        );
        assert_eq!(from_const(u16::MAX as u64).try_to_u16().unwrap(), u16::MAX);
        assert!(from_const(u16::MAX as u64 + 1).try_to_u16().is_none());
        assert!(from_const(u16::MAX as u64 + 2).try_to_u16().is_none());
    }

    #[test]
    fn test_to_le_bytes() {
        assert_eq!(
            parse_scalar("0x1caa16ab866063ef3c466732ba591aa9d6b3e7746611979e0219767cfa80fa45")
                .to_le_bytes(),
            [
                69, 250, 128, 250, 124, 118, 25, 2, 158, 151, 17, 102, 116, 231, 179, 214, 169, 26,
                89, 186, 50, 103, 70, 60, 239, 99, 96, 134, 171, 22, 170, 28
            ]
        );
        assert_eq!(
            parse_scalar("0x645752786f39a23dacbc0c9ff11eead2a96d50b51f4b9519be77e4640668292f")
                .to_le_bytes(),
            [
                47, 41, 104, 6, 100, 228, 119, 190, 25, 149, 75, 31, 181, 80, 109, 169, 210, 234,
                30, 241, 159, 12, 188, 172, 61, 162, 57, 111, 120, 82, 87, 100
            ]
        );
    }

    #[test]
    fn test_to_be_bytes() {
        assert_eq!(
            parse_scalar("0x376d20d4a3fbc47ab59ecfb4f465eef303180ff9b9ed675492bed81f081d3da9")
                .to_be_bytes(),
            [
                55, 109, 32, 212, 163, 251, 196, 122, 181, 158, 207, 180, 244, 101, 238, 243, 3,
                24, 15, 249, 185, 237, 103, 84, 146, 190, 216, 31, 8, 29, 61, 169
            ]
        );
        assert_eq!(
            parse_scalar("0x249a87c2d46034a2111064344be35f69e21900a68d30b2e54a3e4e7145adeefa")
                .to_be_bytes(),
            [
                36, 154, 135, 194, 212, 96, 52, 162, 17, 16, 100, 52, 75, 227, 95, 105, 226, 25, 0,
                166, 141, 48, 178, 229, 74, 62, 78, 113, 69, 173, 238, 250
            ]
        );
    }

    #[test]
    fn test_from_u512_mod_n() {
        assert_eq!(Scalar::from_u512_mod_n("0".parse().unwrap()), from_const(0));
        assert_eq!(Scalar::from_u512_mod_n("1".parse().unwrap()), from_const(1));
        assert_eq!(Scalar::from_u512_mod_n("2".parse().unwrap()), from_const(2));
        assert_eq!(
            Scalar::from_u512_mod_n(
                "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547bfffffffffffffff"
                    .parse()
                    .unwrap()
            ),
            Scalar::MAX - Scalar::ONE
        );
        assert_eq!(
            Scalar::from_u512_mod_n(
                "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
                    .parse()
                    .unwrap()
            ),
            Scalar::MAX
        );
        assert_eq!(
            Scalar::from_u512_mod_n(
                "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000001"
                    .parse()
                    .unwrap()
            ),
            from_const(0)
        );
        assert_eq!(
            Scalar::from_u512_mod_n(
                "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000002"
                    .parse()
                    .unwrap()
            ),
            from_const(1)
        );
    }

    #[test]
    fn test_try_to_u32() {
        assert_eq!(from_const(0).try_to_u32().unwrap(), 0);
        assert_eq!(from_const(1).try_to_u32().unwrap(), 1);
        assert_eq!(from_const(2).try_to_u32().unwrap(), 2);
        assert_eq!(
            from_const(u32::MAX as u64 - 1).try_to_u32().unwrap(),
            u32::MAX - 1
        );
        assert_eq!(from_const(u32::MAX as u64).try_to_u32().unwrap(), u32::MAX);
        assert!(bool::from(
            from_const(u32::MAX as u64 + 1).try_to_u32().is_none()
        ));
        assert!(bool::from(
            from_const(u32::MAX as u64 + 2).try_to_u32().is_none()
        ));
    }

    #[test]
    fn test_try_to_u64() {
        assert_eq!(from_const(0).try_to_u64().unwrap(), 0);
        assert_eq!(from_const(1).try_to_u64().unwrap(), 1);
        assert_eq!(from_const(2).try_to_u64().unwrap(), 2);
        assert_eq!(from_const(u64::MAX - 1).try_to_u64().unwrap(), u64::MAX - 1);
        assert_eq!(from_const(u64::MAX).try_to_u64().unwrap(), u64::MAX);
        assert_eq!(
            parse_scalar("0xffffffffffffffff").try_to_u64().unwrap(),
            u64::MAX
        );
        assert!(bool::from(
            parse_scalar("0x10000000000000000").try_to_u64().is_none()
        ));
        assert!(bool::from(
            parse_scalar("0x10000000000000001").try_to_u64().is_none()
        ));
    }

    #[test]
    fn test_try_to_u128() {
        assert_eq!(from_const(0).try_to_u128().unwrap(), 0);
        assert_eq!(from_const(1).try_to_u128().unwrap(), 1);
        assert_eq!(from_const(2).try_to_u128().unwrap(), 2);
        assert_eq!(
            parse_scalar("0xfffffffffffffffffffffffffffffffe")
                .try_to_u128()
                .unwrap(),
            u128::MAX - 1
        );
        assert_eq!(
            parse_scalar("0xffffffffffffffffffffffffffffffff")
                .try_to_u128()
                .unwrap(),
            u128::MAX
        );
        assert!(bool::from(
            parse_scalar("0x100000000000000000000000000000000")
                .try_to_u128()
                .is_none()
        ));
        assert!(bool::from(
            parse_scalar("0x100000000000000000000000000000001")
                .try_to_u128()
                .is_none()
        ));
    }

    #[test]
    fn test_to_u256() {
        assert_eq!(from_const(0).to_u256(), "0".parse().unwrap());
        assert_eq!(from_const(1).to_u256(), "1".parse().unwrap());
        assert_eq!(from_const(2).to_u256(), "2".parse().unwrap());
        assert_eq!(
            (Scalar::MAX - Scalar::ONE).to_u256(),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547bfffffffffffffff"
                .parse()
                .unwrap()
        );
        assert_eq!(
            Scalar::MAX.to_u256(),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
                .parse()
                .unwrap()
        );
    }

    #[test]
    fn test_to_u512() {
        assert_eq!(from_const(0).to_u512(), "0".parse().unwrap());
        assert_eq!(from_const(1).to_u512(), "1".parse().unwrap());
        assert_eq!(from_const(2).to_u512(), "2".parse().unwrap());
        assert_eq!(
            (Scalar::MAX - Scalar::ONE).to_u512(),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547bfffffffffffffff"
                .parse()
                .unwrap()
        );
        assert_eq!(
            Scalar::MAX.to_u512(),
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000"
                .parse()
                .unwrap()
        );
    }

    #[test]
    fn test_multiplicative_generator() {
        assert_eq!(Scalar::MULTIPLICATIVE_GENERATOR, from_const(5));
        assert_eq!(
            Scalar::MULTIPLICATIVE_GENERATOR.pow(Scalar::MAX / from_const(1u64 << Scalar::S)),
            Scalar::ROOT_OF_UNITY
        );
    }

    #[test]
    fn test_minus_two() {
        assert_eq!(Scalar::MINUS_TWO, -from_const(2));
        assert_eq!(
            from_const(42).invert_unwrap(),
            from_const(42).pow(Scalar::MINUS_TWO)
        );
    }

    #[test]
    fn test_two_inv() {
        assert_eq!(Scalar::TWO_INV, from_const(2).invert_unwrap());
        assert_eq!(Scalar::TWO_INV.invert_unwrap(), from_const(2));
    }

    #[test]
    fn test_root_of_unity() {
        for i in 0..Scalar::S {
            assert_ne!(
                Scalar::ROOT_OF_UNITY.pow(from_const(1u64 << i)),
                Scalar::ONE
            );
        }
        assert_eq!(
            Scalar::ROOT_OF_UNITY.pow(from_const(1u64 << Scalar::S)),
            Scalar::ONE
        );
    }

    #[test]
    fn test_root_of_unity_inverse() {
        assert_eq!(
            Scalar::ROOT_OF_UNITY_INV,
            Scalar::ROOT_OF_UNITY.invert_unwrap()
        );
    }

    #[test]
    fn test_delta() {
        assert_eq!(
            Scalar::DELTA,
            Scalar::MULTIPLICATIVE_GENERATOR.pow(from_const(1u64 << Scalar::S))
        );
    }

    #[test]
    fn test_three_inv() {
        assert_eq!(Scalar::THREE_INV, from_const(3).invert_unwrap());
        assert_eq!(Scalar::THREE_INV.invert_unwrap(), from_const(3));
    }

    #[test]
    fn test_three_adic_root_of_unity() {
        for i in 0..Scalar::T {
            assert_ne!(
                Scalar::THREE_ADIC_ROOT_OF_UNITY.pow(from_const(1u64 << i)),
                Scalar::ONE
            );
        }
        assert_eq!(
            Scalar::THREE_ADIC_ROOT_OF_UNITY.pow(from_const(3).pow_small(Scalar::T as usize)),
            Scalar::ONE
        );
    }

    #[test]
    fn test_three_adic_root_of_unity_inverse() {
        assert_eq!(
            Scalar::THREE_ADIC_ROOT_OF_UNITY_INV,
            Scalar::THREE_ADIC_ROOT_OF_UNITY.invert_unwrap()
        );
    }
}
