use anyhow::Context;
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
        Self::try_from_le_bytes(&value.to_little_endian()).context("overflow")
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
        match value.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => value.subp(),
            _ => value,
        }
    }

    fn invert(&self) -> Option<Self> {
        if self.is_zero().into() {
            None
        } else {
            Some(self.pow(Scalar::MINUS_TWO))
        }
    }

    fn invert_const_time(&self) -> CtOption<Self> {
        CtOption::new(self.pow_const_time(Scalar::MINUS_TWO), !self.is_zero())
    }

    fn pow(mut self, exp: Self) -> Self {
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

    fn pow_const_time(mut self, exp: Self) -> Self {
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

    fn div_int(&self, rhs: &Self) -> Option<(Self, Self)> {
        if rhs.is_zero().into() {
            return None;
        }
        let lhs = self.to_u256();
        let rhs = rhs.to_u256();
        let (quotient, remainder) = lhs.div_mod(rhs);
        Some((quotient.try_into().unwrap(), remainder.try_into().unwrap()))
    }

    fn try_from_le_bytes(bytes: &[u8]) -> Option<Self> {
        assert!(bytes.len() == 32);
        let raw = Self(
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        );
        match raw.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => None,
            _ => Some(Self::mont_mul(&raw, &Self::R2)),
        }
    }

    fn try_from_be_bytes(bytes: &[u8]) -> Option<Self> {
        assert!(bytes.len() == 32);
        let raw = Self(
            u64::from_be_bytes(bytes[24..32].try_into().unwrap()),
            u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_be_bytes(bytes[0..8].try_into().unwrap()),
        );
        match raw.cmp_raw(&Self::MAX_RAW) {
            Ordering::Greater => None,
            _ => Some(Self::mont_mul(&raw, &Self::R2)),
        }
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

    fn try_to_u32(&self) -> Option<u32> {
        let raw = self.to_raw();
        if (raw.1, raw.2, raw.3) != (0, 0, 0) {
            return None;
        }
        if raw.0 > u32::MAX as u64 {
            return None;
        }
        Some(raw.0 as u32)
    }

    fn try_to_u64(&self) -> Option<u64> {
        let raw = self.to_raw();
        if (raw.1, raw.2, raw.3) != (0, 0, 0) {
            return None;
        }
        Some(raw.0)
    }

    fn try_to_u128(&self) -> Option<u128> {
        let raw = self.to_raw();
        if (raw.2, raw.3) != (0, 0) {
            return None;
        }
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&raw.0.to_le_bytes());
        bytes[8..16].copy_from_slice(&raw.1.to_le_bytes());
        Some(u128::from_le_bytes(bytes))
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

    const S: usize = 32;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn from_const(value: u64) -> Scalar {
        Scalar::from_const(value)
    }

    fn parse_scalar(s: &'static str) -> Scalar {
        s.parse().unwrap()
    }

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

    // TODO
}
