use lib::core::Vector;
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    simd::{Simd, SimdElement},
};

#[derive(Clone, Copy)]
pub struct SimdVector<T: SimdElement, const N: usize>(Simd<T, N>);

impl<T, const N: usize> Add<Self> for SimdVector<T, N>
where
    T: SimdElement + Add<Output = T>,
    Simd<T, N>: Add<Output = Simd<T, N>>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<T, const N: usize> AddAssign<Self> for SimdVector<T, N>
where
    T: SimdElement,
    Simd<T, N>: Add<Output = Simd<T, N>>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<T, const N: usize> Sub<Self> for SimdVector<T, N>
where
    T: SimdElement + Sub<Output = T>,
    Simd<T, N>: Sub<Output = Simd<T, N>>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<T, const N: usize> SubAssign<Self> for SimdVector<T, N>
where
    T: SimdElement,
    Simd<T, N>: Sub<Output = Simd<T, N>>,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<T, const N: usize> Mul<T> for SimdVector<T, N>
where
    T: SimdElement,
    Simd<T, N>: Mul<Output = Simd<T, N>>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0 * Simd::splat(rhs))
    }
}

impl<T, const N: usize> MulAssign<T> for SimdVector<T, N>
where
    T: SimdElement,
    Simd<T, N>: Mul<Output = Simd<T, N>>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.0 *= Simd::splat(rhs);
    }
}

impl<T, const N: usize> Div<T> for SimdVector<T, N>
where
    T: SimdElement,
    Simd<T, N>: Div<Output = Simd<T, N>>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self(self.0 / Simd::splat(rhs))
    }
}

impl<T, const N: usize> DivAssign<T> for SimdVector<T, N>
where
    T: SimdElement,
    Simd<T, N>: Div<Output = Simd<T, N>>,
{
    fn div_assign(&mut self, rhs: T) {
        self.0 /= Simd::splat(rhs);
    }
}

impl<T, const N: usize> Neg for SimdVector<T, N>
where
    T: SimdElement,
    Simd<T, N>: Neg<Output = Simd<T, N>>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<T: SimdElement, const N: usize> From<[T; N]> for SimdVector<T, N> {
    fn from(value: [T; N]) -> Self {
        Self(Simd::from_array(value))
    }
}

impl<T, const N: usize> Vector<N> for SimdVector<T, N>
where
    T: SimdElement + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Sum,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Output = Simd<T, N>>
        + Div<Output = Simd<T, N>>
        + Neg<Output = Simd<T, N>>,
{
    type Element = T;

    fn magnitude_squared(self) -> Self::Element {
        (self.0 * self.0).to_array().into_iter().sum()
    }

    fn dot(self, rhs: Self) -> Self::Element {
        (self.0 * rhs.0).to_array().into_iter().sum()
    }

    fn as_array(&self) -> &[Self::Element; N] {
        self.0.as_array()
    }

    fn as_mut_array(&mut self) -> &mut [Self::Element; N] {
        self.0.as_mut_array()
    }
}
