use lib::core::Vector;
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    simd::{Simd, SimdElement},
    slice::{Iter, IterMut},
};

pub struct SimdVector<const N: usize, T: SimdElement>(Simd<T, N>);

impl<const N: usize, T> Add<Self> for SimdVector<N, T>
where
    T: SimdElement + Add<Output = T>,
    Simd<T, N>: Add<Output = Simd<T, N>>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<const N: usize, T> AddAssign<Self> for SimdVector<N, T>
where
    T: SimdElement,
    Simd<T, N>: Add<Output = Simd<T, N>>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<const N: usize, T> Sub<Self> for SimdVector<N, T>
where
    T: SimdElement + Sub<Output = T>,
    Simd<T, N>: Sub<Output = Simd<T, N>>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<const N: usize, T> SubAssign<Self> for SimdVector<N, T>
where
    T: SimdElement,
    Simd<T, N>: Sub<Output = Simd<T, N>>,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<const N: usize, T> Mul<T> for SimdVector<N, T>
where
    T: SimdElement,
    Simd<T, N>: Mul<Output = Simd<T, N>>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0 * Simd::splat(rhs))
    }
}

impl<const N: usize, T> MulAssign<T> for SimdVector<N, T>
where
    T: SimdElement,
    Simd<T, N>: Mul<Output = Simd<T, N>>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.0 *= Simd::splat(rhs);
    }
}

impl<const N: usize, T> Div<T> for SimdVector<N, T>
where
    T: SimdElement,
    Simd<T, N>: Div<Output = Simd<T, N>>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self(self.0 / Simd::splat(rhs))
    }
}

impl<const N: usize, T> DivAssign<T> for SimdVector<N, T>
where
    T: SimdElement,
    Simd<T, N>: Div<Output = Simd<T, N>>,
{
    fn div_assign(&mut self, rhs: T) {
        self.0 /= Simd::splat(rhs);
    }
}

impl<const N: usize, T> Neg for SimdVector<N, T>
where
    T: SimdElement,
    Simd<T, N>: Neg<Output = Simd<T, N>>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<'a, const N: usize, T: SimdElement> IntoIterator for &'a SimdVector<N, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.as_array().iter()
    }
}

impl<'a, const N: usize, T: SimdElement> IntoIterator for &'a mut SimdVector<N, T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.as_mut_array().iter_mut()
    }
}

impl<const N: usize, T> Vector for SimdVector<N, T>
where
    T: SimdElement + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Sum,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Output = Simd<T, N>>
        + Div<Output = Simd<T, N>>
        + Neg<Output = Simd<T, N>>,
{
    const DIM: usize = N;
    type Element = T;

    fn magnitude_squared(self) -> Self::Element {
        (self.0 * self.0).to_array().into_iter().sum()
    }

    fn dot(self, rhs: Self) -> Self::Element {
        (self.0 * rhs.0).to_array().into_iter().sum()
    }
}
