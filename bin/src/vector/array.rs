use lib::core::{Vector, marker::MeaningfulOutput};
use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    iter::Sum,
    mem::{self, MaybeUninit},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Clone, Copy)]
pub struct ArrayVector<T, const N: usize>([T; N]);

impl<T, const N: usize> Add<Self> for ArrayVector<T, N>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut uninit = [const { MaybeUninit::uninit() }; N];
        for ((elem_uninit, elem_self), elem_rhs) in uninit.iter_mut().zip(self.0.into_iter()).zip(rhs.0.into_iter()) {
            elem_uninit.write(elem_self + elem_rhs);
        }
        // SAFETY: - Initialized the contents above.
        //         - `Src` and `Dst` have the same layout.
        Self(unsafe { mem::transmute_copy(&uninit) })
    }
}

impl<T, const N: usize> AddAssign<Self> for ArrayVector<T, N>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        for (elem_self, elem_rhs) in self.0.iter_mut().zip(rhs.0.into_iter()) {
            *elem_self += elem_rhs;
        }
    }
}

impl<T, const N: usize> Sub<Self> for ArrayVector<T, N>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut uninit = [const { MaybeUninit::uninit() }; N];
        for ((elem_uninit, elem_self), elem_rhs) in uninit.iter_mut().zip(self.0.into_iter()).zip(rhs.0.into_iter()) {
            elem_uninit.write(elem_self - elem_rhs);
        }
        // SAFETY: - Initialized the contents above.
        //         - `Src` and `Dst` have the same layout.
        Self(unsafe { mem::transmute_copy(&uninit) })
    }
}

impl<T, const N: usize> SubAssign<Self> for ArrayVector<T, N>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        for (elem_self, elem_rhs) in self.0.iter_mut().zip(rhs.0.into_iter()) {
            *elem_self -= elem_rhs;
        }
    }
}

impl<T, const N: usize> Mul<T> for ArrayVector<T, N>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut uninit = [const { MaybeUninit::uninit() }; N];
        for (elem_uninit, elem_self) in uninit.iter_mut().zip(self.0.into_iter()) {
            elem_uninit.write(elem_self * rhs);
        }
        // SAFETY: - Initialized the contents above.
        //         - `Src` and `Dst` have the same layout.
        Self(unsafe { mem::transmute_copy(&uninit) })
    }
}

impl<T, const N: usize> MulAssign<T> for ArrayVector<T, N>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        for elem in self.0.iter_mut() {
            *elem *= rhs
        }
    }
}

impl<T, const N: usize> Div<T> for ArrayVector<T, N>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut uninit = [const { MaybeUninit::uninit() }; N];
        for (elem_uninit, elem_self) in uninit.iter_mut().zip(self.0.into_iter()) {
            elem_uninit.write(elem_self / rhs);
        }
        // SAFETY: - Initialized the contents above.
        //         - `Src` and `Dst` have the same layout.
        Self(unsafe { mem::transmute_copy(&uninit) })
    }
}

impl<T, const N: usize> DivAssign<T> for ArrayVector<T, N>
where
    T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        for elem in self.0.iter_mut() {
            *elem /= rhs
        }
    }
}

impl<T, const N: usize> Neg for ArrayVector<T, N>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut uninit = [const { MaybeUninit::uninit() }; N];
        for (elem_uninit, elem_self) in uninit.iter_mut().zip(self.0.into_iter()) {
            elem_uninit.write(-elem_self);
        }
        // SAFETY: - Initialized the contents above.
        //         - `Src` and `Dst` have the same layout.
        Self(unsafe { mem::transmute_copy(&uninit) })
    }
}

impl<T, const N: usize> From<[T; N]> for ArrayVector<T, N> {
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T: Display, const N: usize> Display for ArrayVector<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "(")?;
        let mut iterator = self.0.iter();
        let last = iterator.next_back();
        for element in iterator {
            write!(f, "{}, ", *element)?;
        }
        if let Some(element) = last {
            write!(f, "{}", *element)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl<T, const N: usize> MeaningfulOutput for ArrayVector<T, N> {}

impl<T, const N: usize> Vector<N> for ArrayVector<T, N>
where
    T: Copy
        + Add<Output = T>
        + AddAssign
        + Sub<Output = T>
        + SubAssign
        + Mul<Output = T>
        + MulAssign
        + Div<Output = T>
        + DivAssign
        + Neg<Output = T>
        + Sum,
{
    type Element = T;

    fn magnitude_squared(self) -> Self::Element {
        self.0.into_iter().map(|elem| elem * elem).sum()
    }

    fn dot(self, rhs: Self) -> Self::Element {
        self.0.into_iter().zip(rhs.0).map(|(lhs, rhs)| lhs * rhs).sum()
    }

    fn as_array(&self) -> &[Self::Element; N] {
        &self.0
    }

    fn as_mut_array(&mut self) -> &mut [Self::Element; N] {
        &mut self.0
    }
}
