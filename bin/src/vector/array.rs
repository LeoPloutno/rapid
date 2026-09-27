use lib::core::Vector;
use std::{
    iter::Sum,
    mem::{self, MaybeUninit},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    slice::{Iter, IterMut},
};

pub struct ArrayVector<const N: usize, T>([T; N]);

impl<const N: usize, T> Add<Self> for ArrayVector<N, T>
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

impl<const N: usize, T> AddAssign<Self> for ArrayVector<N, T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        for (elem_self, elem_rhs) in self.0.iter_mut().zip(rhs.0.into_iter()) {
            *elem_self += elem_rhs;
        }
    }
}

impl<const N: usize, T> Sub<Self> for ArrayVector<N, T>
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

impl<const N: usize, T> SubAssign<Self> for ArrayVector<N, T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        for (elem_self, elem_rhs) in self.0.iter_mut().zip(rhs.0.into_iter()) {
            *elem_self -= elem_rhs;
        }
    }
}

impl<const N: usize, T> Mul<T> for ArrayVector<N, T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut uninit = [const { MaybeUninit::uninit() }; N];
        for (elem_uninit, elem_self) in uninit.iter_mut().zip(self.0.into_iter()) {
            elem_uninit.write(elem_self * rhs.clone());
        }
        // SAFETY: - Initialized the contents above.
        //         - `Src` and `Dst` have the same layout.
        Self(unsafe { mem::transmute_copy(&uninit) })
    }
}

impl<const N: usize, T> MulAssign<T> for ArrayVector<N, T>
where
    T: Clone + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        for elem in self.0.iter_mut() {
            *elem *= rhs.clone()
        }
    }
}

impl<const N: usize, T> Div<T> for ArrayVector<N, T>
where
    T: Clone + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut uninit = [const { MaybeUninit::uninit() }; N];
        for (elem_uninit, elem_self) in uninit.iter_mut().zip(self.0.into_iter()) {
            elem_uninit.write(elem_self / rhs.clone());
        }
        // SAFETY: - Initialized the contents above.
        //         - `Src` and `Dst` have the same layout.
        Self(unsafe { mem::transmute_copy(&uninit) })
    }
}

impl<const N: usize, T> DivAssign<T> for ArrayVector<N, T>
where
    T: Clone + DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        for elem in self.0.iter_mut() {
            *elem /= rhs.clone()
        }
    }
}

impl<const N: usize, T> Neg for ArrayVector<N, T>
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

impl<'a, const N: usize, T> IntoIterator for &'a ArrayVector<N, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, const N: usize, T> IntoIterator for &'a mut ArrayVector<N, T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<const N: usize, T> Vector for ArrayVector<N, T>
where
    T: Clone
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
    const DIM: usize = N;

    fn magnitude_squared(self) -> Self::Element {
        self.0.into_iter().map(|elem| elem.clone() * elem).sum()
    }

    fn dot(self, rhs: Self) -> Self::Element {
        self.0.into_iter().zip(rhs.0).map(|(lhs, rhs)| lhs * rhs).sum()
    }
}
