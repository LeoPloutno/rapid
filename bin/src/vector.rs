mod simd_vector {
    use lib::core::Vector;
    use std::{
        iter::Sum,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
        simd::{Simd, SimdElement},
    };

    pub struct SimdVector<const N: usize, T: SimdElement>(Simd<T, N>);

    impl<const N: usize, T: SimdElement> From<[T; N]> for SimdVector<N, T> {
        fn from(value: [T; N]) -> Self {
            Self(value.into())
        }
    }

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

    impl<const N: usize, T> Vector<N> for SimdVector<N, T>
    where
        T: SimdElement
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sum,
        Simd<T, N>: Add<Output = Simd<T, N>>
            + Sub<Output = Simd<T, N>>
            + Mul<Output = Simd<T, N>>
            + Div<Output = Simd<T, N>>
            + Neg<Output = Simd<T, N>>,
    {
        type Element = T;

        fn as_array(&self) -> &[Self::Element; N] {
            self.0.as_array()
        }

        fn as_mut_array(&mut self) -> &mut [Self::Element; N] {
            self.0.as_mut_array()
        }

        fn magnitude_squared(self) -> Self::Element {
            (self.0 * self.0).to_array().into_iter().sum()
        }

        fn dot(self, rhs: Self) -> Self::Element {
            (self.0 * rhs.0).to_array().into_iter().sum()
        }
    }
}

mod array_vector {
    use lib::core::Vector;
    use std::{
        iter::Sum,
        mem::{self, MaybeUninit},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    };

    pub struct ArrayVector<const N: usize, T>([T; N]);

    impl<const N: usize, T> From<[T; N]> for ArrayVector<N, T> {
        fn from(value: [T; N]) -> Self {
            Self(value)
        }
    }

    impl<const N: usize, T> Add<Self> for ArrayVector<N, T>
    where
        T: Add<Output = T>,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            let mut uninit = [const { MaybeUninit::uninit() }; N];
            for ((elem_uninit, elem_self), elem_rhs) in uninit
                .iter_mut()
                .zip(self.0.into_iter())
                .zip(rhs.0.into_iter())
            {
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
            for ((elem_uninit, elem_self), elem_rhs) in uninit
                .iter_mut()
                .zip(self.0.into_iter())
                .zip(rhs.0.into_iter())
            {
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

    impl<const N: usize, T> Vector<N> for ArrayVector<N, T>
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

        fn as_array(&self) -> &[Self::Element; N] {
            &self.0
        }

        fn as_mut_array(&mut self) -> &mut [Self::Element; N] {
            &mut self.0
        }

        fn magnitude_squared(self) -> Self::Element {
            self.0.into_iter().map(|elem| elem.clone() * elem).sum()
        }

        fn dot(self, rhs: Self) -> Self::Element {
            self.0
                .into_iter()
                .zip(rhs.0)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum()
        }
    }
}

pub use array_vector::ArrayVector;
pub use simd_vector::SimdVector;
