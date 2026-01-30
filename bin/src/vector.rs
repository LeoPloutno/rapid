mod simd_vector {
    use lib::vector::Vector;
    use std::{
        iter::Sum,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
        simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
    };

    pub struct SimdVector<T, const N: usize>(Simd<T, N>)
    where
        T: SimdElement,
        LaneCount<N>: SupportedLaneCount;

    impl<T, const N: usize> Add<Self> for SimdVector<T, N>
    where
        T: SimdElement + Add<Output = T>,
        LaneCount<N>: SupportedLaneCount,
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
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: Add<Output = Simd<T, N>>,
    {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
        }
    }

    impl<T, const N: usize> Sub<Self> for SimdVector<T, N>
    where
        T: SimdElement + Sub<Output = T>,
        LaneCount<N>: SupportedLaneCount,
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
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: Sub<Output = Simd<T, N>>,
    {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 -= rhs.0;
        }
    }

    impl<T, const N: usize> Mul<T> for SimdVector<T, N>
    where
        T: SimdElement,
        LaneCount<N>: SupportedLaneCount,
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
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: Mul<Output = Simd<T, N>>,
    {
        fn mul_assign(&mut self, rhs: T) {
            self.0 *= Simd::splat(rhs);
        }
    }

    impl<T, const N: usize> Div<T> for SimdVector<T, N>
    where
        T: SimdElement,
        LaneCount<N>: SupportedLaneCount,
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
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: Div<Output = Simd<T, N>>,
    {
        fn div_assign(&mut self, rhs: T) {
            self.0 /= Simd::splat(rhs);
        }
    }

    impl<T, const N: usize> Neg for SimdVector<T, N>
    where
        T: SimdElement,
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: Neg<Output = Simd<T, N>>,
    {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl<T, const N: usize> Vector<N> for SimdVector<T, N>
    where
        T: SimdElement + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Sum,
        LaneCount<N>: SupportedLaneCount,
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

        fn magnitude_squared(&self) -> Self::Element {
            (self.0 * self.0).to_array().into_iter().sum()
        }
    }
}

mod array_vector {
    use lib::vector::Vector;
    use std::{
        iter::Sum,
        mem::{self, MaybeUninit},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    };

    pub struct ArrayVector<T, const N: usize>([T; N]);

    impl<T, const N: usize> Add<Self> for ArrayVector<T, N>
    where
        T: Add<Output = T>,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            let mut uninit = [const { MaybeUninit::uninit() }; N];
            for ((elem_uninit, elem_self), elem_rhs) in uninit.iter_mut().zip(self.0.into_iter()).zip(rhs.0.into_iter())
            {
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
            for ((elem_uninit, elem_self), elem_rhs) in uninit.iter_mut().zip(self.0.into_iter()).zip(rhs.0.into_iter())
            {
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

    impl<T, const N: usize> MulAssign<T> for ArrayVector<T, N>
    where
        T: Clone + MulAssign,
    {
        fn mul_assign(&mut self, rhs: T) {
            for elem in self.0.iter_mut() {
                *elem *= rhs.clone()
            }
        }
    }

    impl<T, const N: usize> Div<T> for ArrayVector<T, N>
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

    impl<T, const N: usize> DivAssign<T> for ArrayVector<T, N>
    where
        T: Clone + DivAssign,
    {
        fn div_assign(&mut self, rhs: T) {
            for elem in self.0.iter_mut() {
                *elem /= rhs.clone()
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

    impl<T, const N: usize> Vector<N> for ArrayVector<T, N>
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

        fn magnitude_squared(&self) -> Self::Element {
            self.0.iter().map(|elem| elem.clone() * elem.clone()).sum()
        }
    }
}
