use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Vector<const N: usize>:
    Sized
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Self::Element, Output = Self>
    + MulAssign<Self::Element>
    + Div<Self::Element, Output = Self>
    + DivAssign<Self::Element>
    + Neg<Output = Self>
{
    type Element;

    fn as_array(&self) -> &[Self::Element; N];

    fn as_mut_array(&mut self) -> &mut [Self::Element; N];

    fn magnitude_squared(&self) -> Self::Element;
}
