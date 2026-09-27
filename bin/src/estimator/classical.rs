use std::ops::{Add, Mul};

use lib::core::marker::MeaningfulOutput;

trait Foo<T, U, V> {}

struct Bar<const N: usize, T, U>(std::marker::PhantomData<(T, U)>);

impl<const N: usize, T, U, V> Foo<T, U, ()> for Bar<N, T, V> {}

impl<const N: usize, T, U, V> Foo<T, U, T> for Bar<N, T, V> where
    T: Add<Output = T> + Mul<Output = T> + From<f32> + MeaningfulOutput
{
}
