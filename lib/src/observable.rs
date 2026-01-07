use crate::vector::Vector;

pub mod debug;
pub mod quantum;

pub enum ScalarOrVector<const N: usize, T, V>
where
    V: Vector<N, Element = T>,
{
    Scalar(T),
    Vector(V),
}
