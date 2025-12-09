#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Stat<D, B> {
    Distinguishable(D),
    Bosonic(B),
}

pub trait Distinguishable {}

pub trait Bosonic {}
