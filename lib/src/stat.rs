use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Stat<D, B> {
    Distinguishable(D),
    Bosonic(B),
}

impl<D, B> Stat<D, B> {
    pub fn as_deref(&self) -> Stat<&<D as Deref>::Target, &<B as Deref>::Target>
    where
        D: Deref,
        B: Deref,
    {
        match self {
            Self::Distinguishable(dist) => Stat::Distinguishable(dist),
            Self::Bosonic(boson) => Stat::Bosonic(boson),
        }
    }

    pub fn as_deref_mut(&mut self) -> Stat<&mut <D as Deref>::Target, &mut <B as Deref>::Target>
    where
        D: DerefMut,
        B: DerefMut,
    {
        match self {
            Self::Distinguishable(dist) => Stat::Distinguishable(dist),
            Self::Bosonic(boson) => Stat::Bosonic(boson),
        }
    }
}

pub trait Distinguishable {}

pub trait Bosonic {}
