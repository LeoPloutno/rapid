//! Types and traits meant to distinguish between different types
//! of ensemble statistics.

use std::ops::{Deref, DerefMut};

/// An enum differentiating between distinguishable and bosonic statistics.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Stat<D, B> {
    /// Distinguishable statistics.
    Distinguishable(D),
    /// Bosonic statistics.
    Bosonic(B),
}

impl<D, B> Stat<D, B> {
    /// Converts from `Stat<D, B>` to
    /// `Stat<&D::Target, &B::Target>`.
    ///
    /// Leaves the original `Stat` in-place,
    /// creating a new one containing references to the inner types' `Deref::Target` types.
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

    /// Converts from `Stat<D, B>` to
    /// `Stat<&mut D::Target, &mut B::Target>`.
    ///
    /// Leaves the original `Stat` in-place,
    /// creating a new one containing mutable references to the inner types' `Deref::Target` types.
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

/// A trait for marking exchange potentials of distinguishable particles.
pub trait Distinguishable {}

/// A trait for marking exchange potentials of bosons.
pub trait Bosonic {}
