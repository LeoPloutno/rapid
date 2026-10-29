use lib::{
    core::Vector,
    potential::physical::{AdditivePhysicalPotential, AtomAdditivePhysicalPotential},
};
use num::Float;
use std::{convert::Infallible, error::Error};

mod error {
    use lib::core::error::{AccessError, EmptyError, InvalidIndexError, InvalidRangeError};
    use std::{convert::Infallible, error::Error, fmt::Display};

    use crate::sync_ops::mutex::SyncError;

    #[derive(Clone, Copy, Debug)]
    pub enum HarmonicPotentialError<T> {
        Access(AccessError),
        Add(T),
    }

    impl<T> From<Infallible> for HarmonicPotentialError<T> {
        fn from(value: Infallible) -> Self {
            value
        }
    }

    impl<T> From<EmptyError> for HarmonicPotentialError<T> {
        fn from(value: EmptyError) -> Self {
            Self::Access(AccessError::Empty(value))
        }
    }

    impl<T> From<InvalidIndexError> for HarmonicPotentialError<T> {
        fn from(value: InvalidIndexError) -> Self {
            Self::Access(AccessError::Index(value))
        }
    }

    impl<T> From<InvalidRangeError> for HarmonicPotentialError<T> {
        fn from(value: InvalidRangeError) -> Self {
            Self::Access(AccessError::Range(value))
        }
    }

    impl From<SyncError> for HarmonicPotentialError<SyncError> {
        fn from(value: SyncError) -> Self {
            Self::Add(value)
        }
    }

    impl<T: Display> Display for HarmonicPotentialError<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Access(err) => write!(f, "access error: {}", err),
                Self::Add(err) => write!(f, "synchronized addition error: {}", err),
            }
        }
    }

    impl<T: Error> Error for HarmonicPotentialError<T> {}
}
pub use error::HarmonicPotentialError;

pub struct Harmonic<T, const N: usize> {
    potential_prefactor: T,
}

impl<T, const N: usize> Harmonic<T, N>
where
    T: From<f32> + Float,
{
    pub fn new<A>(images: usize, spring_constant: T, adder: A) -> AdditivePhysicalPotential<Self, A> {
        assert!(spring_constant >= T::zero(), "spring constant must be non-negative");

        AdditivePhysicalPotential::new(
            adder,
            Self {
                potential_prefactor: spring_constant / (images as f32).into(),
            },
        )
    }
}

impl<T, V, E, const N: usize> AtomAdditivePhysicalPotential<T, V, E> for Harmonic<T, N>
where
    T: From<f32> + Float,
    V: Vector<N, Element = T>,
    E: Error,
    HarmonicPotentialError<E>: From<E>,
{
    type AtomError = Infallible;
    type SystemError = HarmonicPotentialError<E>;

    #[inline]
    fn calculate_energy_and_force(&mut self, atom_index: usize, position: &V) -> Result<(T, V), Self::AtomError> {
        #[allow(deprecated)]
        Ok((
            <Self as AtomAdditivePhysicalPotential<T, V, E>>::calculate_energy(self, atom_index, position)?,
            <Self as AtomAdditivePhysicalPotential<T, V, E>>::calculate_force(self, atom_index, position)?,
        ))
    }

    #[inline]
    fn calculate_energy(&mut self, _atom_index: usize, position: &V) -> Result<T, Self::AtomError> {
        Ok(self.potential_prefactor * position.magnitude_squared())
    }

    #[inline]
    fn calculate_force(&mut self, _atom_index: usize, position: &V) -> Result<V, Self::AtomError> {
        Ok(-*position * self.potential_prefactor * 2.0.into())
    }
}
