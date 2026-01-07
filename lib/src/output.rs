use std::io::Write;

use crate::{core::AtomGroupInfo, observable::ScalarOrVector, vector::Vector};

/// A trait for streams that write to coordinate files, such as '.xyz' files.
pub trait VectorsOutput<const N: usize, T, V>
where
    V: Vector<N, Element = T>,
{
    type Error;

    /// Write the coordinates of the atoms in all groups to the stream.
    #[must_use]
    fn write(
        &mut self,
        step: usize,
        groups: &[AtomGroupInfo<T>],
        vectors: &[V],
    ) -> Result<(), Self::Error>;
}

pub trait ObservablesOutput<const N: usize, T, V, E>
where
    V: Vector<N, Element = T>,
{
    type Input;
    type Error: From<E>;

    #[must_use]
    fn write(
        &mut self,
        step: usize,
        groups: &[AtomGroupInfo<T>],
        observables: &mut dyn Iterator<Item = Result<Self::Input, E>>,
    ) -> Result<(), Self::Error>;
}

pub struct Observables<O, S> {
    pub observables: O,
    pub stream: S,
}

/// An enum which contains the output stream for the two kinds of observables.
pub enum ObservablesOption<Q, D, S> {
    None,
    Quantum(Observables<Q, S>),
    Debug(Observables<D, S>),
    Shared {
        quantum_observables: Q,
        debug_observables: D,
        stream: S,
    },
    Separate {
        quantum: Observables<Q, S>,
        debug: Observables<D, S>,
    },
}
