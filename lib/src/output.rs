use crate::{core::AtomGroupInfo, vector::Vector};

/// A trait for streams that write to coordinate files, such as '.xyz' files.
pub trait VectorsOutput<const N: usize, T, V>
where
    V: Vector<N, Element = T>,
{
    type Error;

    /// Write the coordinates of the atoms in all groups to the stream.
    #[must_use]
    fn write(&mut self, step: usize, groups: &[AtomGroupInfo<T>], vectors: &[V]) -> Result<(), Self::Error>;
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

pub trait ObservableOutput<T> {
    type Error;

    #[must_use]
    fn write_step(&mut self, step: usize) -> Result<(), Self::Error>;

    #[must_use]
    fn write_observable(&mut self, value: T) -> Result<(), Self::Error>;

    #[must_use]
    fn new_line(&mut self) -> Result<(), Self::Error>;
}

pub struct Observables<O, S> {
    pub observables: O,
    pub stream: S,
}

/// An enum which contains the output stream for the two kinds of observables.
pub enum ObservableOutputOption<Q, D, S> {
    None,
    Quantum(Observables<Q, S>),
    Debug(Observables<D, S>),
    Shared {
        quantum: Q,
        debug: D,
        stream: S,
    },
    Separate {
        quantum: Observables<Q, S>,
        debug: Observables<D, S>,
    },
}

/// An enum which contains the obseervables, if any.
pub enum ObservableOption<Q, D> {
    None,
    Quantum(Q),
    Debug(D),
    All { quantum: Q, debug: D },
}
