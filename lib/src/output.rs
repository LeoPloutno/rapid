use std::ops::{Deref, DerefMut};

use crate::{core::GroupTypeHandle, vector::Vector};

/// A trait for streams that write to coordinate files, such as '.xyz' files.
pub trait VectorsOutput<const N: usize, T, V>
where
    V: Vector<N, Element = T>,
{
    type Error;

    /// Write the coordinates of the atoms in all groups to the stream.
    fn write(&mut self, step: usize, vectors: &[GroupTypeHandle<V>]) -> Result<(), Self::Error>;
}

pub trait ObservablesOutput<const N: usize, T, V, E>
where
    V: Vector<N, Element = T>,
{
    type Input;
    type Error: From<E>;

    fn write(
        &mut self,
        step: usize,
        observables: &mut dyn Iterator<Item = Result<Self::Input, E>>,
    ) -> Result<(), Self::Error>;
}

pub trait ObservableOutput<T> {
    type Error;

    fn write_step(&mut self, step: usize) -> Result<(), Self::Error>;

    fn write_observable(&mut self, value: T) -> Result<(), Self::Error>;

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
        quantum_observables: Q,
        debug_observables: D,
        stream: S,
    },
    Separate {
        quantum: Observables<Q, S>,
        debug: Observables<D, S>,
    },
}

impl<Q: DerefMut, D: DerefMut, S: DerefMut> ObservableOutputOption<Q, D, S> {
    pub fn as_deref_mut(
        &mut self,
    ) -> ObservableOutputOption<&mut <Q as Deref>::Target, &mut <D as Deref>::Target, &mut <S as Deref>::Target> {
        match self {
            Self::None => ObservableOutputOption::None,
            Self::Quantum(Observables { observables, stream }) => {
                ObservableOutputOption::Quantum(Observables { observables, stream })
            }
            Self::Debug(Observables { observables, stream }) => {
                ObservableOutputOption::Debug(Observables { observables, stream })
            }
            Self::Shared {
                quantum_observables: quantum,
                debug_observables: debug,
                stream,
            } => ObservableOutputOption::Shared {
                quantum_observables: quantum,
                debug_observables: debug,
                stream,
            },
            Self::Separate {
                quantum:
                    Observables {
                        observables: quantum_observables,
                        stream: quantum_stream,
                    },
                debug:
                    Observables {
                        observables: debug_observables,
                        stream: debug_stream,
                    },
            } => ObservableOutputOption::Separate {
                quantum: Observables {
                    observables: quantum_observables,
                    stream: quantum_stream,
                },
                debug: Observables {
                    observables: debug_observables,
                    stream: debug_stream,
                },
            },
        }
    }
}

/// An enum which contains the obseervables, if any.
pub enum ObservableOption<Q, D> {
    None,
    Quantum(Q),
    Debug(D),
    All { quantum: Q, debug: D },
}
