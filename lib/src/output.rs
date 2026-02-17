use std::ops::{Deref, DerefMut};

use crate::vector::Vector;

/// A trait for streams that write to coordinate files, such as '.xyz' files.
pub trait VectorsOutput<const N: usize, T, V>
where
    V: Vector<N, Element = T>,
{
    type Error;

    /// Write the coordinates of the atoms in all groups to the stream.
    #[must_use]
    fn write(&mut self, step: usize, vectors: &[V]) -> Result<(), Self::Error>;
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

pub enum ObservableStreamOption<S> {
    None,
    One(S),
    Shared(S),
    All { quantum: S, debug: S },
}

impl<Q, D, S> ObservableOutputOption<Q, D, S> {
    pub fn as_deref_mut(
        &mut self,
    ) -> ObservableOutputOption<&mut <Q as Deref>::Target, &mut <D as Deref>::Target, &mut <S as Deref>::Target>
    where
        Q: DerefMut,
        D: DerefMut,
        S: DerefMut,
    {
        match self {
            Self::None => ObservableOutputOption::None,
            Self::Quantum(Observables { observables, stream }) => ObservableOutputOption::Quantum(Observables {
                observables: &mut *observables,
                stream: &mut *stream,
            }),
            Self::Debug(Observables { observables, stream }) => ObservableOutputOption::Debug(Observables {
                observables: &mut *observables,
                stream: &mut *stream,
            }),
            Self::Shared { quantum, debug, stream } => ObservableOutputOption::Shared {
                quantum: &mut *quantum,
                debug: &mut *debug,
                stream: &mut *stream,
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
                    observables: &mut *quantum_observables,
                    stream: &mut *quantum_stream,
                },
                debug: Observables {
                    observables: &mut *debug_observables,
                    stream: &mut *debug_stream,
                },
            },
        }
    }
    pub fn split(self) -> (Option<Q>, Option<D>, ObservableStreamOption<S>) {
        match self {
            Self::None => (None, None, ObservableStreamOption::None),
            Self::Quantum(Observables { observables, stream }) => {
                (Some(observables), None, ObservableStreamOption::One(stream))
            }
            Self::Debug(Observables { observables, stream }) => {
                (None, Some(observables), ObservableStreamOption::One(stream))
            }
            Self::Shared { quantum, debug, stream } => {
                (Some(quantum), Some(debug), ObservableStreamOption::Shared(stream))
            }
            Self::Separate { quantum, debug } => (
                Some(quantum.observables),
                Some(debug.observables),
                ObservableStreamOption::All {
                    quantum: quantum.stream,
                    debug: debug.stream,
                },
            ),
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
