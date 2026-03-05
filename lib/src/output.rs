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

pub trait ValuesOutput<T> {
    type Error;

    fn write_step(&mut self, step: usize) -> Result<(), Self::Error>;

    fn write_value(&mut self, value: T) -> Result<(), Self::Error>;

    fn new_line(&mut self) -> Result<(), Self::Error>;
}

pub struct Estimators<T, U> {
    pub estimators: T,
    pub stream: U,
}

/// An enum which contains the output stream for the two kinds of observables.
pub enum ObservablesOutputOption<Q, C, S> {
    None,
    Quantum(Estimators<Q, S>),
    Classical(Estimators<C, S>),
    Shared {
        quantum_estimators: Q,
        debug_estimators: C,
        stream: S,
    },
    Separate {
        quantum: Estimators<Q, S>,
        debug: Estimators<C, S>,
    },
}

impl<Q: DerefMut, C: DerefMut, S: DerefMut> ObservablesOutputOption<Q, C, S> {
    pub fn as_deref_mut(
        &mut self,
    ) -> ObservablesOutputOption<&mut <Q as Deref>::Target, &mut <C as Deref>::Target, &mut <S as Deref>::Target> {
        match self {
            Self::None => ObservablesOutputOption::None,
            Self::Quantum(Estimators {
                estimators: observables,
                stream,
            }) => ObservablesOutputOption::Quantum(Estimators {
                estimators: observables,
                stream,
            }),
            Self::Classical(Estimators {
                estimators: observables,
                stream,
            }) => ObservablesOutputOption::Classical(Estimators {
                estimators: observables,
                stream,
            }),
            Self::Shared {
                quantum_estimators: quantum,
                debug_estimators: debug,
                stream,
            } => ObservablesOutputOption::Shared {
                quantum_estimators: quantum,
                debug_estimators: debug,
                stream,
            },
            Self::Separate {
                quantum:
                    Estimators {
                        estimators: quantum_observables,
                        stream: quantum_stream,
                    },
                debug:
                    Estimators {
                        estimators: debug_observables,
                        stream: debug_stream,
                    },
            } => ObservablesOutputOption::Separate {
                quantum: Estimators {
                    estimators: quantum_observables,
                    stream: quantum_stream,
                },
                debug: Estimators {
                    estimators: debug_observables,
                    stream: debug_stream,
                },
            },
        }
    }
}
