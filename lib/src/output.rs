//! Types and traits for printing out data collected during the simulation.

use std::ops::{Deref, DerefMut};

use crate::core::{GroupTypeHandle, Vector};

/// A trait for streams that write to coordinate files, such as '.xyz' files.
pub trait VectorsOutput<const N: usize, T, V>
where
    V: Vector<N, Element = T>,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Write the vectors of the atoms in all groups to the stream.
    fn write(&mut self, step: usize, vectors: &[GroupTypeHandle<V>]) -> Result<(), Self::Error>;
}

/// A trait for streams that write values into the output file.
pub trait ValuesOutput<T> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Writes the prelude.
    fn write_step(&mut self, step: usize) -> Result<(), Self::Error>;

    /// Writes the value.
    fn write_value(&mut self, value: T) -> Result<(), Self::Error>;

    /// Ends the current line and starts a new one.
    fn new_line(&mut self) -> Result<(), Self::Error>;
}

/// A struct which contains the estimators and the output stream.
pub struct ObservablesOutput<T, U> {
    /// The estimators.
    pub estimators: T,
    /// The stream.
    pub stream: U,
}

/// An enum which contains the estimators and output stream for the two kinds of observables.
pub enum ObservablesOutputOption<Q, C, S> {
    /// No observales.
    None,
    /// Only quantum estimators.
    Quantum(ObservablesOutput<Q, S>),
    /// Onnly classical estimators.
    Classical(ObservablesOutput<C, S>),
    /// Both kinds of estimators and a single stream dedicated to both.
    Shared {
        /// The quantum estimators.
        quantum_estimators: Q,
        /// The classical estimators.
        classical_estimators: C,
        /// The shared stream.
        stream: S,
    },
    /// Both kinds of estimators, each with its own dedicated stream.
    Separate {
        /// The quantum estimators and the stream dedicated to them.
        quantum: ObservablesOutput<Q, S>,
        /// The classical estimators and the stream dedicated to them.
        debug: ObservablesOutput<C, S>,
    },
}

impl<Q: DerefMut, C: DerefMut, S: DerefMut> ObservablesOutputOption<Q, C, S> {
    /// Converts from `ObservablesOutputOption<Q, C, S>` to
    /// `ObservablesOption<&mut Q::Target, &mut C::Target, &mut S::Target>`.
    ///
    /// Leaves the original `ObservablesOutputOption` in-place,
    /// creating a new one containing mutable references to the inner types' `Deref::Target` types.
    pub fn as_deref_mut(
        &mut self,
    ) -> ObservablesOutputOption<&mut <Q as Deref>::Target, &mut <C as Deref>::Target, &mut <S as Deref>::Target> {
        match self {
            Self::None => ObservablesOutputOption::None,
            Self::Quantum(ObservablesOutput {
                estimators: observables,
                stream,
            }) => ObservablesOutputOption::Quantum(ObservablesOutput {
                estimators: observables,
                stream,
            }),
            Self::Classical(ObservablesOutput {
                estimators: observables,
                stream,
            }) => ObservablesOutputOption::Classical(ObservablesOutput {
                estimators: observables,
                stream,
            }),
            Self::Shared {
                quantum_estimators: quantum,
                classical_estimators: debug,
                stream,
            } => ObservablesOutputOption::Shared {
                quantum_estimators: quantum,
                classical_estimators: debug,
                stream,
            },
            Self::Separate {
                quantum:
                    ObservablesOutput {
                        estimators: quantum_observables,
                        stream: quantum_stream,
                    },
                debug:
                    ObservablesOutput {
                        estimators: debug_observables,
                        stream: debug_stream,
                    },
            } => ObservablesOutputOption::Separate {
                quantum: ObservablesOutput {
                    estimators: quantum_observables,
                    stream: quantum_stream,
                },
                debug: ObservablesOutput {
                    estimators: debug_observables,
                    stream: debug_stream,
                },
            },
        }
    }
}
