//! Types and traits for printing out data collected during the simulation.

use std::ops::{Deref, DerefMut};

use crate::core::{GroupInTypeInImageInSystem, Vector};

/// A trait for streams that write into a file as the simulation progresses.
pub trait StepStream {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Writes a prelude.
    fn write_prelude(&mut self, step: usize) -> Result<(), Self::Error>;

    /// Ends the current line and starts a new one.
    fn new_line(&mut self) -> Result<(), Self::Error>;
}

/// A trait for streams that write to coordinate files, such as '.xyz' files.
pub trait VectorsStream<const N: usize, T, V>: StepStream
where
    V: Vector<N, Element = T>,
{
    /// Writes the vectors.
    fn write_vectors(&mut self, vectors: &GroupInTypeInImageInSystem<V>)
    -> Result<(), Self::Error>;
}

/// A trait for streams that write values into the output file.
pub trait ValuesStream<T>: StepStream {
    /// Writes the value.
    fn write_value(&mut self, value: T) -> Result<(), Self::Error>;
}

/// A struct which contains the estimators and the output stream.
pub struct EstimatorsOutput<E, S> {
    /// The estimators.
    pub estimators: E,
    /// The stream.
    pub stream: S,
}

/// An enum which contains the estimators and output stream for the two kinds of observables.
pub enum EstimatorsOutputOption<Q, C, S> {
    /// No observales.
    None,
    /// Only quantum estimators.
    Quantum(EstimatorsOutput<Q, S>),
    /// Onnly classical estimators.
    Classical(EstimatorsOutput<C, S>),
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
        quantum: EstimatorsOutput<Q, S>,
        /// The classical estimators and the stream dedicated to them.
        classical: EstimatorsOutput<C, S>,
    },
}

impl<Q: DerefMut, C: DerefMut, S: DerefMut> EstimatorsOutputOption<Q, C, S> {
    /// Converts from `ObservablesOutputOption<Q, C, S>` to
    /// `ObservablesOption<&mut Q::Target, &mut C::Target, &mut S::Target>`.
    ///
    /// Leaves the original `ObservablesOutputOption` in-place,
    /// creating a new one containing mutable references to the inner types' `Deref::Target` types.
    pub fn as_deref_mut(
        &mut self,
    ) -> EstimatorsOutputOption<
        &mut <Q as Deref>::Target,
        &mut <C as Deref>::Target,
        &mut <S as Deref>::Target,
    > {
        match self {
            Self::None => EstimatorsOutputOption::None,
            Self::Quantum(EstimatorsOutput {
                estimators: observables,
                stream,
            }) => EstimatorsOutputOption::Quantum(EstimatorsOutput {
                estimators: observables,
                stream,
            }),
            Self::Classical(EstimatorsOutput {
                estimators: observables,
                stream,
            }) => EstimatorsOutputOption::Classical(EstimatorsOutput {
                estimators: observables,
                stream,
            }),
            Self::Shared {
                quantum_estimators: quantum,
                classical_estimators: classical,
                stream,
            } => EstimatorsOutputOption::Shared {
                quantum_estimators: quantum,
                classical_estimators: classical,
                stream,
            },
            Self::Separate {
                quantum:
                    EstimatorsOutput {
                        estimators: quantum_observables,
                        stream: quantum_stream,
                    },
                classical:
                    EstimatorsOutput {
                        estimators: classical_observables,
                        stream: classical_stream,
                    },
            } => EstimatorsOutputOption::Separate {
                quantum: EstimatorsOutput {
                    estimators: quantum_observables,
                    stream: quantum_stream,
                },
                classical: EstimatorsOutput {
                    estimators: classical_observables,
                    stream: classical_stream,
                },
            },
        }
    }
}
