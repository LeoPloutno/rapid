//! Types and traits for printing out data collected during the simulation.

use std::{
    error::Error,
    ops::{Deref, DerefMut},
};

use crate::core::GroupInTypeInImageInSystem;

/// A trait for streams that write into a file as the simulation progresses.
pub trait StepStream {
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Writes a prelude.
    fn write_prelude(&mut self, step: usize) -> Result<(), Self::Error>;

    /// Ends the current line and starts a new one.
    fn new_line(&mut self) -> Result<(), Self::Error>;
}

impl<S> StepStream for S
where
    S: DerefMut<Target: StepStream> + ?Sized,
{
    type Error = <<Self as Deref>::Target as StepStream>::Error;

    #[inline]
    fn write_prelude(&mut self, step: usize) -> Result<(), Self::Error> {
        <<Self as Deref>::Target as StepStream>::write_prelude(&mut *self, step)
    }

    #[inline]
    fn new_line(&mut self) -> Result<(), Self::Error> {
        <<Self as Deref>::Target as StepStream>::new_line(&mut *self)
    }
}

/// A trait for streams that write to coordinate files, such as '.xyz' files.
pub trait VectorsStream<T, V>: StepStream {
    /// Writes the vectors.
    fn write_vectors(&mut self, vectors: &GroupInTypeInImageInSystem<T, V>) -> Result<(), Self::Error>;
}

impl<T, V, S> VectorsStream<T, V> for S
where
    S: DerefMut<Target: VectorsStream<T, V>> + ?Sized,
{
    #[inline]
    fn write_vectors(&mut self, vectors: &GroupInTypeInImageInSystem<T, V>) -> Result<(), Self::Error> {
        <<Self as Deref>::Target as VectorsStream<T, V>>::write_vectors(&mut *self, vectors)
    }
}

/// A trait for streams that write values into the output file.
pub trait ValuesStream<T>: StepStream {
    /// Writes the value.
    fn write_value(&mut self, value: T) -> Result<(), Self::Error>;
}

impl<T, S> ValuesStream<T> for S
where
    S: DerefMut<Target: ValuesStream<T>> + ?Sized,
{
    #[inline]
    fn write_value(&mut self, value: T) -> Result<(), Self::Error> {
        <<Self as Deref>::Target as ValuesStream<T>>::write_value(&mut *self, value)
    }
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
    ) -> EstimatorsOutputOption<&mut <Q as Deref>::Target, &mut <C as Deref>::Target, &mut <S as Deref>::Target> {
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
