//! Traits and types for quantum estimators that can be expressed as a sum of observables
//! that depend only on a single atom.

use crate::core::error::EmptyError;
use std::ops::Add;

/// A trait for quantum estimators that can be expressed as a sum
/// of estimators that each depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveValueQuantumEstimator<E>`] and
/// [`AdditiveVectorQuantumEstimator<E>`] automatically implement [`QuantumEstimator`].
///
/// [`QuantumEstimator`]: super::QuantumEstimator
pub trait AtomAdditiveQuantumEstimator<T: Clone, V> {
    /// The type of output `Self` and [`AdditiveValueQuantumEstimator<Self>`]/[`AdditiveVectorQuantumEstimator<Self>`] return.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type AtomError;
    /// The type of error [`AdditiveValueQuantumEstimator<Self>`]/[`AdditiveVectorQuantumEstimator<Self>`] return.
    type SystemError: From<Self::AtomError> + From<EmptyError>;

    /// Calculates the contribution of an atom to the contribution
    /// of the image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError>;
}

mod value {
    use super::{super::QuantumEstimator, AtomAdditiveQuantumEstimator};
    use crate::{
        core::{
            GroupInTypeInImage,
            error::EmptyError,
            marker::MeaningfulOutput,
            sync_ops::{SyncAddReceiver, SyncAddSender},
        },
        zip_items, zip_iterators,
    };
    use std::{
        ops::Add,
        sync::{Barrier, RwLock},
    };

    /// A wrapper for implementors of the [`AtomAdditiveQuantumEstimator<T, V, Output = T>`] trait.
    pub struct AdditiveValueQuantumEstimator<E: ?Sized>(pub(crate) E);

    impl<E> AdditiveValueQuantumEstimator<E> {
        /// Wraps the provided value with `AdditiveValueQuantumEstimator`.
        pub const fn new(value: E) -> Self {
            Self(value)
        }
    }

    impl<T, V, E> AtomAdditiveQuantumEstimator<T, V> for AdditiveValueQuantumEstimator<E>
    where
        T: Clone + Add<Output = T>,
        E: AtomAdditiveQuantumEstimator<T, V, Output = T> + ?Sized,
    {
        type Output = T;
        type AtomError = E::AtomError;
        type SystemError = E::SystemError;

        #[inline(always)]
        fn calculate(
            &mut self,
            atom_index: usize,
            physical_potential_energy: T,
            exchange_potential_energy: T,
            position: &V,
            physical_force: &V,
            exchange_force: &V,
        ) -> Result<Self::Output, Self::AtomError> {
            self.0.calculate(
                atom_index,
                physical_potential_energy,
                exchange_potential_energy,
                position,
                physical_force,
                exchange_force,
            )
        }
    }

    impl<T, V, A, M, E> QuantumEstimator<T, V, A, M, ()> for AdditiveValueQuantumEstimator<E>
    where
        T: Clone + Add<Output = T>,
        A: SyncAddSender<T> + ?Sized,
        M: ?Sized,
        E: ?Sized,
        Self: AtomAdditiveQuantumEstimator<T, V, Output = T, SystemError: From<A::Error>>,
    {
        type Output = T;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V>>::SystemError;

        fn calculate(
            &mut self,
            _barrier: &Barrier,
            _shared_value: &RwLock<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            exchange_potential_energy: T,
            positions: &GroupInTypeInImage<V>,
            physical_forces: &GroupInTypeInImage<V>,
            exchange_forces: &GroupInTypeInImage<V>,
        ) -> Result<(), Self::Error> {
            let mut iter = zip_iterators!(
                positions.read(),
                physical_forces.read(),
                exchange_forces.read()
            )
            .enumerate()
            .map(
                |(index, zip_items!(position, physical_force, exchange_force))| {
                    AtomAdditiveQuantumEstimator::calculate(
                        self,
                        index,
                        physical_potential_energy.clone(),
                        exchange_potential_energy.clone(),
                        position,
                        physical_force,
                        exchange_force,
                    )
                },
            );
            let first_atom_observable = iter.next().ok_or(EmptyError)??;
            let group_observable = iter.try_fold(
                first_atom_observable,
                |accum_observable, atom_observable| {
                    Ok::<_, <Self as AtomAdditiveQuantumEstimator<T, V>>::AtomError>(
                        accum_observable + atom_observable?,
                    )
                },
            )?;
            adder.send(group_observable)?;
            Ok(())
        }
    }

    impl<T, V, A, M, E> QuantumEstimator<T, V, A, M, T> for AdditiveValueQuantumEstimator<E>
    where
        T: Clone + Add<Output = T> + MeaningfulOutput,
        A: SyncAddReceiver<T> + ?Sized,
        M: ?Sized,
        E: ?Sized,
        Self: AtomAdditiveQuantumEstimator<T, V, Output = T, SystemError: From<A::Error>>,
    {
        type Output = T;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V>>::SystemError;

        fn calculate(
            &mut self,
            _barrier: &Barrier,
            _shared_value: &RwLock<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            exchange_potential_energy: T,
            positions: &GroupInTypeInImage<V>,
            physical_forces: &GroupInTypeInImage<V>,
            exchange_forces: &GroupInTypeInImage<V>,
        ) -> Result<T, Self::Error> {
            let mut iter = zip_iterators!(
                positions.read(),
                physical_forces.read(),
                exchange_forces.read()
            )
            .enumerate()
            .map(
                |(index, zip_items!(position, physical_force, exchange_force))| {
                    AtomAdditiveQuantumEstimator::calculate(
                        self,
                        index,
                        physical_potential_energy.clone(),
                        exchange_potential_energy.clone(),
                        position,
                        physical_force,
                        exchange_force,
                    )
                },
            );
            let first_atom_observable = iter.next().ok_or(EmptyError)??;
            let group_observable = iter.try_fold(
                first_atom_observable,
                |accum_observable, atom_observable| {
                    Ok::<_, <Self as AtomAdditiveQuantumEstimator<T, V>>::AtomError>(
                        accum_observable + atom_observable?,
                    )
                },
            )?;
            match adder.recv_sum()? {
                Some(other_groups_observable) => Ok(group_observable + other_groups_observable),
                None => Ok(group_observable),
            }
        }
    }
}
pub use value::AdditiveValueQuantumEstimator;

mod vector {
    use super::{super::QuantumEstimator, AtomAdditiveQuantumEstimator};
    use crate::{
        core::{
            GroupInTypeInImage, Vector,
            error::EmptyError,
            marker::MeaningfulOutput,
            sync_ops::{SyncAddReceiver, SyncAddSender},
        },
        zip_items, zip_iterators,
    };
    use std::{
        ops::Add,
        sync::{Barrier, RwLock},
    };

    /// A wrapper for implementors of the [`AtomAdditiveQuantumEstimator<T, V, Output = V>`] trait,
    /// where `V` is a [vector](Vector).
    pub struct AdditiveVectorQuantumEstimator<const N: usize, E: ?Sized>(pub(crate) E);

    impl<const N: usize, E> AdditiveVectorQuantumEstimator<N, E> {
        /// Wraps the provided value with `AdditiveVectorQuantumEstimator`.
        pub const fn new(value: E) -> Self {
            Self(value)
        }
    }

    impl<const N: usize, T, V, E> AtomAdditiveQuantumEstimator<T, V>
        for AdditiveVectorQuantumEstimator<N, E>
    where
        T: Clone + Add<Output = T>,
        V: Vector<N, Element = T>,
        E: AtomAdditiveQuantumEstimator<T, V, Output = V> + ?Sized,
    {
        type Output = V;
        type AtomError = E::AtomError;
        type SystemError = E::SystemError;

        #[inline(always)]
        fn calculate(
            &mut self,
            atom_index: usize,
            physical_potential_energy: T,
            exchange_potential_energy: T,
            position: &V,
            physical_force: &V,
            exchange_force: &V,
        ) -> Result<Self::Output, Self::AtomError> {
            self.0.calculate(
                atom_index,
                physical_potential_energy,
                exchange_potential_energy,
                position,
                physical_force,
                exchange_force,
            )
        }
    }

    impl<const N: usize, T, V, A, M, E> QuantumEstimator<T, V, A, M, ()>
        for AdditiveVectorQuantumEstimator<N, E>
    where
        T: Clone + Add<Output = T>,
        V: Vector<N, Element = T>,
        A: SyncAddSender<T> + ?Sized,
        M: ?Sized,
        E: ?Sized,
        Self: AtomAdditiveQuantumEstimator<T, V, Output = V, SystemError: From<A::Error>>,
    {
        type Output = V;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V>>::SystemError;

        fn calculate(
            &mut self,
            barrier: &Barrier,
            _shared_value: &RwLock<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            exchange_potential_energy: T,
            positions: &GroupInTypeInImage<V>,
            physical_forces: &GroupInTypeInImage<V>,
            exchange_forces: &GroupInTypeInImage<V>,
        ) -> Result<(), Self::Error> {
            let mut iter = zip_iterators!(
                positions.read(),
                physical_forces.read(),
                exchange_forces.read()
            )
            .enumerate()
            .map(
                |(index, zip_items!(position, physical_force, exchange_force))| {
                    AtomAdditiveQuantumEstimator::calculate(
                        self,
                        index,
                        physical_potential_energy.clone(),
                        exchange_potential_energy.clone(),
                        position,
                        physical_force,
                        exchange_force,
                    )
                },
            );
            let first_atom_observable = iter.next().ok_or(EmptyError)??;
            let group_observable = iter.try_fold(
                first_atom_observable,
                |accum_observable, atom_observable| {
                    Ok::<_, <Self as AtomAdditiveQuantumEstimator<T, V>>::AtomError>(
                        accum_observable + atom_observable?,
                    )
                },
            )?;
            for element in group_observable.as_array() {
                adder.send(element.clone())?;
                barrier.wait();
                // The receiver receives the sum of all sent values.
                barrier.wait();
            }
            Ok(())
        }
    }

    impl<const N: usize, T, V, A, M, E> QuantumEstimator<T, V, A, M, V>
        for AdditiveVectorQuantumEstimator<N, E>
    where
        T: Clone + Add<Output = T>,
        V: Vector<N, Element = T> + MeaningfulOutput,
        A: SyncAddReceiver<T> + ?Sized,
        M: ?Sized,
        E: ?Sized,
        Self: AtomAdditiveQuantumEstimator<T, V, Output = V, SystemError: From<A::Error>>,
    {
        type Output = V;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V>>::SystemError;

        fn calculate(
            &mut self,
            barrier: &Barrier,
            _shared_value: &RwLock<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            exchange_potential_energy: T,
            positions: &GroupInTypeInImage<V>,
            physical_forces: &GroupInTypeInImage<V>,
            exchange_forces: &GroupInTypeInImage<V>,
        ) -> Result<V, Self::Error> {
            let mut iter = zip_iterators!(
                positions.read(),
                physical_forces.read(),
                exchange_forces.read()
            )
            .enumerate()
            .map(
                |(index, zip_items!(position, physical_force, exchange_force))| {
                    AtomAdditiveQuantumEstimator::calculate(
                        self,
                        index,
                        physical_potential_energy.clone(),
                        exchange_potential_energy.clone(),
                        position,
                        physical_force,
                        exchange_force,
                    )
                },
            );
            let first_atom_observable = iter.next().ok_or(EmptyError)??;
            let mut group_observable = iter.try_fold(
                first_atom_observable,
                |accum_observable, atom_observable| {
                    Ok::<_, <Self as AtomAdditiveQuantumEstimator<T, V>>::AtomError>(
                        accum_observable + atom_observable?,
                    )
                },
            )?;
            for element in group_observable.as_mut_array() {
                // The senders send their values.
                barrier.wait();
                let other_groups_element = adder.recv_sum()?;
                barrier.wait();
                if let Some(other_groups_element) = other_groups_element {
                    *element = element.clone() + other_groups_element
                }
            }
            Ok(group_observable)
        }
    }
}
pub use vector::AdditiveVectorQuantumEstimator;
