//! Traits and types for quantum estimators that can be expressed as a sum of observables
//! that depend only on a single atom.

use crate::core::error::EmptyError;
use std::{
    error::Error,
    ops::{Add, Deref, DerefMut},
};

/// A trait for quantum estimators that can be expressed as a sum
/// of estimators that each depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveValueQuantumEstimator<E>`] and
/// [`AdditiveVectorQuantumEstimator<E>`] automatically implement [`QuantumEstimator`].
///
/// [`QuantumEstimator`]: super::QuantumEstimator
pub trait AtomAdditiveQuantumEstimator<T: Clone, V, E> {
    /// The type of output `Self` and [`AdditiveValueQuantumEstimator<Self>`]/[`AdditiveVectorQuantumEstimator<Self>`] return.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type AtomError: Error;
    /// The type of error [`AdditiveValueQuantumEstimator<Self>`]/[`AdditiveVectorQuantumEstimator<Self>`] return.
    type SystemError: Error + From<Self::AtomError> + From<EmptyError> + From<E>;

    /// Calculates the contribution of an atom to the contribution
    /// of the image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError>;
}

impl<T, V, E, Q> AtomAdditiveQuantumEstimator<T, V, E> for Q
where
    T: Clone,
    Q: DerefMut<Target: AtomAdditiveQuantumEstimator<T, V, E>> + ?Sized,
{
    type Output = <<Self as Deref>::Target as AtomAdditiveQuantumEstimator<T, V, E>>::Output;
    type AtomError = <<Self as Deref>::Target as AtomAdditiveQuantumEstimator<T, V, E>>::AtomError;
    type SystemError = <<Self as Deref>::Target as AtomAdditiveQuantumEstimator<T, V, E>>::SystemError;

    #[inline]
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError> {
        (*self).calculate(
            atom_index,
            physical_potential_energy,
            type_exchange_potential_energy,
            position,
            physical_force,
            exchange_force,
        )
    }
}

mod value {
    use super::{super::QuantumEstimator, AtomAdditiveQuantumEstimator};
    use crate::{
        core::{
            GroupInTypeInImage, Synchronizer,
            error::EmptyError,
            marker::MeaningfulOutput,
            sync_ops::{SyncAddReceiver, SyncAddSender},
        },
        zip_items, zip_iterators,
    };
    use std::ops::Add;

    /// A wrapper for implementors of the [`AtomAdditiveQuantumEstimator<T, V, Output = T>`] trait.
    pub struct AdditiveValueQuantumEstimator<E: ?Sized>(pub(crate) E);

    impl<E> AdditiveValueQuantumEstimator<E> {
        /// Wraps the provided value with `AdditiveValueQuantumEstimator`.
        pub const fn new(value: E) -> Self {
            Self(value)
        }
    }

    impl<T, V, E, Q> AtomAdditiveQuantumEstimator<T, V, E> for AdditiveValueQuantumEstimator<Q>
    where
        T: Add<Output = T> + Clone,
        Q: AtomAdditiveQuantumEstimator<T, V, E, Output = T> + ?Sized,
    {
        type Output = T;
        type AtomError = Q::AtomError;
        type SystemError = Q::SystemError;

        #[inline]
        fn calculate(
            &mut self,
            atom_index: usize,
            physical_potential_energy: T,
            type_exchange_potential_energy: T,
            position: &V,
            physical_force: &V,
            exchange_force: &V,
        ) -> Result<Self::Output, Self::AtomError> {
            self.0.calculate(
                atom_index,
                physical_potential_energy,
                type_exchange_potential_energy,
                position,
                physical_force,
                exchange_force,
            )
        }
    }

    impl<T, V, A, M, E> QuantumEstimator<T, V, A, M, ()> for AdditiveValueQuantumEstimator<E>
    where
        T: Add<Output = T> + Clone,
        A: SyncAddSender<T> + ?Sized,
        M: ?Sized,
        E: ?Sized,
        Self: AtomAdditiveQuantumEstimator<T, V, A::Error, Output = T>,
    {
        type Output = T;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V, A::Error>>::SystemError;

        fn calculate(
            &mut self,
            _synchronizer: &Synchronizer<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            type_exchange_potential_energy: T,
            positions: &GroupInTypeInImage<T, V>,
            physical_forces: &GroupInTypeInImage<T, V>,
            exchange_forces: &GroupInTypeInImage<T, V>,
        ) -> Result<(), Self::Error> {
            let mut iter = zip_iterators!(positions.read(), physical_forces.read(), exchange_forces.read())
                .enumerate()
                .map(|(index, zip_items!(position, physical_force, exchange_force))| {
                    AtomAdditiveQuantumEstimator::calculate(
                        self,
                        index,
                        physical_potential_energy.clone(),
                        type_exchange_potential_energy.clone(),
                        position,
                        physical_force,
                        exchange_force,
                    )
                });
            let first_atom_observable = iter.next().ok_or(EmptyError)??;
            let group_observable = iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as AtomAdditiveQuantumEstimator<T, V, A::Error>>::AtomError>(
                    accum_observable + atom_observable?,
                )
            })?;
            adder.send(group_observable)?;
            Ok(())
        }
    }

    impl<T, V, A, M, E> QuantumEstimator<T, V, A, M, T> for AdditiveValueQuantumEstimator<E>
    where
        T: Add<Output = T> + Clone + MeaningfulOutput,
        A: SyncAddReceiver<T> + ?Sized,
        M: ?Sized,
        E: ?Sized,
        Self: AtomAdditiveQuantumEstimator<T, V, A::Error, Output = T>,
    {
        type Output = T;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V, A::Error>>::SystemError;

        fn calculate(
            &mut self,
            _synchronizer: &Synchronizer<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            type_exchange_potential_energy: T,
            positions: &GroupInTypeInImage<T, V>,
            physical_forces: &GroupInTypeInImage<T, V>,
            exchange_forces: &GroupInTypeInImage<T, V>,
        ) -> Result<T, Self::Error> {
            let mut iter = zip_iterators!(positions.read(), physical_forces.read(), exchange_forces.read())
                .enumerate()
                .map(|(index, zip_items!(position, physical_force, exchange_force))| {
                    AtomAdditiveQuantumEstimator::calculate(
                        self,
                        index,
                        physical_potential_energy.clone(),
                        type_exchange_potential_energy.clone(),
                        position,
                        physical_force,
                        exchange_force,
                    )
                });
            let first_atom_observable = iter.next().ok_or(EmptyError)??;
            let group_observable = iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as AtomAdditiveQuantumEstimator<T, V, A::Error>>::AtomError>(
                    accum_observable + atom_observable?,
                )
            })?;
            Ok(match adder.recv_sum()? {
                Some(other_groups_observable) => group_observable + other_groups_observable,
                None => group_observable,
            })
        }
    }
}
pub use value::AdditiveValueQuantumEstimator;

mod vector {
    use super::{super::QuantumEstimator, AtomAdditiveQuantumEstimator};
    use crate::{
        core::{
            GroupInTypeInImage, Synchronizer, Vector,
            error::EmptyError,
            marker::MeaningfulOutput,
            sync_ops::{SyncAddReceiver, SyncAddSender},
        },
        zip_items, zip_iterators,
    };
    use std::ops::Add;

    /// A wrapper for implementors of the [`AtomAdditiveQuantumEstimator<T, V, Output = V>`] trait,
    /// where `V` is a [vector](Vector).
    pub struct AdditiveVectorQuantumEstimator<E: ?Sized, const N: usize>(pub(crate) E);

    impl<E, const N: usize> AdditiveVectorQuantumEstimator<E, N> {
        /// Wraps the provided value with `AdditiveVectorQuantumEstimator`.
        pub const fn new(value: E) -> Self {
            Self(value)
        }
    }

    impl<T, V, E, Q, const N: usize> AtomAdditiveQuantumEstimator<T, V, E> for AdditiveVectorQuantumEstimator<Q, N>
    where
        T: Add<Output = T> + Clone,
        V: Vector<N, Element = T>,
        Q: AtomAdditiveQuantumEstimator<T, V, E, Output = V> + ?Sized,
    {
        type Output = V;
        type AtomError = Q::AtomError;
        type SystemError = Q::SystemError;

        #[inline]
        fn calculate(
            &mut self,
            atom_index: usize,
            physical_potential_energy: T,
            type_exchange_potential_energy: T,
            position: &V,
            physical_force: &V,
            exchange_force: &V,
        ) -> Result<Self::Output, Self::AtomError> {
            self.0.calculate(
                atom_index,
                physical_potential_energy,
                type_exchange_potential_energy,
                position,
                physical_force,
                exchange_force,
            )
        }
    }

    impl<T, V, A, M, E, const N: usize> QuantumEstimator<T, V, A, M, ()> for AdditiveVectorQuantumEstimator<E, N>
    where
        T: Add<Output = T> + Clone,
        V: Vector<N, Element = T>,
        A: SyncAddSender<T> + ?Sized,
        M: ?Sized,
        E: ?Sized,
        Self: AtomAdditiveQuantumEstimator<T, V, A::Error, Output = V>,
    {
        type Output = V;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V, A::Error>>::SystemError;

        fn calculate(
            &mut self,
            Synchronizer { barrier, .. }: &Synchronizer<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            type_exchange_potential_energy: T,
            positions: &GroupInTypeInImage<T, V>,
            physical_forces: &GroupInTypeInImage<T, V>,
            exchange_forces: &GroupInTypeInImage<T, V>,
        ) -> Result<(), Self::Error> {
            let mut iter = zip_iterators!(positions.read(), physical_forces.read(), exchange_forces.read())
                .enumerate()
                .map(|(index, zip_items!(position, physical_force, exchange_force))| {
                    AtomAdditiveQuantumEstimator::calculate(
                        self,
                        index,
                        physical_potential_energy.clone(),
                        type_exchange_potential_energy.clone(),
                        position,
                        physical_force,
                        exchange_force,
                    )
                });
            let first_atom_observable = iter.next().ok_or(EmptyError)??;
            let group_observable = iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as AtomAdditiveQuantumEstimator<T, V, A::Error>>::AtomError>(
                    accum_observable + atom_observable?,
                )
            })?;
            for element in group_observable.as_array() {
                adder.send(element.clone())?;
                barrier.wait();
                // The receiver receives the sum of all sent values.
                barrier.wait();
            }
            Ok(())
        }
    }

    impl<T, V, A, M, E, const N: usize> QuantumEstimator<T, V, A, M, V> for AdditiveVectorQuantumEstimator<E, N>
    where
        T: Add<Output = T> + Clone,
        V: Vector<N, Element = T> + MeaningfulOutput,
        A: SyncAddReceiver<T> + ?Sized,
        M: ?Sized,
        E: ?Sized,
        Self: AtomAdditiveQuantumEstimator<T, V, A::Error, Output = V>,
    {
        type Output = V;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V, A::Error>>::SystemError;

        fn calculate(
            &mut self,
            Synchronizer { barrier, .. }: &Synchronizer<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            type_exchange_potential_energy: T,
            positions: &GroupInTypeInImage<T, V>,
            physical_forces: &GroupInTypeInImage<T, V>,
            exchange_forces: &GroupInTypeInImage<T, V>,
        ) -> Result<V, Self::Error> {
            let mut iter = zip_iterators!(positions.read(), physical_forces.read(), exchange_forces.read())
                .enumerate()
                .map(|(index, zip_items!(position, physical_force, exchange_force))| {
                    AtomAdditiveQuantumEstimator::calculate(
                        self,
                        index,
                        physical_potential_energy.clone(),
                        type_exchange_potential_energy.clone(),
                        position,
                        physical_force,
                        exchange_force,
                    )
                });
            let first_atom_observable = iter.next().ok_or(EmptyError)??;
            let mut group_observable = iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as AtomAdditiveQuantumEstimator<T, V, A::Error>>::AtomError>(
                    accum_observable + atom_observable?,
                )
            })?;
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
