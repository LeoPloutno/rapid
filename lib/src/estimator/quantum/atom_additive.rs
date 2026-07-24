//! Traits and types for qunatum estimators that can be expressed as a sum of observables
//! that depend only on a single atom.

use super::QuantumEstimator;
use crate::{
    core::{
        GroupInTypeInImage,
        error::{EmptyError, InvalidIndexError},
        sync_ops::SyncAddSender,
    },
    zip_items, zip_iterators,
};
use std::ops::Add;

/// A trait for quantum estimators that can be expressed as a sum
/// of estimators that each depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveQuantumEstimator<E>`]
/// automatically implements [`QuantumEstimator`].
pub trait AtomAdditiveQuantumEstimator<T: Clone, V> {
    /// The type of output `Self` and [`AdditiveQuantumEstimator<Self>`] return.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type AtomError;
    /// The type of error [`AdditiveQuantumEstimator<Self>`] returns.
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
        core::{GroupInTypeInImage, error::EmptyError, sync_ops::SyncAddSender},
        zip_items, zip_iterators,
    };
    use std::ops::Add;

    /// A wrapper for implementors of the [`AtomAdditiveQuantumEstimator<Output = T>`] trait.
    pub struct AdditiveValueQuantumEstimator<E: ?Sized>(pub(crate) E);

    impl<E> AdditiveValueQuantumEstimator<E> {
        /// Wraps the provided value with `AdditiveQuantumEstimator`.
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
        ) -> Result<T, Self::AtomError> {
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
        Self: AtomAdditiveQuantumEstimator<T, V, Output = T>,
        <Self as AtomAdditiveQuantumEstimator<T, V>>::SystemError: From<A::Error>,
    {
        type Output = T;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V>>::SystemError;

        fn calculate(
            &mut self,
            _barrier: &std::sync::Barrier,
            _shared_value: &std::sync::RwLock<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            exchange_potential_energy: T,
            positions: &GroupInTypeInImage<V>,
            physical_forces: &GroupInTypeInImage<V>,
            exchange_forces: &GroupInTypeInImage<V>,
        ) -> Result<(), Self::Error> {
            let mut iter = zip_iterators!(
                positions.read().iter(),
                physical_forces.read().iter(),
                exchange_forces.read().iter()
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
}

mod vector {
    use super::{super::QuantumEstimator, AtomAdditiveQuantumEstimator};
    use crate::{
        core::{GroupInTypeInImage, Vector, error::EmptyError, sync_ops::SyncAddSender},
        zip_items, zip_iterators,
    };
    use std::ops::Add;

    /// A wrapper for implementors of the [`AtomAdditiveQuantumEstimator<Output = V>`] trait,
    /// where `V` is a [vector](Vector).
    pub struct AdditiveVectorQuantumEstimator<E: ?Sized>(pub(crate) E);

    impl<E> AdditiveVectorQuantumEstimator<E> {
        /// Wraps the provided value with `AdditiveQuantumEstimator`.
        pub const fn new(value: E) -> Self {
            Self(value)
        }
    }

    impl<const N: usize, T, V, E> AtomAdditiveQuantumEstimator<T, V>
        for AdditiveVectorQuantumEstimator<E>
    where
        T: Clone,
        V: Vector<N, Element = T>,
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
        ) -> Result<T, Self::AtomError> {
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

    impl<T, V, A, M, E> QuantumEstimator<T, V, A, M, ()> for AdditiveVectorQuantumEstimator<E>
    where
        T: Clone + Add<Output = T>,
        A: SyncAddSender<T> + ?Sized,
        M: ?Sized,
        E: ?Sized,
        Self: AtomAdditiveQuantumEstimator<T, V, Output = T>,
        <Self as AtomAdditiveQuantumEstimator<T, V>>::SystemError: From<A::Error>,
    {
        type Output = T;
        type Error = <Self as AtomAdditiveQuantumEstimator<T, V>>::SystemError;

        fn calculate(
            &mut self,
            _barrier: &std::sync::Barrier,
            _shared_value: &std::sync::RwLock<T>,
            adder: &mut A,
            _multiplier: &mut M,
            physical_potential_energy: T,
            exchange_potential_energy: T,
            positions: &GroupInTypeInImage<V>,
            physical_forces: &GroupInTypeInImage<V>,
            exchange_forces: &GroupInTypeInImage<V>,
        ) -> Result<(), Self::Error> {
            let mut iter = zip_iterators!(
                positions.read().iter(),
                physical_forces.read().iter(),
                exchange_forces.read().iter()
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
}
