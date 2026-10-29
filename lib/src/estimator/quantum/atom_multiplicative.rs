//! Traits and types for quantum estimators that can be expressed as a product of observables
//! that depend only on a single atom.

use super::QuantumEstimator;
use crate::{
    core::{
        GroupInTypeInImage, Synchronizer,
        error::EmptyError,
        marker::MeaningfulOutput,
        sync_ops::{SyncMulReceiver, SyncMulSender},
    },
    zip_items, zip_iterators,
};
use std::{
    error::Error,
    ops::{Deref, DerefMut, Mul},
};

/// A trait for quantum estimators that can be expressed as a product
/// of estimators that each depend only on a single atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeValueQuantumEstimator<E>`]
/// automatically implements [`QuantumEstimator`].
pub trait AtomMultiplicativeQuantumEstimator<T: Clone, V, E> {
    /// The type of output `Self` and [`MultiplicativeValueQuantumEstimator<Self>`] return.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type AtomError: Error;
    /// The type of error [`MultiplicativeValueQuantumEstimator<Self>`] returns.
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

/// A wrapper for implementors of the [`AtomMultiplicativeQuantumEstimator<T, V, Output = T>`] trait.
pub struct MultiplicativeValueQuantumEstimator<E: ?Sized>(pub(crate) E);

impl<E> MultiplicativeValueQuantumEstimator<E> {
    /// Wraps the provided value with `MultiplicativeValueQuantumEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

impl<T, V, E, Q> AtomMultiplicativeQuantumEstimator<T, V, E> for MultiplicativeValueQuantumEstimator<Q>
where
    T: Mul<Output = T> + Clone,
    Q: AtomMultiplicativeQuantumEstimator<T, V, E, Output = T> + ?Sized,
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

impl<T, V, E, Q> AtomMultiplicativeQuantumEstimator<T, V, E> for Q
where
    T: Clone,
    Q: DerefMut<Target: AtomMultiplicativeQuantumEstimator<T, V, E>> + ?Sized,
{
    type Output = <<Self as Deref>::Target as AtomMultiplicativeQuantumEstimator<T, V, E>>::Output;
    type AtomError = <<Self as Deref>::Target as AtomMultiplicativeQuantumEstimator<T, V, E>>::AtomError;
    type SystemError = <<Self as Deref>::Target as AtomMultiplicativeQuantumEstimator<T, V, E>>::SystemError;

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

impl<T, V, A, M, E> QuantumEstimator<T, V, A, M, ()> for MultiplicativeValueQuantumEstimator<E>
where
    T: Mul<Output = T> + Clone,
    A: ?Sized,
    M: SyncMulSender<T> + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeQuantumEstimator<T, V, M::Error, Output = T>,
{
    type Output = T;
    type Error = <Self as AtomMultiplicativeQuantumEstimator<T, V, M::Error>>::SystemError;

    fn calculate(
        &mut self,
        _synchrinizer: &Synchronizer<T>,
        _adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        positions: &GroupInTypeInImage<T, V>,
        physical_forces: &GroupInTypeInImage<T, V>,
        exchange_forces: &GroupInTypeInImage<T, V>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(positions.read(), physical_forces.read(), exchange_forces.read())
            .enumerate()
            .map(|(index, zip_items!(position, physical_force, exchange_force))| {
                AtomMultiplicativeQuantumEstimator::calculate(
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
            Ok::<_, <Self as AtomMultiplicativeQuantumEstimator<T, V, M::Error>>::AtomError>(
                accum_observable * atom_observable?,
            )
        })?;
        multiplier.send(group_observable)?;
        Ok(())
    }
}

impl<T, V, A, M, E> QuantumEstimator<T, V, A, M, T> for MultiplicativeValueQuantumEstimator<E>
where
    T: Mul<Output = T> + Clone + MeaningfulOutput,
    A: ?Sized,
    M: SyncMulReceiver<T> + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeQuantumEstimator<T, V, M::Error, Output = T>,
{
    type Output = T;
    type Error = <Self as AtomMultiplicativeQuantumEstimator<T, V, M::Error>>::SystemError;

    fn calculate(
        &mut self,
        _synchronizer: &Synchronizer<T>,
        _adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        positions: &GroupInTypeInImage<T, V>,
        physical_forces: &GroupInTypeInImage<T, V>,
        exchange_forces: &GroupInTypeInImage<T, V>,
    ) -> Result<T, Self::Error> {
        let mut iter = zip_iterators!(positions.read(), physical_forces.read(), exchange_forces.read())
            .enumerate()
            .map(|(index, zip_items!(position, physical_force, exchange_force))| {
                AtomMultiplicativeQuantumEstimator::calculate(
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
            Ok::<_, <Self as AtomMultiplicativeQuantumEstimator<T, V, M::Error>>::AtomError>(
                accum_observable * atom_observable?,
            )
        })?;
        match multiplier.recv_prod()? {
            Some(other_groups_observable) => Ok(group_observable * other_groups_observable),
            None => Ok(group_observable),
        }
    }
}
