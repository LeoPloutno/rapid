//! Traits and types for classical estimators that can be expressed as a product of observables
//! that depend only on a single atom.

use super::ClassicalEstimator;
use crate::{
    core::{
        GroupInTypeInImageInSystem, Synchronizer,
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

/// A trait for classical estimators that can be expressed as a product
/// of estimators that each depend only on a single atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeValueClassicalEstimator<E>`]
/// automatically implements [`ClassicalEstimator`].
pub trait AtomMultiplicativeClassicalEstimator<T: Clone, V, E> {
    /// The type of output `Self` and [`MultiplicativeValueClassicalEstimator<Self>`] return.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type AtomError: Error;
    /// The type of error [`MultiplicativeValueClassicalEstimator<Self>`] returns.
    type SystemError: Error + From<Self::AtomError> + From<EmptyError> + From<E>;

    /// Calculates the contribution of an atom to the contribution
    /// of the image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        group_heat: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError>;
}

/// A wrapper for implementors of the [`AtomMultiplicativeClassicalEstimator<T, V, Output = T>`] trait.
pub struct MultiplicativeValueClassicalEstimator<E: ?Sized>(pub(crate) E);

impl<E> MultiplicativeValueClassicalEstimator<E> {
    /// Wraps the provided value with `MultiplicativeValueClassicalEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

impl<T, V, E, C> AtomMultiplicativeClassicalEstimator<T, V, E> for MultiplicativeValueClassicalEstimator<C>
where
    T: Mul<Output = T> + Clone,
    C: AtomMultiplicativeClassicalEstimator<T, V, E, Output = T> + ?Sized,
{
    type Output = T;
    type AtomError = C::AtomError;
    type SystemError = C::SystemError;

    #[inline]
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        group_heat: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError> {
        self.0.calculate(
            atom_index,
            physical_potential_energy,
            type_exchange_potential_energy,
            group_heat,
            position,
            momentum,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, E, Q> AtomMultiplicativeClassicalEstimator<T, V, E> for Q
where
    T: Clone,
    Q: DerefMut<Target: AtomMultiplicativeClassicalEstimator<T, V, E>> + ?Sized,
{
    type Output = <<Self as Deref>::Target as AtomMultiplicativeClassicalEstimator<T, V, E>>::Output;
    type AtomError = <<Self as Deref>::Target as AtomMultiplicativeClassicalEstimator<T, V, E>>::AtomError;
    type SystemError = <<Self as Deref>::Target as AtomMultiplicativeClassicalEstimator<T, V, E>>::SystemError;

    #[inline]
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        group_heat: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError> {
        <<Self as Deref>::Target as AtomMultiplicativeClassicalEstimator<T, V, E>>::calculate(
            &mut *self,
            atom_index,
            physical_potential_energy,
            type_exchange_potential_energy,
            group_heat,
            position,
            momentum,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, A, M, E> ClassicalEstimator<T, V, A, M, ()> for MultiplicativeValueClassicalEstimator<E>
where
    T: Mul<Output = T> + Clone,
    A: ?Sized,
    M: SyncMulSender<T> + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeClassicalEstimator<T, V, M::Error, Output = T>,
{
    type Output = T;
    type Error = <Self as AtomMultiplicativeClassicalEstimator<T, V, M::Error>>::SystemError;

    fn calculate(
        &mut self,
        _system_synchronizer: &Synchronizer<T>,
        _image_synchronizer: &Synchronizer<T>,
        _adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        group_heat: T,
        positions: &GroupInTypeInImageInSystem<T, V>,
        momenta: &GroupInTypeInImageInSystem<T, V>,
        physical_forces: &GroupInTypeInImageInSystem<T, V>,
        exchange_forces: &GroupInTypeInImageInSystem<T, V>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            positions.read(),
            momenta.read(),
            physical_forces.read(),
            exchange_forces.read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, momentum, physical_force, exchange_force))| {
                AtomMultiplicativeClassicalEstimator::calculate(
                    self,
                    index,
                    physical_potential_energy.clone(),
                    type_exchange_potential_energy.clone(),
                    group_heat.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        let group_observable = iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
            Ok::<_, <Self as AtomMultiplicativeClassicalEstimator<T, V, M::Error>>::AtomError>(
                accum_observable * atom_observable?,
            )
        })?;
        multiplier.send(group_observable)?;
        Ok(())
    }
}

impl<T, V, A, M, E> ClassicalEstimator<T, V, A, M, T> for MultiplicativeValueClassicalEstimator<E>
where
    T: Mul<Output = T> + Clone + MeaningfulOutput,
    A: ?Sized,
    M: SyncMulReceiver<T> + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeClassicalEstimator<T, V, M::Error, Output = T>,
{
    type Output = T;
    type Error = <Self as AtomMultiplicativeClassicalEstimator<T, V, M::Error>>::SystemError;

    fn calculate(
        &mut self,
        _system_synchronizer: &Synchronizer<T>,
        _image_synchronizer: &Synchronizer<T>,
        _adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        type_exchange_potential_energy: T,
        group_heat: T,
        positions: &GroupInTypeInImageInSystem<T, V>,
        momenta: &GroupInTypeInImageInSystem<T, V>,
        physical_forces: &GroupInTypeInImageInSystem<T, V>,
        exchange_forces: &GroupInTypeInImageInSystem<T, V>,
    ) -> Result<T, Self::Error> {
        let mut iter = zip_iterators!(
            positions.read(),
            momenta.read(),
            physical_forces.read(),
            exchange_forces.read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, momentum, physical_force, exchange_force))| {
                AtomMultiplicativeClassicalEstimator::calculate(
                    self,
                    index,
                    physical_potential_energy.clone(),
                    type_exchange_potential_energy.clone(),
                    group_heat.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        let group_observable = iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
            Ok::<_, <Self as AtomMultiplicativeClassicalEstimator<T, V, M::Error>>::AtomError>(
                accum_observable * atom_observable?,
            )
        })?;
        match multiplier.recv_prod()? {
            Some(other_groups_observable) => Ok(group_observable * other_groups_observable),
            None => Ok(group_observable),
        }
    }
}
