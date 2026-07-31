//! Traits and types for classical estimators that can be expressed as a product of observables
//! that depend only on a single atom.

use super::ClassicalEstimator;
use crate::{
    core::{
        GroupInTypeInImageInSystem, MeaningfulOutput,
        error::EmptyError,
        sync_ops::{SyncMulReceiver, SyncMulSender},
    },
    zip_items, zip_iterators,
};
use std::{
    ops::Mul,
    sync::{Barrier, RwLock},
};

/// A trait for classical estimators that can be expressed as a product
/// of estimators that each depend only on a single atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeValueClassicalEstimator<E>`]
/// automatically implements [`ClassicalEstimator`].
pub trait AtomMultiplicativeClassicalEstimator<T: Clone, V> {
    /// The type of output `Self` and [`MultiplicativeValueClassicalEstimator<Self>`] return.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type AtomError;
    /// The type of error [`MultiplicativeValueClassicalEstimator<Self>`] returns.
    type SystemError: From<Self::AtomError> + From<EmptyError>;

    /// Calculates the contribution of an atom to the contribution
    /// of the image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        group_kinetic_energy: T,
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

impl<T, V, E> AtomMultiplicativeClassicalEstimator<T, V>
    for MultiplicativeValueClassicalEstimator<E>
where
    T: Clone + Mul<Output = T>,
    E: AtomMultiplicativeClassicalEstimator<T, V, Output = T> + ?Sized,
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
        group_kinetic_energy: T,
        group_heat: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError> {
        self.0.calculate(
            atom_index,
            physical_potential_energy,
            exchange_potential_energy,
            group_kinetic_energy,
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
    T: Clone + Mul<Output = T>,
    A: ?Sized,
    M: SyncMulSender<T> + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeClassicalEstimator<T, V, Output = T>,
    <Self as AtomMultiplicativeClassicalEstimator<T, V>>::SystemError: From<M::Error>,
{
    type Output = T;
    type Error = <Self as AtomMultiplicativeClassicalEstimator<T, V>>::SystemError;

    fn calculate(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        group_kinetic_energy: T,
        group_heat: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
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
                    exchange_potential_energy.clone(),
                    group_kinetic_energy.clone(),
                    group_heat.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        let group_observable = iter.try_fold(
            first_atom_observable,
            |accum_observable, atom_observable| {
                Ok::<_, <Self as AtomMultiplicativeClassicalEstimator<T, V>>::AtomError>(
                    accum_observable * atom_observable?,
                )
            },
        )?;
        multiplier.send(group_observable)?;
        Ok(())
    }
}

impl<T, V, A, M, E> ClassicalEstimator<T, V, A, M, T> for MultiplicativeValueClassicalEstimator<E>
where
    T: Clone + Mul<Output = T> + MeaningfulOutput,
    A: ?Sized,
    M: SyncMulReceiver<T> + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeClassicalEstimator<T, V, Output = T>,
    <Self as AtomMultiplicativeClassicalEstimator<T, V>>::SystemError: From<M::Error>,
{
    type Output = T;
    type Error = <Self as AtomMultiplicativeClassicalEstimator<T, V>>::SystemError;

    fn calculate(
        &mut self,
        _barrier: &Barrier,
        _shared_value: &RwLock<T>,
        _adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        group_kinetic_energy: T,
        group_heat: T,
        positions: &GroupInTypeInImageInSystem<V>,
        momenta: &GroupInTypeInImageInSystem<V>,
        physical_forces: &GroupInTypeInImageInSystem<V>,
        exchange_forces: &GroupInTypeInImageInSystem<V>,
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
                    exchange_potential_energy.clone(),
                    group_kinetic_energy.clone(),
                    group_heat.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        let group_observable = iter.try_fold(
            first_atom_observable,
            |accum_observable, atom_observable| {
                Ok::<_, <Self as AtomMultiplicativeClassicalEstimator<T, V>>::AtomError>(
                    accum_observable * atom_observable?,
                )
            },
        )?;
        match multiplier.recv_prod()? {
            Some(other_groups_observable) => Ok(group_observable * other_groups_observable),
            None => Ok(group_observable),
        }
    }
}
