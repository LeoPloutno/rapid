//! Traits and types for estimators that can be expressed as a sum of observables
//! that depend only on a single atom.

use super::{
    EstimatorImages, GroupInTypeInImageInSystem, MinimalQuantumEstimatorSender,
    QuantumEstimatorReciever, QuantumEstimatorSender,
};
use crate::{
    core::{
        Scheme,
        error::EmptyError,
        stat::{Bosonic, Distinguishable},
        sync_ops::{SyncAddReciever, SyncAddSender, SyncMulReciever, SyncMulSender},
    },
    potential::{
        exchange::{ExchangePotential, quadratic::QuadraticExpansionExchangePotential},
        physical::PhysicalPotential,
    },
    zip_items, zip_iterators,
};
use std::ops::Add;

/// A wrapper for implementors of the `AtomAdditiveQuantumEstimator...` traits.
pub struct AdditiveQuantumEstimator<E: ?Sized>(pub(crate) E);

impl<E> AdditiveQuantumEstimator<E> {
    /// Wraps the provided value with `AdditiveQuantumEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

/// A wrapper for implementors of the [`AtomAdditiveMinimalQuantumEstimatorSender`] trait.
pub struct AdditiveMinimalQuantumEstimator<E: ?Sized>(pub(crate) E);

impl<E> AdditiveMinimalQuantumEstimator<E> {
    /// Wraps the provided value with `AdditiveMinimalQuantumEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

/// A trait for recievers of quantum estimators that can be expressed
/// as a sum of observables that depend only on a singe atom.
///
/// For any type `E` that implements this trait, [`AdditiveQuantumEstimator<E>`]
/// atomatically implements [`QuantumEstimatorReciever`].
pub trait AtomAdditiveQuantumEstimatorReciever<T, V, Adder>
where
    Adder: SyncAddReciever<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`AdditiveQuantumEstimator<Self>`] produce.
    type Output;
    /// The type of error [`AdditiveQuantumEstimator<Self>`] returns.
    type Error: From<Adder::Error> + From<EmptyError>;
}

/// A trait for senders of quantum estimators that can be expressed
/// as a sum of observables that depend only on a singe atom.
///
/// For any type `E` that implements this trait, [`AdditiveQuantumEstimator<E>`]
/// atomatically implements [`QuantumEstimatorReciever`].
pub trait AtomAdditiveQuantumEstimatorSender<T, V, Adder, Phys, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AdditiveQuantumEstimator<Self>`] return.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Adder::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Dist, &mut DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

/// A trait for atom-additive estimator senders that do not rely on either
/// the physical nor the exchange potentials.
///
/// For any type `E` that implements this trait, [`AdditiveMinimalQuantumEstimator<E>`]
/// atomatically implements [`MinimalQuantumEstimatorSender`].
pub trait AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder>
where
    Adder: SyncAddSender<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`AdditiveMinimalQuantumEstimator<Self>`] return.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveMinimalQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Adder::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

impl<T, V, Adder, E> AtomAdditiveQuantumEstimatorReciever<T, V, Adder>
    for AdditiveQuantumEstimator<E>
where
    Adder: SyncAddReciever<E::Output> + ?Sized,
    E: AtomAdditiveQuantumEstimatorReciever<T, V, Adder> + ?Sized,
{
    type Output = E::Output;
    type Error = E::Error;
}

impl<T, V, Adder, Multiplier, E> QuantumEstimatorReciever<T, V, Adder, Multiplier>
    for AdditiveQuantumEstimator<E>
where
    Adder: SyncAddReciever<<Self as AtomAdditiveQuantumEstimatorReciever<T, V, Adder>>::Output>
        + ?Sized,
    Multiplier: SyncMulReciever<<Self as AtomAdditiveQuantumEstimatorReciever<T, V, Adder>>::Output>
        + ?Sized,
    E: ?Sized,
    Self: AtomAdditiveQuantumEstimatorReciever<T, V, Adder>,
{
    type Output = <Self as AtomAdditiveQuantumEstimatorReciever<T, V, Adder>>::Output;
    type Error = <Self as AtomAdditiveQuantumEstimatorReciever<T, V, Adder>>::Error;

    #[inline(always)]
    fn calculate(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
    ) -> Result<Self::Output, Self::Error> {
        Ok(adder.recieve_sum()?.ok_or(EmptyError)?)
    }
}

impl<T, V, Adder, Phys, Dist, DistQuad, Boson, BosonQuad, E>
    AtomAdditiveQuantumEstimatorSender<T, V, Adder, Phys, Dist, DistQuad, Boson, BosonQuad>
    for AdditiveQuantumEstimator<E>
where
    Adder: SyncAddSender<E::Output> + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: AtomAdditiveQuantumEstimatorSender<T, V, Adder, Phys, Dist, DistQuad, Boson, BosonQuad>
        + ?Sized,
{
    type Output = E::Output;
    type ErrorAtom = E::ErrorAtom;
    type ErrorSystem = E::ErrorSystem;

    #[inline(always)]
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Dist, &mut DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
            atom_index,
            physical_potential,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad, E>
    QuantumEstimatorSender<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad>
    for AdditiveQuantumEstimator<E>
where
    Adder: SyncAddSender<
            <Self as AtomAdditiveQuantumEstimatorSender<
                T,
                V,
                Adder,
                Phys,
                Dist,
                DistQuad,
                Boson,
                BosonQuad,
            >>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as AtomAdditiveQuantumEstimatorSender<
                T,
                V,
                Adder,
                Phys,
                Dist,
                DistQuad,
                Boson,
                BosonQuad,
            >>::Output,
        > + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: AtomAdditiveQuantumEstimatorSender<T, V, Adder, Phys, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = <Self as AtomAdditiveQuantumEstimatorSender<
        T,
        V,
        Adder,
        Phys,
        Dist,
        DistQuad,
        Boson,
        BosonQuad,
    >>::Output;
    type Error = <Self as AtomAdditiveQuantumEstimatorSender<
        T,
        V,
        Adder,
        Phys,
        Dist,
        DistQuad,
        Boson,
        BosonQuad,
    >>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Dist, &mut DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            positions.read(),
            physical_forces.read(),
            exchange_forces.read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, physical_force, exchange_force))| {
                AtomAdditiveQuantumEstimatorSender::calculate(
                    self,
                    index,
                    physical_potential,
                    exchange_potential.as_deref_mut(),
                    group_physical_potential_energy,
                    group_exchange_potential_energy,
                    position,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        Ok(adder.send(iter.try_fold(
            first_atom_observable,
            |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as AtomAdditiveQuantumEstimatorSender<
                        T,
                        V,
                        Adder,
                        Phys,
                        Dist,
                        DistQuad,
                        Boson,
                        BosonQuad,
                    >>::ErrorAtom,
                >(accum_observable + atom_observable?)
            },
        )?)?)
    }

    fn calculate_bosonic(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Boson, &mut BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            positions.read(),
            physical_forces.read(),
            exchange_forces.read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, physical_force, exchange_force))| {
                AtomAdditiveQuantumEstimatorSender::calculate(
                    self,
                    index,
                    physical_potential,
                    exchange_potential.as_deref_mut(),
                    group_physical_potential_energy,
                    group_exchange_potential_energy,
                    position,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        Ok(adder.send(iter.try_fold(
            first_atom_observable,
            |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as AtomAdditiveQuantumEstimatorSender<
                        T,
                        V,
                        Adder,
                        Phys,
                        Dist,
                        DistQuad,
                        Boson,
                        BosonQuad,
                    >>::ErrorAtom,
                >(accum_observable + atom_observable?)
            },
        )?)?)
    }
}

impl<T, V, Adder, E> AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder>
    for AdditiveQuantumEstimator<E>
where
    Adder: SyncAddSender<E::Output> + ?Sized,
    E: AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder> + ?Sized,
{
    type Output = E::Output;
    type ErrorAtom = E::ErrorAtom;
    type ErrorSystem = E::ErrorSystem;

    #[inline(always)]
    fn calculate(
        &mut self,
        atom_index: usize,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
            atom_index,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, Multiplier, E> MinimalQuantumEstimatorSender<T, V, Adder, Multiplier>
    for AdditiveQuantumEstimator<E>
where
    Adder: SyncAddSender<<Self as AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder>>::Output>
        + ?Sized,
    Multiplier: SyncMulSender<<Self as AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder>>::Output>
        + ?Sized,
    E: ?Sized,
    Self: AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder>,
{
    type Output = <Self as AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder>>::Output;
    type Error = <Self as AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        exchange_potential_is_cyclic: bool,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            positions.read(),
            physical_forces.read(),
            exchange_forces.read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, physical_force, exchange_force))| {
                AtomAdditiveMinimalQuantumEstimatorSender::calculate(
                    self,
                    index,
                    group_physical_potential_energy,
                    group_exchange_potential_energy,
                    position,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        Ok(adder.send(
            iter.try_fold(
                first_atom_observable,
                |accum_observable, atom_observable| {
                    Ok::<
                        _,
                        <Self as AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder>>::ErrorAtom,
                    >(accum_observable + atom_observable?)
                },
            )?,
        )?)
    }

    fn calculate_bosonic(
        &mut self,
        exchange_potential_is_cyclic: bool,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        positions: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        physical_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
        exchange_forces: &EstimatorImages<GroupInTypeInImageInSystem<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            positions.read(),
            physical_forces.read(),
            exchange_forces.read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, physical_force, exchange_force))| {
                AtomAdditiveMinimalQuantumEstimatorSender::calculate(
                    self,
                    index,
                    group_physical_potential_energy,
                    group_exchange_potential_energy,
                    position,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        Ok(adder.send(
            iter.try_fold(
                first_atom_observable,
                |accum_observable, atom_observable| {
                    Ok::<
                        _,
                        <Self as AtomAdditiveMinimalQuantumEstimatorSender<T, V, Adder>>::ErrorAtom,
                    >(accum_observable + atom_observable?)
                },
            )?,
        )?)
    }
}
