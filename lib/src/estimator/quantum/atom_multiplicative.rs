//! Traits and types for estimators that can be expressed as a product of observables
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
use std::ops::Mul;

/// A wrapper for implementors of the `AtomMultiplicativeQuantumEstimator...` traits.
pub struct MultiplicativeQuantumEstimator<E: ?Sized>(pub(crate) E);

impl<E> MultiplicativeQuantumEstimator<E> {
    /// Wraps the provided value with `MultiplicativeQuantumEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

/// A wrapper for implementors of the [`AtomMultiplicativeMinimalQuantumEstimatorSender`] trait.
pub struct MultiplicativeMinimalQuantumEstimator<E: ?Sized>(pub(crate) E);

impl<E> MultiplicativeMinimalQuantumEstimator<E> {
    /// Wraps the provided value with `MultiplicativeMinimalQuantumEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

/// A trait for recievers of quantum estimators that can be expressed
/// as a product of observables that depend only on a singe atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`QuantumEstimatorReciever`].
pub trait AtomMultiplicativeQuantumEstimatorReciever<T, V, Multiplier>
where
    Multiplier: SyncMulReciever<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`MultiplicativeQuantumEstimator<Self>`] produce.
    type Output;
    /// The type of error [`MultiplicativeQuantumEstimator<Self>`] returns.
    type Error: From<Multiplier::Error> + From<EmptyError>;
}

/// A trait for senders of quantum estimators that can be expressed
/// as a product of observables that depend only on a singe atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`QuantumEstimatorReciever`].
pub trait AtomMultiplicativeQuantumEstimatorSender<
    T,
    V,
    Multiplier,
    Phys,
    Dist,
    DistQuad,
    Boson,
    BosonQuad,
> where
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`MultiplicativeQuantumEstimator<Self>`] return.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`MultiplicativeQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

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

/// A trait for atom-multiplicative estimator senders that do not rely on either
/// the physical nor the exchange potentials.
///
/// For any type `E` that implements this trait, [`MultiplicativeMinimalQuantumEstimator<E>`]
/// atomatically implements [`MinimalQuantumEstimatorSender`].
pub trait AtomMultiplicativeMinimalQuantumEstimatorSender<T, V, Multiplier>
where
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`MultiplicativeMinimalQuantumEstimator<Self>`] return.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`MultiplicativeMinimalQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

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

impl<T, V, Multiplier, E> AtomMultiplicativeQuantumEstimatorReciever<T, V, Multiplier>
    for MultiplicativeQuantumEstimator<E>
where
    Multiplier: SyncMulReciever<E::Output> + ?Sized,
    E: AtomMultiplicativeQuantumEstimatorReciever<T, V, Multiplier> + ?Sized,
{
    type Output = E::Output;
    type Error = E::Error;
}

impl<T, V, Adder, Multiplier, E> QuantumEstimatorReciever<T, V, Adder, Multiplier>
    for MultiplicativeQuantumEstimator<E>
where
    Adder: SyncAddReciever<
            <Self as AtomMultiplicativeQuantumEstimatorReciever<T, V, Multiplier>>::Output,
        > + ?Sized,
    Multiplier: SyncMulReciever<
            <Self as AtomMultiplicativeQuantumEstimatorReciever<T, V, Multiplier>>::Output,
        > + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeQuantumEstimatorReciever<T, V, Multiplier>,
{
    type Output = <Self as AtomMultiplicativeQuantumEstimatorReciever<T, V, Multiplier>>::Output;
    type Error = <Self as AtomMultiplicativeQuantumEstimatorReciever<T, V, Multiplier>>::Error;

    #[inline(always)]
    fn calculate(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
    ) -> Result<Self::Output, Self::Error> {
        Ok(multiplier.recieve_product()?.ok_or(EmptyError)?)
    }
}

impl<T, V, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad, E>
    AtomMultiplicativeQuantumEstimatorSender<
        T,
        V,
        Multiplier,
        Phys,
        Dist,
        DistQuad,
        Boson,
        BosonQuad,
    > for MultiplicativeQuantumEstimator<E>
where
    Multiplier: SyncMulSender<E::Output> + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: AtomMultiplicativeQuantumEstimatorSender<
            T,
            V,
            Multiplier,
            Phys,
            Dist,
            DistQuad,
            Boson,
            BosonQuad,
        > + ?Sized,
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
    for MultiplicativeQuantumEstimator<E>
where
    Adder: SyncAddSender<
            <Self as AtomMultiplicativeQuantumEstimatorSender<
                T,
                V,
                Multiplier,
                Phys,
                Dist,
                DistQuad,
                Boson,
                BosonQuad,
            >>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as AtomMultiplicativeQuantumEstimatorSender<
                T,
                V,
                Multiplier,
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
    Self: AtomMultiplicativeQuantumEstimatorSender<
            T,
            V,
            Multiplier,
            Phys,
            Dist,
            DistQuad,
            Boson,
            BosonQuad,
        >,
{
    type Output = <Self as AtomMultiplicativeQuantumEstimatorSender<
        T,
        V,
        Multiplier,
        Phys,
        Dist,
        DistQuad,
        Boson,
        BosonQuad,
    >>::Output;
    type Error = <Self as AtomMultiplicativeQuantumEstimatorSender<
        T,
        V,
        Multiplier,
        Phys,
        Dist,
        DistQuad,
        Boson,
        BosonQuad,
    >>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
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
                AtomMultiplicativeQuantumEstimatorSender::calculate(
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
        Ok(multiplier.send(iter.try_fold(
            first_atom_observable,
            |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as AtomMultiplicativeQuantumEstimatorSender<
                        T,
                        V,
                        Multiplier,
                        Phys,
                        Dist,
                        DistQuad,
                        Boson,
                        BosonQuad,
                    >>::ErrorAtom,
                >(accum_observable * atom_observable?)
            },
        )?)?)
    }

    fn calculate_bosonic(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
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
                AtomMultiplicativeQuantumEstimatorSender::calculate(
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
        Ok(multiplier.send(iter.try_fold(
            first_atom_observable,
            |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as AtomMultiplicativeQuantumEstimatorSender<
                        T,
                        V,
                        Multiplier,
                        Phys,
                        Dist,
                        DistQuad,
                        Boson,
                        BosonQuad,
                    >>::ErrorAtom,
                >(accum_observable * atom_observable?)
            },
        )?)?)
    }
}

impl<T, V, Multiplier, E> AtomMultiplicativeMinimalQuantumEstimatorSender<T, V, Multiplier>
    for MultiplicativeQuantumEstimator<E>
where
    Multiplier: SyncMulSender<E::Output> + ?Sized,
    E: AtomMultiplicativeMinimalQuantumEstimatorSender<T, V, Multiplier> + ?Sized,
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
    for MultiplicativeQuantumEstimator<E>
where
    Adder: SyncAddSender<
            <Self as AtomMultiplicativeMinimalQuantumEstimatorSender<T, V, Multiplier>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as AtomMultiplicativeMinimalQuantumEstimatorSender<T, V, Multiplier>>::Output,
        > + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeMinimalQuantumEstimatorSender<T, V, Multiplier>,
{
    type Output =
        <Self as AtomMultiplicativeMinimalQuantumEstimatorSender<T, V, Multiplier>>::Output;
    type Error =
        <Self as AtomMultiplicativeMinimalQuantumEstimatorSender<T, V, Multiplier>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        exchange_potential_is_cyclic: bool,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
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
                AtomMultiplicativeMinimalQuantumEstimatorSender::calculate(
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
        Ok(multiplier.send(iter.try_fold(
            first_atom_observable,
            |accum_observable, atom_observable| {
                Ok::<
                        _,
                        <Self as AtomMultiplicativeMinimalQuantumEstimatorSender<
                            T,
                            V,
                            Multiplier,
                        >>::ErrorAtom,
                    >(accum_observable * atom_observable?)
            },
        )?)?)
    }

    fn calculate_bosonic(
        &mut self,
        exchange_potential_is_cyclic: bool,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
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
                AtomMultiplicativeMinimalQuantumEstimatorSender::calculate(
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
        Ok(multiplier.send(iter.try_fold(
            first_atom_observable,
            |accum_observable, atom_observable| {
                Ok::<
                        _,
                        <Self as AtomMultiplicativeMinimalQuantumEstimatorSender<
                            T,
                            V,
                            Multiplier,
                        >>::ErrorAtom,
                    >(accum_observable + atom_observable?)
            },
        )?)?)
    }
}
