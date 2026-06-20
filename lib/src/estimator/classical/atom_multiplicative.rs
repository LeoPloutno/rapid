//! Traits and types for classical estimators that can be expressed as a product of observables
//! that depend only on a single atom.

use super::{
    ClassicalEstimatorReceiver, ClassicalEstimatorSender, GroupInTypeInImageInSystem,
    MinimalClassicalEstimatorSender,
};
use crate::{
    core::{
        Scheme,
        error::EmptyError,
        stat::{Bosonic, Distinguishable},
        sync_ops::{SyncAddReceiver, SyncAddSender, SyncMulReceiver, SyncMulSender},
    },
    potential::{
        exchange::{ExchangePotential, quadratic::QuadraticExpansionExchangePotential},
        physical::PhysicalPotential,
    },
    zip_items, zip_iterators,
};
use std::ops::Mul;

/// A wrapper for implementors of the `AtomMultiplicativeClassicalEstimator...` traits.
pub struct MultiplicativeClassicalEstimator<E: ?Sized>(pub(crate) E);

impl<E> MultiplicativeClassicalEstimator<E> {
    /// Wraps the provided value with `MultiplicativeClassicalEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

/// A wrapper for implementors of the [`AtomMultiplicativeMinimalClassicalEstimatorSender`] trait.
pub struct MultiplicativeMinimalClassicalEstimator<E: ?Sized>(pub(crate) E);

impl<E> MultiplicativeMinimalClassicalEstimator<E> {
    /// Wraps the provided value with `MultiplicativeMinimalClassicalEstimator`.
    pub const fn new(value: E) -> Self {
        Self(value)
    }
}

/// A trait for receivers of classical estimators that can be expressed
/// as a product of observables that depend only on a singe atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeClassicalEstimator<E>`]
/// atomatically implements [`ClassicalEstimatorReceiver`].
pub trait AtomMultiplicativeClassicalEstimatorReceiver<T, V, Multiplier>
where
    Multiplier: SyncMulReceiver<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`MultiplicativeClassicalEstimator<Self>`] produce.
    type Output;
    /// The type of error [`MultiplicativeClassicalEstimator<Self>`] returns.
    type Error: From<Multiplier::Error> + From<EmptyError>;
}

/// A trait for senders of classical estimators that can be expressed
/// as a product of observables that depend only on a singe atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeClassicalEstimator<E>`]
/// atomatically implements [`ClassicalEstimatorReceiver`].
pub trait AtomMultiplicativeClassicalEstimatorSender<
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
    /// The type of output `Self` and [`MultiplicativeClassicalEstimator<Self>`] return.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`MultiplicativeClassicalEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        physical_potential: &mut Phys,
        exchange_potential: Scheme<&mut Dist, &mut DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

/// A trait for atom-multiplicative estimator senders that do not rely on either
/// the physical nor the exchange potentials.
///
/// For any type `E` that implements this trait, [`MultiplicativeMinimalClassicalEstimator<E>`]
/// atomatically implements [`MinimalClassicalEstimatorSender`].
pub trait AtomMultiplicativeMinimalClassicalEstimatorSender<T, V, Multiplier>
where
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`MultiplicativeMinimalClassicalEstimator<Self>`] return.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`MultiplicativeMinimalClassicalEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

impl<T, V, Multiplier, E> AtomMultiplicativeClassicalEstimatorReceiver<T, V, Multiplier>
    for MultiplicativeClassicalEstimator<E>
where
    Multiplier: SyncMulReceiver<E::Output> + ?Sized,
    E: AtomMultiplicativeClassicalEstimatorReceiver<T, V, Multiplier> + ?Sized,
{
    type Output = E::Output;
    type Error = E::Error;
}

impl<T, V, Adder, Multiplier, E> ClassicalEstimatorReceiver<T, V, Adder, Multiplier>
    for MultiplicativeClassicalEstimator<E>
where
    Adder: SyncAddReceiver<
            <Self as AtomMultiplicativeClassicalEstimatorReceiver<T, V, Multiplier>>::Output,
        > + ?Sized,
    Multiplier: SyncMulReceiver<
            <Self as AtomMultiplicativeClassicalEstimatorReceiver<T, V, Multiplier>>::Output,
        > + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeClassicalEstimatorReceiver<T, V, Multiplier>,
{
    type Output = <Self as AtomMultiplicativeClassicalEstimatorReceiver<T, V, Multiplier>>::Output;
    type Error = <Self as AtomMultiplicativeClassicalEstimatorReceiver<T, V, Multiplier>>::Error;

    #[inline(always)]
    fn calculate(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
    ) -> Result<Self::Output, Self::Error> {
        Ok(multiplier.receive_product()?.ok_or(EmptyError)?)
    }
}

impl<T, V, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad, E>
    AtomMultiplicativeClassicalEstimatorSender<
        T,
        V,
        Multiplier,
        Phys,
        Dist,
        DistQuad,
        Boson,
        BosonQuad,
    > for MultiplicativeClassicalEstimator<E>
where
    Multiplier: SyncMulSender<E::Output> + ?Sized,
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: AtomMultiplicativeClassicalEstimatorSender<
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
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
            atom_index,
            physical_potential,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            group_heat,
            group_kinetic_energy,
            position,
            momentum,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad, E>
    ClassicalEstimatorSender<T, V, Adder, Multiplier, Phys, Dist, DistQuad, Boson, BosonQuad>
    for MultiplicativeClassicalEstimator<E>
where
    Adder: SyncAddSender<
            <Self as AtomMultiplicativeClassicalEstimatorSender<
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
            <Self as AtomMultiplicativeClassicalEstimatorSender<
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
    Self: AtomMultiplicativeClassicalEstimatorSender<
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
    type Output = <Self as AtomMultiplicativeClassicalEstimatorSender<
        T,
        V,
        Multiplier,
        Phys,
        Dist,
        DistQuad,
        Boson,
        BosonQuad,
    >>::Output;
    type Error = <Self as AtomMultiplicativeClassicalEstimatorSender<
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
        group_heat: T,
        group_kinetic_energy: T,
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
                AtomMultiplicativeClassicalEstimatorSender::calculate(
                    self,
                    index,
                    physical_potential,
                    exchange_potential.as_deref_mut(),
                    group_physical_potential_energy,
                    group_exchange_potential_energy,
                    group_heat,
                    group_kinetic_energy,
                    position,
                    momentum,
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
                    <Self as AtomMultiplicativeClassicalEstimatorSender<
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
        group_heat: T,
        group_kinetic_energy: T,
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
                AtomMultiplicativeClassicalEstimatorSender::calculate(
                    self,
                    index,
                    physical_potential,
                    exchange_potential.as_deref_mut(),
                    group_physical_potential_energy,
                    group_exchange_potential_energy,
                    group_heat,
                    group_kinetic_energy,
                    position,
                    momentum,
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
                    <Self as AtomMultiplicativeClassicalEstimatorSender<
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

impl<T, V, Multiplier, E> AtomMultiplicativeMinimalClassicalEstimatorSender<T, V, Multiplier>
    for MultiplicativeClassicalEstimator<E>
where
    Multiplier: SyncMulSender<E::Output> + ?Sized,
    E: AtomMultiplicativeMinimalClassicalEstimatorSender<T, V, Multiplier> + ?Sized,
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
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
            atom_index,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            group_heat,
            group_kinetic_energy,
            position,
            momentum,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, Multiplier, E> MinimalClassicalEstimatorSender<T, V, Adder, Multiplier>
    for MultiplicativeClassicalEstimator<E>
where
    Adder: SyncAddSender<
            <Self as AtomMultiplicativeMinimalClassicalEstimatorSender<T, V, Multiplier>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as AtomMultiplicativeMinimalClassicalEstimatorSender<T, V, Multiplier>>::Output,
        > + ?Sized,
    E: ?Sized,
    Self: AtomMultiplicativeMinimalClassicalEstimatorSender<T, V, Multiplier>,
{
    type Output =
        <Self as AtomMultiplicativeMinimalClassicalEstimatorSender<T, V, Multiplier>>::Output;
    type Error =
        <Self as AtomMultiplicativeMinimalClassicalEstimatorSender<T, V, Multiplier>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        exchange_potential_is_cyclic: bool,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
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
                AtomMultiplicativeMinimalClassicalEstimatorSender::calculate(
                    self,
                    index,
                    group_physical_potential_energy,
                    group_exchange_potential_energy,
                    group_heat,
                    group_kinetic_energy,
                    position,
                    momentum,
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
                        <Self as AtomMultiplicativeMinimalClassicalEstimatorSender<
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
        group_heat: T,
        group_kinetic_energy: T,
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
                AtomMultiplicativeMinimalClassicalEstimatorSender::calculate(
                    self,
                    index,
                    group_physical_potential_energy,
                    group_exchange_potential_energy,
                    group_heat,
                    group_kinetic_energy,
                    position,
                    momentum,
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
                        <Self as AtomMultiplicativeMinimalClassicalEstimatorSender<
                            T,
                            V,
                            Multiplier,
                        >>::ErrorAtom,
                    >(accum_observable + atom_observable?)
            },
        )?)?)
    }
}
