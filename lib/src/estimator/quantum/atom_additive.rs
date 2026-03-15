//! Traits and types for estimators that can be expressed as a sum of observables
//! that depend only on a single atom.

use std::ops::Add;

use arc_rw_lock::ElementRwLock;

use crate::{
    ImageHandle,
    core::{
        Additive as AdditiveQuantumEstimator, Scheme,
        error::EmptyError,
        marker::{InnerIsLeading, InnerIsTrailing},
        stat::{Bosonic, Distinguishable, Stat},
        sync_ops::{SyncAddReciever, SyncAddSender, SyncMulReciever, SyncMulSender},
    },
    estimator::quantum::{
        InnerQuantumEstimator, LeadingQuantumEstimator, MainQuantumEstimator, TrailingQuantumEstimator,
    },
    potential::exchange::{
        InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential,
        quadratic::{
            InnerQuadraticExpansionExchangePotential, LeadingQuadraticExpansionExchangePotential,
            TrailingQuadraticExpansionExchangePotential,
        },
    },
    zip_items, zip_iterators,
};

/// A trait for main quantum estimators that can be expressed as a sum
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveQuantumEstimator<E>`]
/// atomatically implements [`MainQuantumEstimator`].
pub trait MainAtomAdditiveQuantumEstimator<T, V, Adder>
where
    Adder: SyncAddReciever<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`AdditiveQuantumEstimator<Self>`] produce.
    type Output;
    /// The type of error [`AdditiveQuantumEstimator<Self>`] returns.
    type Error: From<Adder::Error> + From<EmptyError>;
}

/// A trait for leading quantum estimators that can be expressed as a sum
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveQuantumEstimator<E>`]
/// atomatically implements [`LeadingQuantumEstimator`].
pub trait LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AdditiveQuantumEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Adder::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in the first image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

/// A trait for inner quantum estimators that can be expressed as a sum
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveQuantumEstimator<E>`]
/// atomatically implements [`InnerQuantumEstimator`].
pub trait InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AdditiveQuantumEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Adder::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in this image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

/// A trait for trailing quantum estimators that can be expressed as a sum
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveQuantumEstimator<E>`]
/// atomatically implements [`TrailingQuantumEstimator`].
pub trait TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AdditiveQuantumEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Adder::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in the last image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for E
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: InnerExchangePotential<T, V> + LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V>
        + Distinguishable
        + ?Sized,
    Boson: InnerExchangePotential<T, V> + LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V>
        + Bosonic
        + ?Sized,
    E: InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + InnerIsLeading + ?Sized,
{
    type Output = <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomAdditiveQuantumEstimator::calculate(
            self,
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for E
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: InnerExchangePotential<T, V> + TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V>
        + Distinguishable
        + ?Sized,
    Boson: InnerExchangePotential<T, V> + TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V>
        + Bosonic
        + ?Sized,
    E: InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + InnerIsTrailing + ?Sized,
{
    type Output = <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomAdditiveQuantumEstimator::calculate(
            self,
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, E> MainAtomAdditiveQuantumEstimator<T, V, Adder> for AdditiveQuantumEstimator<E>
where
    Adder: SyncAddReciever<E::Output> + ?Sized,
    E: MainAtomAdditiveQuantumEstimator<T, V, Adder> + ?Sized,
{
    type Output = E::Output;
    type Error = E::Error;
}

impl<T, V, Adder, Multiplier, E> MainQuantumEstimator<T, V, Adder, Multiplier> for AdditiveQuantumEstimator<E>
where
    Adder: SyncAddReciever<<Self as MainAtomAdditiveQuantumEstimator<T, V, Adder>>::Output> + ?Sized,
    Multiplier: SyncMulReciever<<Self as MainAtomAdditiveQuantumEstimator<T, V, Adder>>::Output> + ?Sized,
    E: ?Sized,
    Self: MainAtomAdditiveQuantumEstimator<T, V, Adder>,
{
    type Output = <Self as MainAtomAdditiveQuantumEstimator<T, V, Adder>>::Output;
    type Error = <Self as MainAtomAdditiveQuantumEstimator<T, V, Adder>>::Error;

    fn calculate(&mut self, adder: &mut Adder, _multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error> {
        Ok(adder.recieve_sum()?.ok_or(EmptyError)?)
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<E::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = E::Output;
    type ErrorAtom = E::ErrorAtom;
    type ErrorSystem = E::ErrorSystem;

    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    LeadingQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read()
        )
        .enumerate()
        .map(|(index, zip_items!(position, physical_force, exchange_force))| {
            LeadingAtomAdditiveQuantumEstimator::calculate(
                self,
                index,
                exchange_potential.clone(),
                group_physical_potential_energy.clone(),
                group_exchange_potential_energy.clone(),
                position,
                physical_force,
                exchange_force,
            )
        });
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        adder.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
            })?,
        )?;
        Ok(())
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<E::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = E::Output;
    type ErrorAtom = E::ErrorAtom;
    type ErrorSystem = E::ErrorSystem;

    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    InnerQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read()
        )
        .enumerate()
        .map(|(index, zip_items!(position, physical_force, exchange_force))| {
            InnerAtomAdditiveQuantumEstimator::calculate(
                self,
                index,
                exchange_potential.clone(),
                group_physical_potential_energy.clone(),
                group_exchange_potential_energy.clone(),
                position,
                physical_force,
                exchange_force,
            )
        });
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        adder.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
            })?,
        )?;
        Ok(())
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<E::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = E::Output;
    type ErrorAtom = E::ErrorAtom;
    type ErrorSystem = E::ErrorSystem;

    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    TrailingQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read()
        )
        .enumerate()
        .map(|(index, zip_items!(position, physical_force, exchange_force))| {
            TrailingAtomAdditiveQuantumEstimator::calculate(
                self,
                index,
                exchange_potential.clone(),
                group_physical_potential_energy.clone(),
                group_exchange_potential_energy.clone(),
                position,
                physical_force,
                exchange_force,
            )
        });
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        adder.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
            })?,
        )?;
        Ok(())
    }
}
