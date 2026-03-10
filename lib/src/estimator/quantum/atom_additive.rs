//! Traits and types for estimators that can be expressed as a sum of observables
//! that depend only on a single atom.

use std::ops::Add;

use crate::{
    core::{
        Scheme,
        error::EmptyError,
        marker::{InnerIsLeading, InnerIsTrailing},
        stat::{Bosonic, Distinguishable, Stat},
        sync_ops::{SyncAddReciever, SyncAddSender},
    },
    estimator::quantum::{
        AdditiveQuantumEstimator,
        group_additive::{
            InnerGroupAdditiveQuantumEstimator, LeadingGroupAdditiveQuantumEstimator,
            MainGroupAdditiveQuantumEstimator, TrailingGroupAdditiveQuantumEstimator,
        },
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
/// For any type `E` that implements this trait, [`AtomAdditiveQuantumEstimator<E>`]
/// atomatically implements [`MainQuantumEstimator`].
///
/// [`MainQuantumEstimator`]: super::MainQuantumEstimator
pub trait MainAtomAdditiveQuantumEstimator<T, V, Adder>
where
    Adder: SyncAddReciever<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`AtomAdditiveQuantumEstimator<Self>`] produce.
    type Output;
    /// The type of error [`AtomAdditiveQuantumEstimator<Self>`] returns.
    type Error: From<Adder::Error> + From<EmptyError>;
}

/// A trait for leading quantum estimators that can be expressed as a sum
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AtomAdditiveQuantumEstimator<E>`]
/// atomatically implements [`LeadingQuantumEstimator`].
///
/// [`LeadingQuantumEstimator`]: super::LeadingQuantumEstimator
pub trait LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AtomAdditiveQuantumEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomAdditiveQuantumEstimator<Self>`] returns.
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
/// For any type `E` that implements this trait, [`AtomAdditiveQuantumEstimator<E>`]
/// atomatically implements [`InnerQuantumEstimator`].
///
/// [`InnerQuantumEstimator`]: super::InnerQuantumEstimator
pub trait InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AtomAdditiveQuantumEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomAdditiveQuantumEstimator<Self>`] returns.
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
/// For any type `E` that implements this trait, [`AtomAdditiveQuantumEstimator<E>`]
/// atomatically implements [`TrailingQuantumEstimator`].
///
/// [`TrailingQuantumEstimator`]: super::TrailingQuantumEstimator
pub trait TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AtomAdditiveQuantumEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomAdditiveQuantumEstimator<Self>`] returns.
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

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for U
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
    U: InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + InnerIsLeading + ?Sized,
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
        physical_potential_energy: T,
        exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomAdditiveQuantumEstimator::calculate(
            self,
            atom_index,
            exchange_potential,
            physical_potential_energy,
            exchange_potential_energy,
            position,
            physical_force,
            exchange_force,
        )
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for U
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
    U: InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + InnerIsTrailing + ?Sized,
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

impl<T, V, Adder, U> MainAtomAdditiveQuantumEstimator<T, V, Adder> for AdditiveQuantumEstimator<U>
where
    Adder: SyncAddReciever<U::Output> + ?Sized,
    U: MainAtomAdditiveQuantumEstimator<T, V, Adder> + ?Sized,
{
    type Output = U::Output;
    type Error = U::Error;
}

impl<T, V, Adder, U> MainGroupAdditiveQuantumEstimator<T, V, Adder> for AdditiveQuantumEstimator<U>
where
    Adder: SyncAddReciever<<Self as MainAtomAdditiveQuantumEstimator<T, V, Adder>>::Output> + ?Sized,
    U: ?Sized,
    Self: MainAtomAdditiveQuantumEstimator<T, V, Adder>,
{
    type Output = <Self as MainAtomAdditiveQuantumEstimator<T, V, Adder>>::Output;
    type Error = <Self as MainAtomAdditiveQuantumEstimator<T, V, Adder>>::Error;
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<U::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = U::Output;
    type ErrorAtom = U::ErrorAtom;
    type ErrorSystem = U::ErrorSystem;

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

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: ?Sized,
    Self: LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorGroup =
        <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;
    type ErrorSystem =
        <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_positions: &[V],
        group_physical_forces: &[V],
        group_exchange_forces: &[V],
    ) -> Result<Self::Output, Self::ErrorGroup> {
        let mut iter = zip_iterators!(group_positions, group_physical_forces, group_exchange_forces)
            .enumerate()
            .map(|(index, zip_items!(position, physical_force, exchange_force))| {
                LeadingAtomAdditiveQuantumEstimator::calculate(
                    self,
                    index,
                    exchange_potential,
                    group_physical_potential_energy.clone(),
                    group_exchange_potential_energy.clone(),
                    position,
                    physical_force,
                    exchange_force,
                )
            });
        let first_atom_observable = iter.next().ok_or(EmptyError)??;

        Ok(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as LeadingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
            })?,
        )
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<U::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = U::Output;
    type ErrorAtom = U::ErrorAtom;
    type ErrorSystem = U::ErrorSystem;

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

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: ?Sized,
    Self: InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorGroup =
        <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;
    type ErrorSystem =
        <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_positions: &[V],
        group_physical_forces: &[V],
        group_exchange_forces: &[V],
    ) -> Result<Self::Output, Self::ErrorGroup> {
        let mut iter = zip_iterators!(group_positions, group_physical_forces, group_exchange_forces)
            .enumerate()
            .map(|(index, zip_items!(position, physical_force, exchange_force))| {
                InnerAtomAdditiveQuantumEstimator::calculate(
                    self,
                    index,
                    exchange_potential,
                    group_physical_potential_energy.clone(),
                    group_exchange_potential_energy.clone(),
                    position,
                    physical_force,
                    exchange_force,
                )
            });
        let first_atom_observable = iter.next().ok_or(EmptyError)??;

        Ok(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as InnerAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
            })?,
        )
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<U::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = U::Output;
    type ErrorAtom = U::ErrorAtom;
    type ErrorSystem = U::ErrorSystem;

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

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveQuantumEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: ?Sized,
    Self: TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorGroup =
        <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;
    type ErrorSystem =
        <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_positions: &[V],
        group_physical_forces: &[V],
        group_exchange_forces: &[V],
    ) -> Result<Self::Output, Self::ErrorGroup> {
        let mut iter = zip_iterators!(group_positions, group_physical_forces, group_exchange_forces)
            .enumerate()
            .map(|(index, zip_items!(position, physical_force, exchange_force))| {
                TrailingAtomAdditiveQuantumEstimator::calculate(
                    self,
                    index,
                    exchange_potential,
                    group_physical_potential_energy.clone(),
                    group_exchange_potential_energy.clone(),
                    position,
                    physical_force,
                    exchange_force,
                )
            });
        let first_atom_observable = iter.next().ok_or(EmptyError)??;

        Ok(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as TrailingAtomAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
            })?,
        )
    }
}
