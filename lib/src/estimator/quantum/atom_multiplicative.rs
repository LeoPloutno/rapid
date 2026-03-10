//! Traits and types for estimators that can be expressed as a product of observables
//! that depend only on a single atom.

use std::ops::Mul;

use arc_rw_lock::ElementRwLock;

use crate::{
    ImageHandle,
    core::{
        Scheme,
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

/// A wrapper that automatically implements a `QuantumEstimator` trait if
/// the inner type inmplements its `AtomMultiplicative` variant.
pub struct AtomMultiplicativeQuantumEstimator<E: ?Sized>(pub E);

/// A trait for main quantum estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AtomMultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`MainQuantumEstimator`].
///
/// [`MainQuantumEstimator`]: super::MainQuantumEstimator
pub trait MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>
where
    Multiplier: SyncMulReciever<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`AtomMultiplicativeQuantumEstimator<Self>`] produce.
    type Output;
    /// The type of error [`AtomMultiplicativeQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Multiplier::Error> + From<EmptyError>;
}

/// A trait for leading quantum estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AtomMultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`LeadingQuantumEstimator`].
///
/// [`LeadingQuantumEstimator`]: super::LeadingQuantumEstimator
pub trait LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AtomMultiplicativeQuantumEstimator<Self>`] produce.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomMultiplicativeQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

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

/// A trait for inner quantum estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AtomMultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`InnerQuantumEstimator`].
///
/// [`InnerQuantumEstimator`]: super::InnerQuantumEstimator
pub trait InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AtomMultiplicativeQuantumEstimator<Self>`] produce.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomMultiplicativeQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

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

/// A trait for trailing quantum estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AtomMultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`TrailingQuantumEstimator`].
///
/// [`TrailingQuantumEstimator`]: super::TrailingQuantumEstimator
pub trait TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AtomMultiplicativeQuantumEstimator<Self>`] produce.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomMultiplicativeQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

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

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> for U
where
    T: Clone,
    Multiplier: SyncMulSender<
            <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
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
    U: InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> + InnerIsLeading + ?Sized,
{
    type Output =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

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
        InnerAtomMultiplicativeQuantumEstimator::calculate(
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

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> for U
where
    T: Clone,
    Multiplier: SyncMulSender<
            <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
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
    U: InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
        + InnerIsTrailing
        + ?Sized,
{
    type Output =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

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
        InnerAtomMultiplicativeQuantumEstimator::calculate(
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

impl<T, V, Adder, Multiplier, U> MainQuantumEstimator<T, V, Adder, Multiplier> for AtomMultiplicativeQuantumEstimator<U>
where
    Adder: SyncAddReciever<<U as MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>>::Output> + ?Sized,
    Multiplier: SyncMulReciever<<U as MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>>::Output> + ?Sized,
    U: MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier> + ?Sized,
{
    type Output = <U as MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>>::Output;
    type Error = <U as MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>>::ErrorSystem;

    fn calculate(&mut self, _adder: &mut Adder, multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error> {
        Ok(multiplier.recieve_prod()?.ok_or(EmptyError)?)
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AtomMultiplicativeQuantumEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<
            <U as LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <U as LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> + ?Sized,
{
    type Output =
        <U as LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <U as LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
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
            LeadingAtomMultiplicativeQuantumEstimator::calculate(
                &mut self.0,
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, U::ErrorAtom>(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    InnerQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AtomMultiplicativeQuantumEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<
            <U as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <U as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> + ?Sized,
{
    type Output =
        <U as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <U as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
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
            InnerAtomMultiplicativeQuantumEstimator::calculate(
                &mut self.0,
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, U::ErrorAtom>(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AtomMultiplicativeQuantumEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<
            <U as TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <U as TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> + ?Sized,
{
    type Output =
        <U as TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <U as TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
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
            TrailingAtomMultiplicativeQuantumEstimator::calculate(
                &mut self.0,
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, U::ErrorAtom>(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }
}
