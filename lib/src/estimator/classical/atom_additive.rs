//! Traits and types for estimators that can be expressed as a sum of observables
//! that depend only on a single atom.

use std::ops::Add;

use arc_rw_lock::ElementRwLock;

use crate::{
    ImageHandle,
    core::{
        Additive as AdditiveClassicalEstimator, Scheme,
        error::EmptyError,
        marker::{InnerIsLeading, InnerIsTrailing},
        stat::{Bosonic, Distinguishable},
        sync_ops::{SyncAddReciever, SyncAddSender, SyncMulReciever, SyncMulSender},
    },
    estimator::classical::{
        InnerClassicalEstimator, LeadingClassicalEstimator, MainClassicalEstimator, TrailingClassicalEstimator,
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

/// A trait for main classical estimators that can be expressed as a sum
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveClassicalEstimator<E>`]
/// atomatically implements [`MainClassicalEstimator`].
pub trait MainAtomAdditiveClassicalEstimator<T, V, Adder>
where
    Adder: SyncAddReciever<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`AdditiveClassicalEstimator<Self>`] produce.
    type Output;
    /// The type of error [`AdditiveClassicalEstimator<Self>`] returns.
    type Error: From<Adder::Error> + From<EmptyError>;
}

/// A trait for leading classical estimators that can be expressed as a sum
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveClassicalEstimator<E>`]
/// atomatically implements [`LeadingClassicalEstimator`].
pub trait LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AdditiveClassicalEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveClassicalEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Adder::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in the first image to the observable
    /// given that the whole group has distinguishable statistics.
    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;

    /// Calculates the contribution of this atom in the first image to the observable
    /// given that the whole group has bosonic statistics.
    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
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

/// A trait for inner classical estimators that can be expressed as a sum
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveClassicalEstimator<E>`]
/// atomatically implements [`InnerClassicalEstimator`].
pub trait InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AdditiveClassicalEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveClassicalEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Adder::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in this image to the observable
    /// given that the whole group has distinguishable statistics.
    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;

    /// Calculates the contribution of this atom in this image to the observable
    /// given that the whole group has bosonic statistics.
    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
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

/// A trait for trailing classical estimators that can be expressed as a sum
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AdditiveClassicalEstimator<E>`]
/// atomatically implements [`TrailingClassicalEstimator`].
pub trait TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AdditiveClassicalEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AdditiveClassicalEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Adder::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in the last image to the observable
    /// given that the whole group has distinguishable statistics.
    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;

    /// Calculates the contribution of this atom in the last image to the observable
    /// given that the whole group has bosonic statistics.
    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
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

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for E
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
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
    E: InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + InnerIsLeading + ?Sized,
{
    type Output = <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomAdditiveClassicalEstimator::calculate_distinguishable(
            self,
            atom_index,
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

    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomAdditiveClassicalEstimator::calculate_bosonic(
            self,
            atom_index,
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

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for E
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
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
    E: InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + InnerIsTrailing + ?Sized,
{
    type Output = <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomAdditiveClassicalEstimator::calculate_distinguishable(
            self,
            atom_index,
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

    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomAdditiveClassicalEstimator::calculate_bosonic(
            self,
            atom_index,
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

impl<T, V, Adder, E> MainAtomAdditiveClassicalEstimator<T, V, Adder> for AdditiveClassicalEstimator<E>
where
    Adder: SyncAddReciever<E::Output> + ?Sized,
    E: MainAtomAdditiveClassicalEstimator<T, V, Adder> + ?Sized,
{
    type Output = E::Output;
    type Error = E::Error;
}

impl<T, V, Adder, Multiplier, E> MainClassicalEstimator<T, V, Adder, Multiplier> for AdditiveClassicalEstimator<E>
where
    Adder: SyncAddReciever<<Self as MainAtomAdditiveClassicalEstimator<T, V, Adder>>::Output> + ?Sized,
    Multiplier: SyncMulReciever<<Self as MainAtomAdditiveClassicalEstimator<T, V, Adder>>::Output> + ?Sized,
    E: ?Sized,
    Self: MainAtomAdditiveClassicalEstimator<T, V, Adder>,
{
    type Output = <Self as MainAtomAdditiveClassicalEstimator<T, V, Adder>>::Output;
    type Error = <Self as MainAtomAdditiveClassicalEstimator<T, V, Adder>>::Error;

    fn calculate(&mut self, adder: &mut Adder, _multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error> {
        Ok(adder.recieve_sum()?.ok_or(EmptyError)?)
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
    for AdditiveClassicalEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<E::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = E::Output;
    type ErrorAtom = E::ErrorAtom;
    type ErrorSystem = E::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_distinguishable(
            atom_index,
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

    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_bosonic(
            atom_index,
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

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    LeadingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AdditiveClassicalEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output =
        <Self as LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_momenta: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_momenta.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, momentum, physical_force, exchange_force))| {
                LeadingAtomAdditiveClassicalEstimator::calculate_distinguishable(
                    self,
                    index,
                    exchange_potential.clone(),
                    group_physical_potential_energy.clone(),
                    group_exchange_potential_energy.clone(),
                    group_heat.clone(),
                    group_kinetic_energy.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        adder.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
            })?,
        )?;
        Ok(())
    }

    fn calculate_bosonic(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_momenta: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_momenta.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, momentum, physical_force, exchange_force))| {
                LeadingAtomAdditiveClassicalEstimator::calculate_bosonic(
                    self,
                    index,
                    exchange_potential.clone(),
                    group_physical_potential_energy.clone(),
                    group_exchange_potential_energy.clone(),
                    group_heat.clone(),
                    group_kinetic_energy.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        adder.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as LeadingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
            })?,
        )?;
        Ok(())
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for AdditiveClassicalEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<E::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = E::Output;
    type ErrorAtom = E::ErrorAtom;
    type ErrorSystem = E::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_distinguishable(
            atom_index,
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

    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_bosonic(
            atom_index,
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

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad> for AdditiveClassicalEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_momenta: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_momenta.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, momentum, physical_force, exchange_force))| {
                InnerAtomAdditiveClassicalEstimator::calculate_distinguishable(
                    self,
                    index,
                    exchange_potential.clone(),
                    group_physical_potential_energy.clone(),
                    group_exchange_potential_energy.clone(),
                    group_heat.clone(),
                    group_kinetic_energy.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        adder.send(
                    iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                        Ok::<_, <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
                    })?,
                )?;
        Ok(())
    }

    fn calculate_bosonic(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_momenta: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_momenta.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, momentum, physical_force, exchange_force))| {
                InnerAtomAdditiveClassicalEstimator::calculate_bosonic(
                    self,
                    index,
                    exchange_potential.clone(),
                    group_physical_potential_energy.clone(),
                    group_exchange_potential_energy.clone(),
                    group_heat.clone(),
                    group_kinetic_energy.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        adder.send(
                    iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                        Ok::<_, <Self as InnerAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
                    })?,
                )?;
        Ok(())
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, E>
    TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
    for AdditiveClassicalEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<E::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = E::Output;
    type ErrorAtom = E::ErrorAtom;
    type ErrorSystem = E::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_distinguishable(
            atom_index,
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

    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_bosonic(
            atom_index,
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

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    TrailingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AdditiveClassicalEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output =
        <Self as TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_momenta: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_momenta.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, momentum, physical_force, exchange_force))| {
                TrailingAtomAdditiveClassicalEstimator::calculate_distinguishable(
                    self,
                    index,
                    exchange_potential.clone(),
                    group_physical_potential_energy.clone(),
                    group_exchange_potential_energy.clone(),
                    group_heat.clone(),
                    group_kinetic_energy.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        adder.send(
                    iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                        Ok::<_, <Self as TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
                    })?,
                )?;
        Ok(())
    }

    fn calculate_bosonic(
        &mut self,
        adder: &mut Adder,
        _multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_momenta: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
    ) -> Result<(), Self::Error> {
        let mut iter = zip_iterators!(
            images_groups_positions.read().read().read(),
            images_groups_momenta.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read()
        )
        .enumerate()
        .map(
            |(index, zip_items!(position, momentum, physical_force, exchange_force))| {
                TrailingAtomAdditiveClassicalEstimator::calculate_bosonic(
                    self,
                    index,
                    exchange_potential.clone(),
                    group_physical_potential_energy.clone(),
                    group_exchange_potential_energy.clone(),
                    group_heat.clone(),
                    group_kinetic_energy.clone(),
                    position,
                    momentum,
                    physical_force,
                    exchange_force,
                )
            },
        );
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        adder.send(
                    iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                        Ok::<_, <Self as TrailingAtomAdditiveClassicalEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable + atom_observable?)
                    })?,
                )?;
        Ok(())
    }
}
