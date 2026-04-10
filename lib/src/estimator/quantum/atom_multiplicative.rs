//! Traits and types for estimators that can be expressed as a product of observables
//! that depend only on a single atom.

use std::ops::Mul;

use arc_rw_lock::ElementRwLock;

use crate::{
    ImageHandle,
    core::{
        Multiplicative as MultiplicativeQuantumEstimator, Scheme,
        error::EmptyError,
        marker::{InnerIsLeading, InnerIsTrailing},
        stat::{Bosonic, Distinguishable},
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

/// A trait for main quantum estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`MainQuantumEstimator`].
pub trait MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>
where
    Multiplier: SyncMulReciever<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`MultiplicativeQuantumEstimator<Self>`] produce.
    type Output;
    /// The type of error [`MultiplicativeQuantumEstimator<Self>`] returns.
    type Error: From<Multiplier::Error> + From<EmptyError>;
}

/// A trait for leading quantum estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`LeadingQuantumEstimator`].
pub trait LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`MultiplicativeQuantumEstimator<Self>`] produce.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`MultiplicativeQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in the first image to the observable
    /// given that the whole group has distinguishable statistics.
    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
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
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

/// A trait for inner quantum estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`InnerQuantumEstimator`].
pub trait InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`MultiplicativeQuantumEstimator<Self>`] produce.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`MultiplicativeQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in this image to the observable
    /// given that the whole group has distinguishable statistics.
    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
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
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

/// A trait for trailing quantum estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`MultiplicativeQuantumEstimator<E>`]
/// atomatically implements [`TrailingQuantumEstimator`].
pub trait TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`MultiplicativeQuantumEstimator<Self>`] produce.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`MultiplicativeQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in the last image to the observable
    /// given that the whole group has distinguishable statistics.
    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
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
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom>;
}

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> for E
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
    E: InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> + InnerIsLeading + ?Sized,
{
    type Output =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomMultiplicativeQuantumEstimator::calculate_distinguishable(self, atom_index, exchange_potential, group_physical_potential_energy, group_exchange_potential_energy, position, physical_force, exchange_force)
    }

    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomMultiplicativeQuantumEstimator::calculate_bosonic(
            self,
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force
        )
    }
}

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> for E
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
    E: InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
        + InnerIsTrailing
        + ?Sized,
{
    type Output =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Dist, &DistQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomMultiplicativeQuantumEstimator::calculate_distinguishable(
            self,
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force
        )
    }

    fn calculate_bosonic(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomMultiplicativeQuantumEstimator::calculate_bosonic(
            self,
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
            physical_force,
            exchange_force
        )
    }
}

impl<T, V, Multiplier, E> MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier> for MultiplicativeQuantumEstimator<E>
where
    Multiplier: SyncMulReciever<E::Output> + ?Sized,
    E: MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier> + ?Sized,
{
    type Output = E::Output;
    type Error = E::Error;
}

impl<T, V, Adder, Multiplier, E> MainQuantumEstimator<T, V, Adder, Multiplier> for MultiplicativeQuantumEstimator<E>
where
    Adder: SyncAddReciever<<Self as MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>>::Output> + ?Sized,
    Multiplier: SyncMulReciever<<Self as MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>>::Output> + ?Sized,
    E: ?Sized,
    Self: MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>,
{
    type Output = <Self as MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>>::Output;
    type Error = <Self as MainAtomMultiplicativeQuantumEstimator<T, V, Multiplier>>::Error;

    fn calculate(&mut self, _adder: &mut Adder, multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error> {
        Ok(multiplier.recieve_prod()?.ok_or(EmptyError)?)
    }
}

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for MultiplicativeQuantumEstimator<E>
where
    T: Clone,
    Multiplier: SyncMulSender<E::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
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
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_distinguishable(
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
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
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_bosonic(
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
    LeadingQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for MultiplicativeQuantumEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output =
        <Self as LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as LeadingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Dist, &DistQuad>,
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
            LeadingAtomMultiplicativeQuantumEstimator::calculate_distinguishable(
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as LeadingAtomMultiplicativeQuantumEstimator<
                        T,
                        V,
                        Multiplier,
                        Dist,
                        DistQuad,
                        Boson,
                        BosonQuad,
                    >>::ErrorAtom,
                >(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }

    fn calculate_bosonic(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
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
            LeadingAtomMultiplicativeQuantumEstimator::calculate_bosonic(
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as LeadingAtomMultiplicativeQuantumEstimator<
                        T,
                        V,
                        Multiplier,
                        Dist,
                        DistQuad,
                        Boson,
                        BosonQuad,
                    >>::ErrorAtom,
                >(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }
}

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for MultiplicativeQuantumEstimator<E>
where
    T: Clone,
    Multiplier: SyncMulSender<E::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
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
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_distinguishable(
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
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
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_bosonic(
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
    InnerQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for MultiplicativeQuantumEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Dist, &DistQuad>,
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
            InnerAtomMultiplicativeQuantumEstimator::calculate_distinguishable(
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }

    fn calculate_bosonic(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
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
            InnerAtomMultiplicativeQuantumEstimator::calculate_bosonic(
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as InnerAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }
}

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, E>
    TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for MultiplicativeQuantumEstimator<E>
where
    T: Clone,
    Multiplier: SyncMulSender<E::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
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
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_distinguishable(
            atom_index,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            position,
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
        position: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate_bosonic(
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
    TrailingQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for MultiplicativeQuantumEstimator<E>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    E: ?Sized,
    Self: TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output =
        <Self as TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as TrailingAtomMultiplicativeQuantumEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate_distinguishable(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Dist, &DistQuad>,
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
            TrailingAtomMultiplicativeQuantumEstimator::calculate_distinguishable(
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as TrailingAtomMultiplicativeQuantumEstimator<
                        T,
                        V,
                        Multiplier,
                        Dist,
                        DistQuad,
                        Boson,
                        BosonQuad,
                    >>::ErrorAtom,
                >(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }

    fn calculate_bosonic(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<&Boson, &BosonQuad>,
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
            TrailingAtomMultiplicativeQuantumEstimator::calculate_bosonic(
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as TrailingAtomMultiplicativeQuantumEstimator<
                        T,
                        V,
                        Multiplier,
                        Dist,
                        DistQuad,
                        Boson,
                        BosonQuad,
                    >>::ErrorAtom,
                >(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }
}
