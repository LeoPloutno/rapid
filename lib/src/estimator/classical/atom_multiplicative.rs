//! Traits and types for estimators that can be expressed as a product of observables
//! that depend only on a single atom.

use std::ops::Mul;

use arc_rw_lock::ElementRwLock;

use crate::{
    ImageHandle,
    core::{
        Multiplicative as AtomMultiplicativeClassicalEstimator, Scheme,
        error::EmptyError,
        marker::{InnerIsLeading, InnerIsTrailing},
        stat::{Bosonic, Distinguishable, Stat},
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

/// A trait for main classical estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AtomMultiplicativeClassicalEstimator<E>`]
/// atomatically implements [`MainClassicalEstimator`].
pub trait MainAtomMultiplicativeClassicalEstimator<T, V, Multiplier>
where
    Multiplier: SyncMulReciever<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`AtomMultiplicativeClassicalEstimator<Self>`] produce.
    type Output;
    /// The type of error [`AtomMultiplicativeClassicalEstimator<Self>`] returns.
    type Error: From<Multiplier::Error> + From<EmptyError>;
}

/// A trait for leading classical estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AtomMultiplicativeClassicalEstimator<E>`]
/// atomatically implements [`LeadingClassicalEstimator`].
pub trait LeadingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AtomMultiplicativeClassicalEstimator<Self>`] produce.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomMultiplicativeClassicalEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in the first image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
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

/// A trait for inner classical estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AtomMultiplicativeClassicalEstimator<E>`]
/// atomatically implements [`InnerClassicalEstimator`].
pub trait InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AtomMultiplicativeClassicalEstimator<Self>`] produce.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomMultiplicativeClassicalEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in this image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
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

/// A trait for trailing classical estimators that can be expressed as a product
/// of observables that depend only on a single atom.
///
/// For any type `E` that implements this trait, [`AtomMultiplicativeClassicalEstimator<E>`]
/// atomatically implements [`TrailingClassicalEstimator`].
pub trait TrailingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    T: Clone,
    Multiplier: SyncMulSender<Self::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`AtomMultiplicativeClassicalEstimator<Self>`] produce.
    type Output: Mul<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorAtom;
    /// The type of error [`AtomMultiplicativeClassicalEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorAtom> + From<Multiplier::Error> + From<EmptyError>;

    /// Calculates the contribution of this atom in the last image to the observable.
    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
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

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> for U
where
    T: Clone,
    Multiplier: SyncMulSender<
            <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
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
    U: InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> + InnerIsLeading + ?Sized,
{
    type Output =
        <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomMultiplicativeClassicalEstimator::calculate(
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

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad> for U
where
    T: Clone,
    Multiplier: SyncMulSender<
            <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
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
    U: InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
        + InnerIsTrailing
        + ?Sized,
{
    type Output =
        <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorAtom =
        <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom;
    type ErrorSystem =
        <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        atom_index: usize,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        InnerAtomMultiplicativeClassicalEstimator::calculate(
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

impl<T, V, Multiplier, U> MainAtomMultiplicativeClassicalEstimator<T, V, Multiplier>
    for AtomMultiplicativeClassicalEstimator<U>
where
    Multiplier: SyncMulReciever<U::Output> + ?Sized,
    U: MainAtomMultiplicativeClassicalEstimator<T, V, Multiplier> + ?Sized,
{
    type Output = U::Output;
    type Error = U::Error;
}

impl<T, V, Adder, Multiplier, U> MainClassicalEstimator<T, V, Adder, Multiplier>
    for AtomMultiplicativeClassicalEstimator<U>
where
    Adder: SyncAddReciever<<Self as MainAtomMultiplicativeClassicalEstimator<T, V, Multiplier>>::Output> + ?Sized,
    Multiplier: SyncMulReciever<<Self as MainAtomMultiplicativeClassicalEstimator<T, V, Multiplier>>::Output> + ?Sized,
    U: ?Sized,
    Self: MainAtomMultiplicativeClassicalEstimator<T, V, Multiplier>,
{
    type Output = <Self as MainAtomMultiplicativeClassicalEstimator<T, V, Multiplier>>::Output;
    type Error = <Self as MainAtomMultiplicativeClassicalEstimator<T, V, Multiplier>>::Error;

    fn calculate(&mut self, _adder: &mut Adder, multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error> {
        Ok(multiplier.recieve_prod()?.ok_or(EmptyError)?)
    }
}

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AtomMultiplicativeClassicalEstimator<U>
where
    T: Clone,
    Multiplier: SyncMulSender<U::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: LeadingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
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
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
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

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AtomMultiplicativeClassicalEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as LeadingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as LeadingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: ?Sized,
    Self: LeadingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output =
        <Self as LeadingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as LeadingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
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
        .map(|(index, zip_items!(position, momentum, physical_force, exchange_force))| {
            LeadingAtomMultiplicativeClassicalEstimator::calculate(
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
        });
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as LeadingAtomMultiplicativeClassicalEstimator<
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

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AtomMultiplicativeClassicalEstimator<U>
where
    T: Clone,
    Multiplier: SyncMulSender<U::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
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
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
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

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AtomMultiplicativeClassicalEstimator<U>
where
    T: Clone,
    Adder: SyncAddSender<
            <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: ?Sized,
    Self: InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output =
        <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
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
        .map(|(index, zip_items!(position, momentum, physical_force, exchange_force))| {
            InnerAtomMultiplicativeClassicalEstimator::calculate(
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
        });
        let first_atom_observable = iter.next().ok_or(EmptyError)??;
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<_, <Self as InnerAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::ErrorAtom>(accum_observable * atom_observable?)
            })?,
        )?;
        Ok(())
    }
}

impl<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AtomMultiplicativeClassicalEstimator<U>
where
    T: Clone,
    Multiplier: SyncMulSender<U::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: TrailingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
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
        group_heat: T,
        group_kinetic_energy: T,
        position: &V,
        momentum: &V,
        physical_force: &V,
        exchange_force: &V,
    ) -> Result<Self::Output, Self::ErrorAtom> {
        self.0.calculate(
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

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for AtomMultiplicativeClassicalEstimator<U>
where
    T: Clone,
    Adder:
        SyncAddSender<
                <Self as TrailingAtomMultiplicativeClassicalEstimator<
                    T,
                    V,
                    Multiplier,
                    Dist,
                    DistQuad,
                    Boson,
                    BosonQuad,
                >>::Output,
            > + ?Sized,
    Multiplier:
        SyncMulSender<
                <Self as TrailingAtomMultiplicativeClassicalEstimator<
                    T,
                    V,
                    Multiplier,
                    Dist,
                    DistQuad,
                    Boson,
                    BosonQuad,
                >>::Output,
            > + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: ?Sized,
    Self: TrailingAtomMultiplicativeClassicalEstimator<T, V, Multiplier, Dist, DistQuad, Boson, BosonQuad>,
{
    type Output = <Self as TrailingAtomMultiplicativeClassicalEstimator<
        T,
        V,
        Multiplier,
        Dist,
        DistQuad,
        Boson,
        BosonQuad,
    >>::Output;
    type Error = <Self as TrailingAtomMultiplicativeClassicalEstimator<
        T,
        V,
        Multiplier,
        Dist,
        DistQuad,
        Boson,
        BosonQuad,
    >>::ErrorSystem;

    fn calculate(
        &mut self,
        _adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
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
                TrailingAtomMultiplicativeClassicalEstimator::calculate(
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
        multiplier.send(
            iter.try_fold(first_atom_observable, |accum_observable, atom_observable| {
                Ok::<
                    _,
                    <Self as TrailingAtomMultiplicativeClassicalEstimator<
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
