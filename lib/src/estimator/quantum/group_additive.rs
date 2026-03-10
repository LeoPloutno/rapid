//! Traits and types for estimators that can be expressed as a sum of observables
//! that depend on disjoint groups of atoms.

use std::ops::Add;

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
};

/// A wrapper that automatically implements a `QuantumEstimator` trait if
/// the inner type inmplements its `GroupAdditive` variant.
pub struct GroupAdditiveQuantumEstimator<E: ?Sized>(pub E);

/// A trait for main quantum estimators that can be expressed as a sum
/// of observables that depend on disjoint groups of atoms.
///
/// For any type `E` that implements this trait, [`GroupAdditiveQuantumEstimator<E>`]
/// atomatically implements [`MainQuantumEstimator`].
///
/// [`MainQuantumEstimator`]: super::MainQuantumEstimator
pub trait MainGroupAdditiveQuantumEstimator<T, V, Adder>
where
    Adder: SyncAddReciever<Self::Output> + ?Sized,
{
    /// The type of output `Self` and [`GroupAdditiveQuantumEstimator<Self>`] produce.
    type Output;
    /// The type of error [`GroupAdditiveQuantumEstimator<Self>`] returns.
    type Error: From<Adder::Error> + From<EmptyError>;
}

/// A trait for leading quantum estimators that can be expressed as a sum
/// of observables that depend on disjoint groups of atoms.
///
/// For any type `E` that implements this trait, [`GroupAdditiveQuantumEstimator<E>`]
/// atomatically implements [`LeadingQuantumEstimator`].
///
/// [`LeadingQuantumEstimator`]: super::LeadingQuantumEstimator
pub trait LeadingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`GroupAdditiveQuantumEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorGroup;
    /// The type of error [`GroupAdditiveQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorGroup> + From<Adder::Error>;

    /// Calculates the contribution of this group in the first image to the observable.
    fn calculate(
        &mut self,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_positions: &[V],
        group_physical_forces: &[V],
        group_exchange_forces: &[V],
    ) -> Result<Self::Output, Self::ErrorGroup>;
}

/// A trait for inner quantum estimators that can be expressed as a sum
/// of observables that depend on disjoint groups of atoms.
///
/// For any type `E` that implements this trait, [`GroupAdditiveQuantumEstimator<E>`]
/// atomatically implements [`InnerQuantumEstimator`].
///
/// [`InnerQuantumEstimator`]: super::InnerQuantumEstimator
pub trait InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`GroupAdditiveQuantumEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorGroup;
    /// The type of error [`GroupAdditiveQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorGroup> + From<Adder::Error>;

    /// Calculates the contribution of this group in this image to the observable.
    fn calculate(
        &mut self,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_positions: &[V],
        group_physical_forces: &[V],
        group_exchange_forces: &[V],
    ) -> Result<Self::Output, Self::ErrorGroup>;
}

/// A trait for trailing quantum estimators that can be expressed as a sum
/// of observables that depend only on a group of atoms.
///
/// For any type `E` that implements this trait, [`GroupAdditiveQuantumEstimator<E>`]
/// atomatically implements [`TrailingQuantumEstimator`].
///
/// [`TrailingQuantumEstimator`]: super::TrailingQuantumEstimator
pub trait TrailingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSender<Self::Output> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type of output `Self` and [`GroupAdditiveQuantumEstimator<Self>`] produce.
    type Output: Add<Output = Self::Output>;
    /// The type of error `Self` returns.
    type ErrorGroup;
    /// The type of error [`GroupAdditiveQuantumEstimator<Self>`] returns.
    type ErrorSystem: From<Self::ErrorGroup> + From<Adder::Error>;

    /// Calculates the contribution of this group in the last image to the observable.
    fn calculate(
        &mut self,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_positions: &[V],
        group_physical_forces: &[V],
        group_exchange_forces: &[V],
    ) -> Result<Self::Output, Self::ErrorGroup>;
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for U
where
    Adder: SyncAddSender<
            <Self as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
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
    U: InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + InnerIsLeading + ?Sized,
{
    type Output = <Self as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorGroup =
        <Self as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorGroup;
    type ErrorSystem =
        <Self as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        group_positions: &[V],
        group_physical_forces: &[V],
        group_exchange_forces: &[V],
    ) -> Result<Self::Output, Self::ErrorGroup> {
        InnerGroupAdditiveQuantumEstimator::calculate(
            self,
            exchange_potential,
            physical_potential_energy,
            exchange_potential_energy,
            group_positions,
            group_physical_forces,
            group_exchange_forces,
        )
    }
}

impl<T, V, Adder, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> for U
where
    Adder: SyncAddSender<
            <Self as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
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
    U: InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + InnerIsTrailing + ?Sized,
{
    type Output = <Self as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type ErrorGroup =
        <Self as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorGroup;
    type ErrorSystem =
        <Self as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

    fn calculate(
        &mut self,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        group_physical_potential_energy: T,
        group_exchange_potential_energy: T,
        group_positions: &[V],
        group_physical_forces: &[V],
        group_exchange_forces: &[V],
    ) -> Result<Self::Output, Self::ErrorGroup> {
        InnerGroupAdditiveQuantumEstimator::calculate(
            self,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            group_positions,
            group_physical_forces,
            group_exchange_forces,
        )
    }
}

impl<T, V, Adder, Multiplier, U> MainQuantumEstimator<T, V, Adder, Multiplier> for GroupAdditiveQuantumEstimator<U>
where
    Adder: SyncAddReciever<<U as MainGroupAdditiveQuantumEstimator<T, V, Adder>>::Output> + ?Sized,
    Multiplier: SyncMulReciever<<U as MainGroupAdditiveQuantumEstimator<T, V, Adder>>::Output> + ?Sized,
    U: MainGroupAdditiveQuantumEstimator<T, V, Adder> + ?Sized,
{
    type Output = <U as MainGroupAdditiveQuantumEstimator<T, V, Adder>>::Output;
    type Error = <U as MainGroupAdditiveQuantumEstimator<T, V, Adder>>::Error;

    fn calculate(&mut self, adder: &mut Adder, _multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error> {
        Ok(adder.recieve_sum()?.ok_or(EmptyError)?)
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for GroupAdditiveQuantumEstimator<U>
where
    Adder: SyncAddSender<
            <U as LeadingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <U as LeadingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: LeadingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + ?Sized,
{
    type Output = <U as LeadingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <U as LeadingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

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
        adder.send(LeadingGroupAdditiveQuantumEstimator::calculate(
            &mut self.0,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            images_groups_positions.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read(),
        )?)?;
        Ok(())
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    InnerQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for GroupAdditiveQuantumEstimator<U>
where
    Adder: SyncAddSender<<U as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output>
        + ?Sized,
    Multiplier: SyncMulSender<<U as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output>
        + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + ?Sized,
{
    type Output = <U as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error = <U as InnerGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

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
        adder.send(InnerGroupAdditiveQuantumEstimator::calculate(
            &mut self.0,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            images_groups_positions.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read(),
        )?)?;
        Ok(())
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingQuantumEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
    for GroupAdditiveQuantumEstimator<U>
where
    Adder: SyncAddSender<
            <U as TrailingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Multiplier: SyncMulSender<
            <U as TrailingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output,
        > + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    U: TrailingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad> + ?Sized,
{
    type Output = <U as TrailingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error =
        <U as TrailingGroupAdditiveQuantumEstimator<T, V, Adder, Dist, DistQuad, Boson, BosonQuad>>::ErrorSystem;

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
        adder.send(TrailingGroupAdditiveQuantumEstimator::calculate(
            &mut self.0,
            exchange_potential,
            group_physical_potential_energy,
            group_exchange_potential_energy,
            images_groups_positions.read().read().read(),
            images_groups_physical_forces.read().read().read(),
            images_groups_exchange_forces.read().read().read(),
        )?)?;
        Ok(())
    }
}
