//! Traits for calculating classical quantities.

use crate::{
    ImageHandle,
    core::{
        Scheme,
        marker::{InnerIsLeading, InnerIsTrailing},
        stat::{Bosonic, Distinguishable, Stat},
        sync_ops::{SyncAddReciever, SyncAddSender, SyncMulReciever, SyncMulSender},
    },
    potential::exchange::{
        InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential,
        quadratic::{
            InnerQuadraticExpansionExchangePotential, LeadingQuadraticExpansionExchangePotential,
            TrailingQuadraticExpansionExchangePotential,
        },
    },
};

/// A trait for quantities calculated from the whole system treated as a classical one.
/// The implementor of this trait recieves the calculations of
/// the other classical estimators and produces an output.
pub trait MainClassicalgEstimator<T, V, Adder, Multiplier>
where
    Adder: SyncAddReciever<T> + ?Sized,
    Multiplier: SyncMulReciever<T> + ?Sized,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the quantity.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(&mut self, adder: &mut Adder, multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error>;
}

/// A trait for quantities calculated from the whole system treated as a classical one,
/// operating in the first replica for a specific group of atoms.
pub trait LeadingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSender<T> + ?Sized,
    Multiplier: SyncMulSender<T> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group in the first replica
    /// to the quantity and sends it to a `MainClassicalEstimator`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        heat: T,
        kinetic_energy: T,
        groups_positions: &[ImageHandle<V>],
        groups_momenta: &[ImageHandle<V>],
        groups_physical_forces: &[ImageHandle<V>],
        groups_exchange_forces: &[ImageHandle<V>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantities calculated from the whole system treated as a classical one,
/// operating in an inner replica for a specific group of atoms.
pub trait InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSender<T> + ?Sized,
    Multiplier: SyncMulSender<T> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group in this replica
    /// to the quantity and sends it to a `MainClassicalEstimator`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        heat: T,
        kinetic_energy: T,
        groups_positions: &[ImageHandle<V>],
        groups_momenta: &[ImageHandle<V>],
        groups_physical_forces: &[ImageHandle<V>],
        groups_exchange_forces: &[ImageHandle<V>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantities calculated from the whole system treated as a classical one,
/// operating in the last replica for a specific group.
pub trait TrailingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSender<T> + ?Sized,
    Multiplier: SyncMulSender<T> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    /// The type associated with the output returned by the implementor.
    type Output;
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Calculates the contribution of this group in the last replica
    /// to the quantity and sends it to a `MainClassicalEstimator`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        heat: T,
        kinetic_energy: T,
        groups_positions: &[ImageHandle<V>],
        groups_momenta: &[ImageHandle<V>],
        groups_physical_forces: &[ImageHandle<V>],
        groups_exchange_forces: &[ImageHandle<V>],
    ) -> Result<(), Self::Error>;
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad> for U
where
    Adder: SyncAddSender<T> + ?Sized,
    Multiplier: SyncMulSender<T> + ?Sized,
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
    U: InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad> + InnerIsLeading + ?Sized,
{
    type Output = <Self as InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error = <Self as InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Error;

    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        heat: T,
        kinetic_energy: T,
        groups_positions: &[ImageHandle<V>],
        groups_momenta: &[ImageHandle<V>],
        groups_physical_forces: &[ImageHandle<V>],
        groups_exchange_forces: &[ImageHandle<V>],
    ) -> Result<(), Self::Error> {
        InnerClassicalEstimator::calculate(
            self,
            adder,
            multiplier,
            exchange_potential,
            physical_potential_energy,
            exchange_potential_energy,
            heat,
            kinetic_energy,
            groups_positions,
            groups_momenta,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    TrailingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad> for U
where
    Adder: SyncAddSender<T> + ?Sized,
    Multiplier: SyncMulSender<T> + ?Sized,
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
    U: InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad> + InnerIsTrailing + ?Sized,
{
    type Output = <Self as InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Output;
    type Error = <Self as InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>>::Error;

    fn calculate(
        &mut self,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        exchange_potential: Scheme<Stat<&Dist, &Boson>, Stat<&DistQuad, &BosonQuad>>,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        heat: T,
        kinetic_energy: T,
        groups_positions: &[ImageHandle<V>],
        groups_momenta: &[ImageHandle<V>],
        groups_physical_forces: &[ImageHandle<V>],
        groups_exchange_forces: &[ImageHandle<V>],
    ) -> Result<(), Self::Error> {
        InnerClassicalEstimator::calculate(
            self,
            adder,
            multiplier,
            exchange_potential,
            physical_potential_energy,
            exchange_potential_energy,
            heat,
            kinetic_energy,
            groups_positions,
            groups_momenta,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}
