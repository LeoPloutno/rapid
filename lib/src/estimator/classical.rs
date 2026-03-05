use crate::{
    core::{GroupImageHandle, GroupTypeHandle, Scheme},
    marker::{InnerIsLeading, InnerIsTrailing},
    potential::exchange::{
        InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential,
        quadratic::{
            InnerQuadraticExpansionExchangePotential, LeadingQuadraticExpansionExchangePotential,
            TrailingQuadraticExpansionExchangePotential,
        },
    },
    stat::{Bosonic, Distinguishable, Stat},
    sync_ops::{SyncAddRecv, SyncAddSend, SyncMulRecv, SyncMulSend},
};

/// A trait for quantities calculated from the whole system treated as a classical one.
/// The implementor of this trait recieves the calculations of
/// the other classical estimators and produces an output.
pub trait MainClassicalgEstimator<T, V, Adder, Multiplier>
where
    Adder: SyncAddRecv<T> + ?Sized,
    Multiplier: SyncMulRecv<T> + ?Sized,
{
    type Output;
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
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    type Output;
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
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantities calculated from the whole system treated as a classical one,
/// operating in an inner replica for a specific group of atoms.
pub trait InnerClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    type Output;
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
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantities calculated from the whole system treated as a classical one,
/// operating in the last replica for a specific group.
pub trait TrailingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    DistQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    BosonQuad: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
{
    type Output;
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
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

impl<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad, U>
    LeadingClassicalEstimator<T, V, Adder, Multiplier, Dist, DistQuad, Boson, BosonQuad> for U
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
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
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error> {
        InnerClassicalEstimator::calculate(
            self,
            adder,
            multiplier,
            exchange_potential,
            physical_potential_energy,
            exchange_potential_energy,
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
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
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
        kinetic_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_momenta: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error> {
        InnerClassicalEstimator::calculate(
            self,
            adder,
            multiplier,
            exchange_potential,
            physical_potential_energy,
            exchange_potential_energy,
            kinetic_energy,
            groups_positions,
            groups_momenta,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}
