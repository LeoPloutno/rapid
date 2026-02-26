use crate::{
    core::{GroupImageHandle, GroupTypeHandle},
    marker::{InnerIsLeading, InnerIsTrailing},
    potential::exchange::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential},
    stat::{Bosonic, Distinguishable, Stat},
    sync_ops::{SyncAddRecv, SyncAddSend, SyncMulRecv, SyncMulSend},
};

/// A trait for quantum estimators.
/// The implementor of this trait recieves the calculations of
/// the other quantum observables and produces an output.
pub trait MainQuantumObservable<T, V, Adder, Multiplier>
where
    Adder: SyncAddRecv<T> + ?Sized,
    Multiplier: SyncMulRecv<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the observable.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(&mut self, adder: &mut Adder, multiplier: &mut Multiplier) -> Result<Self::Output, Self::Error>;
}

/// A trait for quantum estimators operating in the first replica for a specific group.
pub trait LeadingQuantumObservable<T, V, Adder, Multiplier, Dist, Boson>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable,
    Boson: LeadingExchangePotential<T, V> + Bosonic,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in the first replica
    /// to the observable and sends it to a `MainQuantumObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        exchange_potential: &Stat<Dist, Boson>,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantum estimators operating in an inner replica for a specific group.
pub trait InnerQuantumObservable<T, V, Adder, Multiplier, Dist, Boson>
where
    Dist: InnerExchangePotential<T, V> + Distinguishable,
    Boson: InnerExchangePotential<T, V> + Bosonic,
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in this replica
    /// to the observable and sends it to a `MainQuantumObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        exchange_potential: &Stat<Dist, Boson>,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

/// A trait for quantum estimators operating in the last replica for a specific group.
pub trait TrailingQuantumObservable<T, V, Adder, Multiplier, Dist, Boson>
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable,
    Boson: TrailingExchangePotential<T, V> + Bosonic,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in the last replica
    /// to the observable and sends it to a `MainQuantumObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        exchange_potential: &Stat<Dist, Boson>,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error>;
}

impl<T, V, Adder, Multiplier, Dist, Boson, U> LeadingQuantumObservable<T, V, Adder, Multiplier, Dist, Boson> for U
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + InnerExchangePotential<T, V> + Distinguishable,
    Boson: LeadingExchangePotential<T, V> + InnerExchangePotential<T, V> + Bosonic,
    U: InnerQuantumObservable<T, V, Adder, Multiplier, Dist, Boson> + InnerIsLeading,
{
    type Output = <Self as InnerQuantumObservable<T, V, Adder, Multiplier, Dist, Boson>>::Output;
    type Error = <Self as InnerQuantumObservable<T, V, Adder, Multiplier, Dist, Boson>>::Error;

    fn calculate(
        &mut self,
        exchange_potential: &Stat<Dist, Boson>,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error> {
        InnerQuantumObservable::calculate(
            self,
            exchange_potential,
            adder,
            multiplier,
            physical_potential_energy,
            exchange_potential_energy,
            groups_positions,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}

impl<T, V, Adder, Multiplier, Dist, Boson, U> TrailingQuantumObservable<T, V, Adder, Multiplier, Dist, Boson> for U
where
    Adder: SyncAddSend<T> + ?Sized,
    Multiplier: SyncMulSend<T> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Distinguishable,
    Boson: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Bosonic,
    U: InnerQuantumObservable<T, V, Adder, Multiplier, Dist, Boson> + InnerIsTrailing,
{
    type Output = <Self as InnerQuantumObservable<T, V, Adder, Multiplier, Dist, Boson>>::Output;
    type Error = <Self as InnerQuantumObservable<T, V, Adder, Multiplier, Dist, Boson>>::Error;

    fn calculate(
        &mut self,
        exchange_potential: &Stat<Dist, Boson>,
        adder: &mut Adder,
        multiplier: &mut Multiplier,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        groups_positions: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_physical_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
        groups_exchange_forces: &[GroupImageHandle<GroupTypeHandle<V>>],
    ) -> Result<(), Self::Error> {
        InnerQuantumObservable::calculate(
            self,
            exchange_potential,
            adder,
            multiplier,
            physical_potential_energy,
            exchange_potential_energy,
            groups_positions,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}
