use arc_rw_lock::{ElementRwLock, UniqueArcSliceRwLock};

use crate::{
    marker::{InnerIsLeading, InnerIsTrailing},
    potential::exchange::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential},
    stat::{Bosonic, Distinguishable, Stat},
    sync_ops::{SyncAddRecv, SyncAddSend, SyncMulRecv, SyncMulSend},
};

/// A trait for quantities which may be used to debug the simulation.
/// The implementor of this trait recieves the calculations of
/// the other debug observables and produces an output.
pub trait MainDebugObservable<T, V, A, M>
where
    A: SyncAddRecv<T> + ?Sized,
    M: SyncMulRecv<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the quantity.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(&mut self, adder: &mut A, multiplier: &mut M) -> Result<Self::Output, Self::Error>;
}

/// A trait for quantities which may be used to debug the simulation,
/// operating in the first replica for a specific group of atoms.
pub trait LeadingDebugObservable<T, V, D, B, A, M>
where
    D: LeadingExchangePotential<T, V> + Distinguishable,
    B: LeadingExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in the first replica
    /// to the quantity and sends it to a `MainDebugObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_physical_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_exchange_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error>;
}

/// A trait for quantities which may be used to debug the simulation,
/// operating in an inner replica for a specific group of atoms.
pub trait InnerDebugObservable<T, V, D, B, A, M>
where
    D: InnerExchangePotential<T, V> + Distinguishable,
    B: InnerExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in this replica
    /// to the quantity and sends it to a `MainDebugObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_physical_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_exchange_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error>;
}

/// A trait for quantities which may be used to debug the simulation,
/// operating in the last replica for a specific group.
pub trait TrailingDebugObservable<T, V, D, B, A, M>
where
    D: TrailingExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the contribution of this group in the last replica
    /// to the quantity and sends it to a `MainDebugObservable`.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_physical_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_exchange_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error>;
}

impl<T, V, D, B, A, M, U> LeadingDebugObservable<T, V, D, B, A, M> for U
where
    D: LeadingExchangePotential<T, V> + InnerExchangePotential<T, V> + Distinguishable,
    B: LeadingExchangePotential<T, V> + InnerExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
    U: InnerDebugObservable<T, V, D, B, A, M> + InnerIsLeading,
{
    type Output = <Self as InnerDebugObservable<T, V, D, B, A, M>>::Output;
    type Error = <Self as InnerDebugObservable<T, V, D, B, A, M>>::Error;

    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_physical_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_exchange_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error> {
        InnerDebugObservable::calculate(
            self,
            exchange_potential,
            adder,
            multiplier,
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

impl<T, V, D, B, A, M, U> TrailingDebugObservable<T, V, D, B, A, M> for U
where
    D: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
    U: InnerDebugObservable<T, V, D, B, A, M> + InnerIsTrailing,
{
    type Output = <Self as InnerDebugObservable<T, V, D, B, A, M>>::Output;
    type Error = <Self as InnerDebugObservable<T, V, D, B, A, M>>::Error;

    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        adder: &mut A,
        multiplier: &mut M,
        physical_potential_energy: T,
        exchange_potential_energy: T,
        kinetic_energy: T,
        groups_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_physical_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_exchange_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error> {
        InnerDebugObservable::calculate(
            self,
            exchange_potential,
            adder,
            multiplier,
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
