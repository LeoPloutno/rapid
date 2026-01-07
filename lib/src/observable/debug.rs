use arc_rw_lock::{ElementRwLock, UniqueArcSliceRwLock};

use crate::{
    core::AtomGroupInfo,
    marker::InnerIsTrailing,
    potential::exchange::{
        InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential,
    },
    stat::{Bosonic, Distinguishable, Stat},
    sync_ops::{SyncAddRecv, SyncAddSend, SyncMulRecv, SyncMulSend},
};

/// A trait for quantities which may be used to debug the simulation,
/// returned from the first replica.
pub trait LeadingDebugObservable<T, V, D, B, A, M>
where
    D: LeadingExchangePotential<T, V> + Distinguishable,
    B: LeadingExchangePotential<T, V> + Bosonic,
    A: SyncAddRecv<T> + ?Sized,
    M: SyncMulRecv<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the quantity.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        phys_potential_energy: T,
        exch_potential_energy: T,
        kinetic_energy: T,
        exchange_potential: &Stat<D, B>,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<Self::Output, Self::Error>;
}

/// A trait for objects that assist
/// an [`LeadingDebugObservable`] in calculating the observable value
/// from an inner replica.
pub trait InnerDebugObservable<T, V, D, B, A, M>
where
    D: InnerExchangePotential<T, V> + Distinguishable,
    B: InnerExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the quantity.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        phys_potential_energy: T,
        exch_potential_energy: T,
        kinetic_energy: T,
        exchange_potential: &Stat<D, B>,
        replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error>;
}

/// A trait for objects that assist
/// a [`LeadingDebugObservable`] in calculating the observable value
/// from the last replica.
pub trait TrailingDebugObservable<T, V, D, B, A, M>
where
    D: TrailingExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the quantity.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        phys_potential_energy: T,
        exch_potential_energy: T,
        kinetic_energy: T,
        exchange_potential: &Stat<D, B>,
        last_replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error>;
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
        adder: &mut A,
        multiplier: &mut M,
        phys_potential_energy: T,
        exch_potential_energy: T,
        kinetic_energy: T,
        exchange_potential: &Stat<D, B>,
        last_replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error> {
        InnerDebugObservable::calculate(
            self,
            adder,
            multiplier,
            phys_potential_energy,
            exch_potential_energy,
            kinetic_energy,
            exchange_potential,
            last_replica,
            groups,
            group_idx,
            positions,
            momenta,
            forces,
        )
    }
}
