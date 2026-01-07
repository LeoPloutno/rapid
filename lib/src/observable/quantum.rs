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

/// A trait for quantum estimators that return the observable value.
pub trait LeadingQuantumObservable<T, V, D, B, A, M>
where
    D: LeadingExchangePotential<T, V> + Distinguishable,
    B: LeadingExchangePotential<T, V> + Bosonic,
    A: SyncAddRecv<T> + ?Sized,
    M: SyncMulRecv<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the observable.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        exchange_potential: &Stat<D, B>,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<Self::Output, Self::Error>;
}

/// A trait for quantum estimators that assist
/// a [`LeadingQuantumObservable`] in calculating the observable value
/// from an inner replica.
pub trait InnerQuantumObservable<T, V, D, B, A, M>
where
    D: InnerExchangePotential<T, V> + Distinguishable,
    B: InnerExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the observable.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        exchange_potential: &Stat<D, B>,
        replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error>;
}

/// A trait for quantum estimators that assist
/// a [`LeadingQuantumObservable`] in calculating the observable value
/// from the last replica.
pub trait TrailingQuantumObservable<T, V, D, B, A, M>
where
    D: TrailingExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
{
    type Output;
    type Error;

    /// Calculates the observable.
    ///
    /// Returns an error if a synchronization failure occurs.
    #[must_use]
    fn calculate(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        exchange_potential: &Stat<D, B>,
        last_replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error>;
}

impl<T, V, D, B, A, M, U> TrailingQuantumObservable<T, V, D, B, A, M> for U
where
    D: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T> + ?Sized,
    M: SyncMulSend<T> + ?Sized,
    U: InnerQuantumObservable<T, V, D, B, A, M> + InnerIsTrailing,
{
    type Output = <Self as InnerQuantumObservable<T, V, D, B, A, M>>::Output;
    type Error = <Self as InnerQuantumObservable<T, V, D, B, A, M>>::Error;

    fn calculate(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        exchange_potential: &Stat<D, B>,
        last_replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), Self::Error> {
        InnerQuantumObservable::calculate(
            self,
            adder,
            multiplier,
            exchange_potential,
            last_replica,
            groups,
            group_idx,
            positions,
            forces,
        )
    }
}
