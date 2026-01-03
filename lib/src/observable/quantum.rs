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
pub trait LeadingQuantumObservable<T, V, A, M, D, B, E>
where
    A: SyncAddRecv<Self::Output> + ?Sized,
    M: SyncMulRecv<Self::Output> + ?Sized,
    D: LeadingExchangePotential<T, V> + Distinguishable,
    B: LeadingExchangePotential<T, V> + Bosonic,
    E: From<A::Error> + From<M::Error>,
{
    type Output;

    /// Calculates the observable.
    ///
    /// Returns an error if a synchronization failure occurs.
    fn calculate(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        exchange_potential: &Stat<D, B>,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<Self::Output, E>;
}

/// A trait for quantum estimators that assist
/// a [`LeadingQuantumObservable`] in calculating the observable value
/// from an inner replica.
pub trait InnerQuantumObservable<T, V, A, M, D, B, E>
where
    A: SyncAddSend<Self::Output> + ?Sized,
    M: SyncMulSend<Self::Output> + ?Sized,
    D: InnerExchangePotential<T, V> + Distinguishable,
    B: InnerExchangePotential<T, V> + Bosonic,
    E: From<A::Error> + From<M::Error>,
{
    type Output;

    /// Calculates the observable.
    ///
    /// Returns an error if a synchronization failure occurs.
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
    ) -> Result<(), E>;
}

/// A trait for quantum estimators that assist
/// a [`LeadingQuantumObservable`] in calculating the observable value
/// from the last replica.
pub trait TrailingQuantumObservable<T, V, A, M, D, B, E>
where
    A: SyncAddSend<Self::Output> + ?Sized,
    M: SyncMulSend<Self::Output> + ?Sized,
    D: TrailingExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + Bosonic,
    E: From<A::Error> + From<M::Error>,
{
    type Output;

    /// Calculates the observable.
    ///
    /// Returns an error if a synchronization failure occurs.
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
    ) -> Result<(), E>;
}

impl<T, V, A, M, D, B, E, U> TrailingQuantumObservable<T, V, A, M, D, B, E> for U
where
    A: SyncAddSend<<Self as InnerQuantumObservable<T, V, A, M, D, B, E>>::Output> + ?Sized,
    M: SyncMulSend<<Self as InnerQuantumObservable<T, V, A, M, D, B, E>>::Output> + ?Sized,
    D: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Bosonic,
    E: From<A::Error> + From<M::Error>,
    U: InnerQuantumObservable<T, V, A, M, D, B, E> + InnerIsTrailing,
{
    type Output = <Self as InnerQuantumObservable<T, V, A, M, D, B, E>>::Output;

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
    ) -> Result<(), E> {
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
