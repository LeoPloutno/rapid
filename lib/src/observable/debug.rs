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
pub trait LeadingDebugObservable<T, V, D, B, A, M, E>
where
    D: LeadingExchangePotential<T, V> + Distinguishable,
    B: LeadingExchangePotential<T, V> + Bosonic,
    A: SyncAddRecv<Self::Output> + ?Sized,
    M: SyncMulRecv<Self::Output> + ?Sized,
    E: From<A::Error> + From<M::Error>,
{
    type Output;

    /// Calculates the quantity.
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
        momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<Self::Output, E>;
}

/// A trait for objects that assist
/// an [`LeadingDebugObservable`] in calculating the observable value
/// from an inner replica.
pub trait InnerDebugObservable<T, V, D, B, A, M, E>
where
    D: InnerExchangePotential<T, V> + Distinguishable,
    B: InnerExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<Self::Output> + ?Sized,
    M: SyncMulSend<Self::Output> + ?Sized,
    E: From<A::Error> + From<M::Error>,
{
    type Output;

    /// Calculates the quantity.
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
        momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), E>;
}

/// A trait for objects that assist
/// a [`LeadingDebugObservable`] in calculating the observable value
/// from the last replica.
pub trait TrailingDebugObservable<T, V, D, B, A, M, E>
where
    D: TrailingExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<Self::Output> + ?Sized,
    M: SyncMulSend<Self::Output> + ?Sized,
    E: From<A::Error> + From<M::Error>,
{
    type Output;

    /// Calculates the quantity.
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
        momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), E>;
}

impl<T, V, D, B, A, M, E, U> TrailingDebugObservable<T, V, D, B, A, M, E> for U
where
    D: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<<Self as InnerDebugObservable<T, V, D, B, A, M, E>>::Output> + ?Sized,
    M: SyncMulSend<<Self as InnerDebugObservable<T, V, D, B, A, M, E>>::Output> + ?Sized,
    E: From<A::Error> + From<M::Error>,
    U: InnerDebugObservable<T, V, D, B, A, M, E> + InnerIsTrailing,
{
    type Output = <Self as InnerDebugObservable<T, V, D, B, A, M, E>>::Output;

    fn calculate(
        &mut self,
        adder: &mut A,
        multiplier: &mut M,
        exchange_potential: &Stat<D, B>,
        last_replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(), E> {
        InnerDebugObservable::calculate(
            self,
            adder,
            multiplier,
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
