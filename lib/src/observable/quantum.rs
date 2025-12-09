use arc_rw_lock::{ElementRwLock, UniqueArcSliceRwLock};

use crate::{
    core::AtomGroupInfo,
    marker::{InnerIsLeading, InnerIsTrailing},
    potential::exchange::{
        InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential,
    },
    stat::{Bosonic, Distinguishable, Stat},
};

/// A trait for quantum estimators that yield the contribution of
/// the first replica to the observable value.
pub trait LeadingQuantumObservable<T, V, D, B>
where
    D: LeadingExchangePotential<T, V> + Distinguishable,
    B: LeadingExchangePotential<T, V> + Bosonic,
{
    type Output;

    /// Calculates the contribution of this group in the first replica
    /// to the value of the observable, such that the sum of all contributions
    /// off all groups in all replicas is the final value.
    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Self::Output;
}

/// A trait for quantum estimators that yield the contribution of
/// an inner replica to the observable value.
pub trait InnerQuantumObservable<T, V, D, B>
where
    D: InnerExchangePotential<T, V> + Distinguishable,
    B: InnerExchangePotential<T, V> + Bosonic,
{
    type Output;

    /// Calculates the contribution of this group in this replica
    /// to the value of the observable, such that the sum of all contributions
    /// off all groups in all replicas is the final value.
    ///
    /// Returns `None` if this replica does not contribute to the
    /// observable value. This might happen with indistinguishable particles.
    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Option<Self::Output>;
}

/// A trait for quantum estimators that yield the contribution of
/// the last replica to the observable value.
pub trait TrailingQuantumObservable<T, V, D, B>
where
    D: TrailingExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + Bosonic,
{
    type Output;

    /// Calculates the contribution of this group in the last replica
    /// to the value of the observable, such that the sum of all contributions
    /// off all groups in all replicas is the final value.
    ///
    /// Returns `None` if the last replica does not contribute to the
    /// observable value. This might happen with indistinguishable particles.
    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        last_replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Option<Self::Output>;
}

impl<T, V, D, B, U> LeadingQuantumObservable<T, V, D, B> for U
where
    D: LeadingExchangePotential<T, V> + InnerExchangePotential<T, V> + Distinguishable,
    B: LeadingExchangePotential<T, V> + InnerExchangePotential<T, V> + Bosonic,
    U: InnerQuantumObservable<T, V, D, B> + InnerIsLeading,
{
    type Output = <Self as InnerQuantumObservable<T, V, D, B>>::Output;

    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Self::Output {
        InnerQuantumObservable::calculate(
            self,
            exchange_potential,
            0,
            groups,
            group_idx,
            positions,
            forces,
        )
        .expect("The first replica always contributes to a quantum observable")
    }
}

impl<T, V, D, B, U> TrailingQuantumObservable<T, V, D, B> for U
where
    D: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Distinguishable,
    B: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V> + Bosonic,
    U: InnerQuantumObservable<T, V, D, B> + InnerIsTrailing,
{
    type Output = <Self as InnerQuantumObservable<T, V, D, B>>::Output;

    fn calculate(
        &mut self,
        exchange_potential: &Stat<D, B>,
        last_replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Option<Self::Output> {
        InnerQuantumObservable::calculate(
            self,
            exchange_potential,
            last_replica,
            groups,
            group_idx,
            positions,
            forces,
        )
    }
}
