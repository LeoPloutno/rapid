use arc_rw_lock::{ElementRwLock, UniqueArcSliceRwLock};

use crate::{
    core::AtomGroupInfo,
    potential::{
        exchange::quadratic::{
            InnerQuadraticExpansionExchangePotential, LeadingQuadraticExpansionExchangePotential,
            TrailingQuadraticExpansionExchangePotential,
        },
        physical::PhysicalPotential,
    },
    stat::{Bosonic, Distinguishable, Stat},
    sync_ops::{SyncAddRecv, SyncAddSend},
    thermostat::Thermostat,
};

/// A trait for a propagator of a group in the first replica.
pub trait LeadingPropagator<T, V, D, B, A>
where
    D: LeadingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: LeadingQuadraticExpansionExchangePotential<T, V> + Bosonic,
    A: SyncAddRecv<T>,
{
    /// Propagates the positions, momenta, and forces by a single step.
    fn propagate(
        &mut self,
        step: usize,
        step_size: T,
        replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        phys_potential: &mut dyn PhysicalPotential<T, V>,
        groups_exch_potentials: &mut [Stat<D, B>],
        thermostat: &mut dyn Thermostat<T, V>,
        positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(T, T), A::Error>;
}

/// A trait for a propagator of a group in an inner replica.
pub trait InnerPropagator<T, V, D, B, A>
where
    D: InnerQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: InnerQuadraticExpansionExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T>,
{
    /// Propagates the positions, momenta, and forces by a single step.
    fn propagate(
        &mut self,
        step: usize,
        step_size: T,
        replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        phys_potential: &mut dyn PhysicalPotential<T, V>,
        groups_exch_potentials: &mut [Stat<D, B>],
        thermostat: &mut dyn Thermostat<T, V>,
        positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(T, T), A::Error>;
}

/// A trait for a propagator of a group in the last replica.
pub trait TrailingPropagator<T, V, D, B, A>
where
    D: TrailingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: TrailingQuadraticExpansionExchangePotential<T, V> + Bosonic,
    A: SyncAddSend<T>,
{
    /// Propagates the positions, momenta, and forces by a single step.
    fn propagate(
        &mut self,
        step: usize,
        step_size: T,
        replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        phys_potential: &mut dyn PhysicalPotential<T, V>,
        groups_exch_potentials: &mut [Stat<D, B>],
        thermostat: &mut dyn Thermostat<T, V>,
        positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(T, T), A::Error>;
}
