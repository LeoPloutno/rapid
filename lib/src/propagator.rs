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
pub trait LeadingPropagator<T, V, D, B>
where
    D: LeadingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: LeadingQuadraticExpansionExchangePotential<T, V> + Bosonic,
{
    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the first replica
    /// to the physical and exchange potential energies.
    fn propagate(
        &mut self,
        step: usize,
        step_size: T,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        phys_potential: &mut dyn PhysicalPotential<T, V>,
        groups_exch_potentials: &mut [Stat<D, B>],
        thermostat: &mut dyn Thermostat<T, V>,
        positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> (T, T);
}

/// A trait for a propagator of a group in an inner replica.
pub trait InnerPropagator<T, V, D, B>
where
    D: InnerQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: InnerQuadraticExpansionExchangePotential<T, V> + Bosonic,
{
    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in this replica
    /// to the physical and exchange potential energies.
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
    ) -> (T, T);
}

/// A trait for a propagator of a group in the last replica.
pub trait TrailingPropagator<T, V, D, B>
where
    D: TrailingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: TrailingQuadraticExpansionExchangePotential<T, V> + Bosonic,
{
    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the last replica
    /// to the physical and exchange potential energies.
    fn propagate(
        &mut self,
        step: usize,
        step_size: T,
        last_replica: usize,
        groups: &[AtomGroupInfo<T>],
        group_idx: usize,
        phys_potential: &mut dyn PhysicalPotential<T, V>,
        groups_exch_potentials: &mut [Stat<D, B>],
        thermostat: &mut dyn Thermostat<T, V>,
        positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> (T, T);
}
