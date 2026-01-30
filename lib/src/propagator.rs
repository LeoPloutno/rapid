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
    thermostat::Thermostat,
};

/// A trait for a propagator of a group in the first replica.
pub trait LeadingPropagator<T, V, D, B, E>
where
    D: LeadingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: LeadingQuadraticExpansionExchangePotential<T, V> + Bosonic,
{
    type Error: From<E>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the first replica
    /// to the physical and exchange potential energies.
    fn propagate(
        &mut self,
        step: usize,
        step_size: T,
        group_idx: usize,
        groups: &[AtomGroupInfo<T>],
        physical_potential: &mut dyn PhysicalPotential<T, V>,
        groups_exchange_potentials: &mut [Stat<D, B>],
        thermostat: &mut dyn Thermostat<T, V, Error = E>,
        positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(T, T), Self::Error>;
}

/// A trait for a propagator of a group in an inner replica.
pub trait InnerPropagator<T, V, D, B, E>
where
    D: InnerQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: InnerQuadraticExpansionExchangePotential<T, V> + Bosonic,
{
    type Error: From<E>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in this replica
    /// to the physical and exchange potential energies.
    fn propagate(
        &mut self,
        step: usize,
        step_size: T,
        replica: usize,
        group_idx: usize,
        groups: &[AtomGroupInfo<T>],
        physical_potential: &mut dyn PhysicalPotential<T, V>,
        groups_exchange_potentials: &mut [Stat<D, B>],
        thermostat: &mut dyn Thermostat<T, V, Error = E>,
        positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(T, T), Self::Error>;
}

/// A trait for a propagator of a group in the last replica.
pub trait TrailingPropagator<T, V, D, B, E>
where
    D: TrailingQuadraticExpansionExchangePotential<T, V> + Distinguishable,
    B: TrailingQuadraticExpansionExchangePotential<T, V> + Bosonic,
{
    type Error: From<E>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the last replica
    /// to the physical and exchange potential energies.
    #[must_use]
    fn propagate(
        &mut self,
        step: usize,
        step_size: T,
        last_replica: usize,
        group_idx: usize,
        groups: &[AtomGroupInfo<T>],
        physical_potential: &mut dyn PhysicalPotential<T, V>,
        groups_exchange_potentials: &mut [Stat<D, B>],
        thermostat: &mut dyn Thermostat<T, V, Error = E>,
        positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(T, T), Self::Error>;
}
