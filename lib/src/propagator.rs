use arc_rw_lock::{ElementRwLock, UniqueArcSliceRwLock};

use crate::{
    potential::{
        exchange::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential},
        physical::PhysicalPotential,
    },
    stat::{Bosonic, Distinguishable, Stat},
    thermostat::Thermostat,
};

pub mod quadratic;

/// A trait for a propagator of a group in the first replica.
pub trait LeadingPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable,
    Boson: LeadingExchangePotential<T, V> + Bosonic,
    Therm: Thermostat<T, V> + ?Sized,
{
    type Error: From<Therm::Error>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the first replica
    /// to the physical and exchange potential energies.
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: &mut Stat<Dist, Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(T, T), Self::Error>;
}

/// A trait for a propagator of a group in an inner replica.
pub trait InnerPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable,
    Boson: InnerExchangePotential<T, V> + Bosonic,
    Therm: Thermostat<T, V> + ?Sized,
{
    type Error: From<Therm::Error>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in this replica
    /// to the physical and exchange potential energies.
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: &mut Stat<Dist, Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(T, T), Self::Error>;
}

/// A trait for a propagator of a group in the last replica.
pub trait TrailingPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable,
    Boson: TrailingExchangePotential<T, V> + Bosonic,
    Therm: Thermostat<T, V> + ?Sized,
{
    type Error: From<Therm::Error>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the last replica
    /// to the physical and exchange potential energies.
    #[must_use]
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: &mut Stat<Dist, Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_momenta: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_physical_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
        groups_exchange_forces: &mut ElementRwLock<UniqueArcSliceRwLock<V>>,
    ) -> Result<(T, T), Self::Error>;
}
