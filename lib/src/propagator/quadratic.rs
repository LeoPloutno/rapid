//! Traits for propagating the system using an exchange potential
//! expanded to the second order.

use super::GroupRwLockInTypeInImageInSystem;
use crate::{
    core::stat::{Bosonic, Distinguishable, Stat},
    potential::{
        exchange::quadratic::QuadraticExpansionExchangePotential, physical::PhysicalPotential,
    },
    thermostat::Thermostat,
};
use macros::heavy_computation;

/// A trait for a propagator of a group in the first image.
/// Uses quadratic expansion exchange potentials instead of regular ones.
pub trait QuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: for<'a> QuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the first image
    /// to the physical and exchange potential energies,
    /// as well as the heat absorbed by the system from the thermostat.
    #[heavy_computation]
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        positions: &mut GroupRwLockInTypeInImageInSystem<V>,
        momenta: &mut GroupRwLockInTypeInImageInSystem<V>,
        physical_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
        exchange_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
    ) -> Result<(T, T, T), Self::Error>;
}
