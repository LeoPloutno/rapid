//! Traits for propagating the system.

use crate::{
    core::{
        AtomGroupRwLock, AtomTypeReaderLock, MapInWhole, MapOutsideWhole,
        stat::{Bosonic, Distinguishable, Stat},
    },
    potential::{exchange::ExchangePotential, physical::PhysicalPotential},
    thermostat::Thermostat,
};
use macros::heavy_computation;

pub mod quadratic;

pub type GroupRwLockInTypeInImageInSystem<'a, V> = MapOutsideWhole<
    &'a mut AtomGroupRwLock<V>,
    MapInWhole<
        &'a AtomTypeReaderLock<V>,
        MapInWhole<&'a [AtomTypeReaderLock<V>], &'a [AtomTypeReaderLock<V>]>,
    >,
>;

/// A trait for a propagator of a group in an image.
pub trait Propagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: ExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in this image
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
