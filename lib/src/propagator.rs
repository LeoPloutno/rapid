//! Traits for propagating the system.

use crate::{
    core::{
        AtomGroupRwLock, AtomTypeReaderLock, MapInWhole, MapOutsideWhole,
        marker::ValidOutput,
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
pub trait Propagator<T, V, Phys, Dist, Boson, Therm, OutPhys, OutExch>
where
    Phys: PhysicalPotential<T, V, OutPhys> + ?Sized,
    Dist: ExchangePotential<T, V, OutExch> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V, OutExch> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    OutPhys: ValidOutput<T>,
    OutExch: ValidOutput<T>,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the physical and exchange potential energies
    /// if `OutPhys` or `OutExch` are `T`.
    /// Also returns the heat absorbed by the group from the thermostat.
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
    ) -> Result<(OutPhys, OutExch, T), Self::Error>;
}
