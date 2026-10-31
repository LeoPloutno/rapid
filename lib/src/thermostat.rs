//! A trait for thermalizing the system.

use std::{
    error::Error,
    ops::{Deref, DerefMut},
};

use crate::core::GroupInTypeInImageInSystem;
use macros::heavy_computation;

mod atom_decoupled;
pub use atom_decoupled::{AtomDecoupledThermostat, DecoupledThermostat};

/// A trait for thermostats.
///
/// A thermostat is an entity that thermalized a system
/// in the canonical ensemble such that different energies
/// are sampled while keeping the temperature fixed.
pub trait Thermostat<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Performs thermalization of a group.
    ///
    /// Returns the contribution of the group to the
    /// change in the internal energy of the system due to thermalization.
    #[heavy_computation]
    fn thermalize(
        &mut self,
        step_size: T,
        step: usize,
        positions: &GroupInTypeInImageInSystem<T, V>,
        physical_forces: &GroupInTypeInImageInSystem<T, V>,
        exchange_forces: &GroupInTypeInImageInSystem<T, V>,
        momenta: &mut [V],
    ) -> Result<T, Self::Error>;
}

impl<T, V, U> Thermostat<T, V> for U
where
    U: DerefMut<Target: Thermostat<T, V>> + ?Sized,
{
    type Error = <<Self as Deref>::Target as Thermostat<T, V>>::Error;

    #[inline]
    fn thermalize(
        &mut self,
        step_size: T,
        step: usize,
        positions: &GroupInTypeInImageInSystem<T, V>,
        physical_forces: &GroupInTypeInImageInSystem<T, V>,
        exchange_forces: &GroupInTypeInImageInSystem<T, V>,
        momenta: &mut [V],
    ) -> Result<T, Self::Error> {
        <<Self as Deref>::Target as Thermostat<T, V>>::thermalize(
            &mut *self,
            step_size,
            step,
            positions,
            physical_forces,
            exchange_forces,
            momenta,
        )
    }
}
