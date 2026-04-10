//! A trait for thermalizing the system.

use crate::ImageHandle;
use arc_rw_lock::ElementRwLock;
use macros::heavy_computation;

mod atom_decoupled;
pub use atom_decoupled::AtomDecoupledThermostat;

/// A trait for thermostats.
///
/// A thermostat is an entity that thermalized a system
/// in the canonical ensemble such that different energies
/// are sampled while keeping the temperature fixed.
pub trait Thermostat<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Performs thermalization of the system.
    ///
    /// Returns the contribution of this group in this image to the
    /// change in the internal energy of the system due to thermalization.
    #[heavy_computation]
    fn thermalize(
        &mut self,
        step_size: T,
        images_groups_positions: &ElementRwLock<ImageHandle<V>>,
        images_groups_physical_forces: &ElementRwLock<ImageHandle<V>>,
        images_groups_exchange_forces: &ElementRwLock<ImageHandle<V>>,
        group_momenta: &mut [V],
    ) -> Result<T, Self::Error>;
}
