use arc_rw_lock::{ElementRwLock, UniqueArcSliceRwLock};

/// A trait for thermostats.
///
/// A thermostat is an entity that thermalized a system
/// in the canonical ensemble such that different energies
/// are sampled while keeping the temperature fixed.
pub trait Thermostat<T, V> {
    type Error;

    /// Performs thermalization of the system at a given step.
    ///
    /// Returns the contribution of this group in this replica to the
    /// change in the internal energy of the system.
    #[must_use]
    fn thermalize(
        &mut self,
        step: usize,
        replicas_groups_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        replicas_groups_forces: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_momenta: &mut [V],
    ) -> Result<T, Self::Error>;
}
