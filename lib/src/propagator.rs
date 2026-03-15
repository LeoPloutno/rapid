//! Traits for propagating the system.

use crate::{
    ImageHandle,
    core::{
        marker::{InnerIsLeading, InnerIsTrailing},
        stat::{Bosonic, Distinguishable, Stat},
    },
    potential::{
        exchange::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential},
        physical::PhysicalPotential,
    },
    thermostat::Thermostat,
};

pub mod quadratic;

/// A trait for a propagator of a group in the first image.
pub trait LeadingPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the first image
    /// to the physical and exchange potential energies,
    /// as well as the heat absorbed by the system from the thermostat.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut ImageHandle<V>,
        groups_momenta: &mut ImageHandle<V>,
        groups_physical_forces: &mut ImageHandle<V>,
        groups_exchange_forces: &mut ImageHandle<V>,
    ) -> Result<(T, T, T), Self::Error>;
}

/// A trait for a propagator of a group in an inner image.
pub trait InnerPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in this image
    /// to the physical and exchange potential energies,
    /// as well as the heat absorbed by the system from the thermostat.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut ImageHandle<V>,
        groups_momenta: &mut ImageHandle<V>,
        groups_physical_forces: &mut ImageHandle<V>,
        groups_exchange_forces: &mut ImageHandle<V>,
    ) -> Result<(T, T, T), Self::Error>;
}

/// A trait for a propagator of a group in the last image.
pub trait TrailingPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the last image
    /// to the physical and exchange potential energies,
    /// as well as the heat absorbed by the system from the thermostat.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut ImageHandle<V>,
        groups_momenta: &mut ImageHandle<V>,
        groups_physical_forces: &mut ImageHandle<V>,
        groups_exchange_forces: &mut ImageHandle<V>,
    ) -> Result<(T, T, T), Self::Error>;
}

impl<T, V, Phys, Dist, Boson, Therm, P> LeadingPropagator<T, V, Phys, Dist, Boson, Therm> for P
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: InnerExchangePotential<T, V> + LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    P: InnerPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsLeading + ?Sized,
{
    type Error = <Self as InnerPropagator<T, V, Phys, Dist, Boson, Therm>>::Error;

    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut ImageHandle<V>,
        groups_momenta: &mut ImageHandle<V>,
        groups_physical_forces: &mut ImageHandle<V>,
        groups_exchange_forces: &mut ImageHandle<V>,
    ) -> Result<(T, T, T), Self::Error> {
        InnerPropagator::propagate(
            self,
            step,
            physical_potential,
            exchange_potential,
            thermostat,
            groups_positions,
            groups_momenta,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}

impl<T, V, Phys, Dist, Boson, Therm, P> TrailingPropagator<T, V, Phys, Dist, Boson, Therm> for P
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: InnerExchangePotential<T, V> + TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    P: InnerPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsTrailing + ?Sized,
{
    type Error = <Self as InnerPropagator<T, V, Phys, Dist, Boson, Therm>>::Error;

    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut ImageHandle<V>,
        groups_momenta: &mut ImageHandle<V>,
        groups_physical_forces: &mut ImageHandle<V>,
        groups_exchange_forces: &mut ImageHandle<V>,
    ) -> Result<(T, T, T), Self::Error> {
        InnerPropagator::propagate(
            self,
            step,
            physical_potential,
            exchange_potential,
            thermostat,
            groups_positions,
            groups_momenta,
            groups_physical_forces,
            groups_exchange_forces,
        )
    }
}
