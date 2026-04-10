//! Traits for propagating the system using an exchange potential
//! expanded to the second order.

use crate::{
    ImageHandle,
    core::{
        marker::{InnerIsLeading, InnerIsTrailing},
        stat::{Bosonic, Distinguishable, Stat},
    },
    potential::{
        exchange::quadratic::{
            InnerQuadraticExpansionExchangePotential, LeadingQuadraticExpansionExchangePotential,
            TrailingQuadraticExpansionExchangePotential,
        },
        physical::PhysicalPotential,
    },
    thermostat::Thermostat,
};

/// A trait for a propagator of a group in the first image.
/// Uses quadratic expansion exchange potentials instead of regular ones.
pub trait LeadingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
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
        groups_positions: &mut ImageHandle<V>,
        groups_momenta: &mut ImageHandle<V>,
        groups_physical_forces: &mut ImageHandle<V>,
        groups_exchange_forces: &mut ImageHandle<V>,
    ) -> Result<(T, T, T), Self::Error>;
}

/// A trait for a propagator of a group in an inner image.
/// Uses quadratic expansion exchange potentials instead of regular ones.
pub trait InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
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
        groups_positions: &mut ImageHandle<V>,
        groups_momenta: &mut ImageHandle<V>,
        groups_physical_forces: &mut ImageHandle<V>,
        groups_exchange_forces: &mut ImageHandle<V>,
    ) -> Result<(T, T, T), Self::Error>;
}

/// A trait for a propagator of a group in the last image.
/// Uses quadratic expansion exchange potentials instead of regular ones.
pub trait TrailingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
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
        groups_positions: &mut ImageHandle<V>,
        groups_momenta: &mut ImageHandle<V>,
        groups_physical_forces: &mut ImageHandle<V>,
        groups_exchange_forces: &mut ImageHandle<V>,
    ) -> Result<(T, T, T), Self::Error>;
}

impl<T, V, Phys, Dist, Boson, Therm, P> LeadingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> for P
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V>
        + Distinguishable
        + ?Sized,
    Boson: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V>
        + Bosonic
        + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    P: InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsLeading + ?Sized,
{
    type Error = <Self as InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>>::Error;

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
        InnerQuadraticExpansionPropagator::propagate(
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

impl<T, V, Phys, Dist, Boson, Therm, P> TrailingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> for P
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V>
        + Distinguishable
        + ?Sized,
    Boson: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V>
        + Bosonic
        + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    P: InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsTrailing + ?Sized,
{
    type Error = <Self as InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>>::Error;

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
        InnerQuadraticExpansionPropagator::propagate(
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
