use crate::{
    core::{GroupImageHandle, GroupTypeHandle},
    marker::{InnerIsLeading, InnerIsTrailing},
    potential::{
        exchange::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential},
        physical::PhysicalPotential,
    },
    stat::{Bosonic, Distinguishable, Stat},
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
    type Error: From<Therm::Error>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the first image
    /// to the physical and exchange potential energies.
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_momenta: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_physical_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_exchange_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
    ) -> Result<(T, T), Self::Error>;
}

/// A trait for a propagator of a group in an inner image.
pub trait InnerPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: InnerExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    type Error: From<Therm::Error>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in this image
    /// to the physical and exchange potential energies.
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_momenta: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_physical_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_exchange_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
    ) -> Result<(T, T), Self::Error>;
}

/// A trait for a propagator of a group in the last image.
pub trait TrailingPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    type Error: From<Therm::Error>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the last image
    /// to the physical and exchange potential energies.
    #[must_use]
    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_momenta: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_physical_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_exchange_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
    ) -> Result<(T, T), Self::Error>;
}

impl<T, V, Phys, Dist, Boson, Therm, U> LeadingPropagator<T, V, Phys, Dist, Boson, Therm> for U
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: InnerExchangePotential<T, V> + LeadingExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + LeadingExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    U: InnerPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsLeading,
{
    type Error = <Self as InnerPropagator<T, V, Phys, Dist, Boson, Therm>>::Error;

    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_momenta: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_physical_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_exchange_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
    ) -> Result<(T, T), Self::Error> {
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

impl<T, V, Phys, Dist, Boson, Therm, U> TrailingPropagator<T, V, Phys, Dist, Boson, Therm> for U
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: InnerExchangePotential<T, V> + TrailingExchangePotential<T, V> + Distinguishable + ?Sized,
    Boson: InnerExchangePotential<T, V> + TrailingExchangePotential<T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    U: InnerPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsTrailing,
{
    type Error = <Self as InnerPropagator<T, V, Phys, Dist, Boson, Therm>>::Error;

    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        groups_positions: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_momenta: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_physical_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
        groups_exchange_forces: &mut GroupImageHandle<GroupTypeHandle<V>>,
    ) -> Result<(T, T), Self::Error> {
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
