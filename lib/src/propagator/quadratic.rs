use crate::{
    core::{GroupImageHandle, GroupTypeHandle},
    marker::{InnerIsLeading, InnerIsTrailing},
    potential::{
        exchange::quadratic::{
            InnerQuadraticExpansionExchangePotential, LeadingQuadraticExpansionExchangePotential,
            TrailingQuadraticExpansionExchangePotential,
        },
        physical::PhysicalPotential,
    },
    stat::{Bosonic, Distinguishable, Stat},
    thermostat::Thermostat,
};

/// A trait for a propagator of a group in the first image.
/// Uses quadratic expansion exchange potentials instead of regular ones.
trait LeadingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable,
    Boson: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic,
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
/// Uses quadratic expansion exchange potentials instead of regular ones.
trait InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable,
    Boson: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic,
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

/// A trait for a propagator of a group in the last image.
/// Uses quadratic expansion exchange potentials instead of regular ones.
trait TrailingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable,
    Boson: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic,
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

impl<T, V, Phys, Dist, Boson, Therm, U> LeadingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> for U
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    for<'a> Dist: InnerQuadraticExpansionExchangePotential<'a, T, V>
        + LeadingQuadraticExpansionExchangePotential<'a, T, V>
        + Distinguishable,
    for<'a> Boson: InnerQuadraticExpansionExchangePotential<'a, T, V>
        + LeadingQuadraticExpansionExchangePotential<'a, T, V>
        + Bosonic,
    Therm: Thermostat<T, V> + ?Sized,
    U: InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsLeading,
{
    type Error = <Self as InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>>::Error;

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

impl<T, V, Phys, Dist, Boson, Therm, U> TrailingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> for U
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    for<'a> Dist: InnerQuadraticExpansionExchangePotential<'a, T, V>
        + TrailingQuadraticExpansionExchangePotential<'a, T, V>
        + Distinguishable,
    for<'a> Boson: InnerQuadraticExpansionExchangePotential<'a, T, V>
        + TrailingQuadraticExpansionExchangePotential<'a, T, V>
        + Bosonic,
    Therm: Thermostat<T, V> + ?Sized,
    U: InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsTrailing,
{
    type Error = <Self as InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>>::Error;

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
