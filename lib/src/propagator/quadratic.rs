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
pub trait LeadingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    type Error: From<Therm::Error>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the first image
    /// to the physical and exchange potential energies.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
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
pub trait InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    type Error: From<Therm::Error>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the first image
    /// to the physical and exchange potential energies.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
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
pub trait TrailingQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm>
where
    Phys: PhysicalPotential<T, V> + ?Sized,
    Dist: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Distinguishable + ?Sized,
    Boson: for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
{
    type Error: From<Therm::Error>;

    /// Propagates the positions, momenta, and forces by a single step.
    ///
    /// Returns the contribution of this group in the first image
    /// to the physical and exchange potential energies.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
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
    Dist: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V>
        + Distinguishable
        + ?Sized,
    Boson: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> LeadingQuadraticExpansionExchangePotential<'a, T, V>
        + Bosonic
        + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    U: InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsLeading + ?Sized,
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
    Dist: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V>
        + Distinguishable
        + ?Sized,
    Boson: for<'a> InnerQuadraticExpansionExchangePotential<'a, T, V>
        + for<'a> TrailingQuadraticExpansionExchangePotential<'a, T, V>
        + Bosonic
        + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    U: InnerQuadraticExpansionPropagator<T, V, Phys, Dist, Boson, Therm> + InnerIsTrailing + ?Sized,
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
