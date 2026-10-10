use lib::{
    core::{
        GroupRwLockInTypeInImageInSystem, MapInWhole, MapOutsideWhole, Vector,
        error::{EmptyError, InvalidIndexError, InvalidRangeError},
        marker::ValidOutput,
        stat::{Bosonic, Distinguishable, Stat},
    },
    potential::{exchange::ExchangePotential, physical::PhysicalPotential},
    propagator::Propagator,
    thermostat::Thermostat,
    zip_items, zip_iterators,
};
use num::Float;
use std::sync::{Arc, Barrier};

mod error {
    use std::sync::PoisonError;

    use lib::core::error::{AccessError, EmptyError, InvalidIndexError, InvalidRangeError};
    pub enum VelocityVerletError<Phys, Exch, Therm> {
        PhysicalPotential(Phys),
        ExchangePotential(Exch),
        Thermostat(Therm),
        Access(AccessError),
        Poison,
    }

    impl<Phys, Exch, Therm> From<InvalidIndexError> for VelocityVerletError<Phys, Exch, Therm> {
        fn from(value: InvalidIndexError) -> Self {
            Self::Access(AccessError::Index(value))
        }
    }

    impl<Phys, Exch, Therm> From<InvalidRangeError> for VelocityVerletError<Phys, Exch, Therm> {
        fn from(value: InvalidRangeError) -> Self {
            Self::Access(AccessError::Range(value))
        }
    }

    impl<Phys, Exch, Therm> From<EmptyError> for VelocityVerletError<Phys, Exch, Therm> {
        fn from(value: EmptyError) -> Self {
            Self::Access(AccessError::Empty(value))
        }
    }

    impl<Phys, Exch, Therm> From<AccessError> for VelocityVerletError<Phys, Exch, Therm> {
        fn from(value: AccessError) -> Self {
            Self::Access(value)
        }
    }

    impl<Phys, Exch, Therm, G> From<PoisonError<G>> for VelocityVerletError<Phys, Exch, Therm> {
        fn from(_value: PoisonError<G>) -> Self {
            Self::Poison
        }
    }
}
pub use error::VelocityVerletError;

pub struct VelocityVerlet<T> {
    step_size: T,
    mass: T,
    barrier: Arc<Barrier>,
}

impl<T, V, Phys, Dist, Boson, Therm, OutPhys, OutExch> Propagator<T, V, Phys, Dist, Boson, Therm, OutPhys, OutExch>
    for VelocityVerlet<T>
where
    T: From<f32> + Float,
    V: Vector<Element = T>,
    Phys: PhysicalPotential<T, V, OutPhys> + ?Sized,
    Dist: ExchangePotential<T, V, OutExch> + Distinguishable + ?Sized,
    Boson: ExchangePotential<T, V, OutExch> + Bosonic + ?Sized,
    Therm: Thermostat<T, V> + ?Sized,
    OutPhys: ValidOutput<T>,
    OutExch: ValidOutput<T>,
{
    type Error = VelocityVerletError<Phys::Error, Stat<Dist::Error, Boson::Error>, Therm::Error>;

    fn propagate(
        &mut self,
        step: usize,
        physical_potential: &mut Phys,
        mut exchange_potential: Stat<&mut Dist, &mut Boson>,
        thermostat: &mut Therm,
        positions: &mut GroupRwLockInTypeInImageInSystem<V>,
        momenta: &mut GroupRwLockInTypeInImageInSystem<V>,
        physical_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
        exchange_forces: &mut GroupRwLockInTypeInImageInSystem<V>,
    ) -> Result<(OutPhys, OutExch, T), Self::Error> {
        let mut momenta_lock_guard = momenta.write();
        let momenta = &mut *momenta_lock_guard.write();
        let mut heat = thermostat
            .thermalize(
                self.step_size * 0.5.into(),
                step,
                &positions.as_map_ref().map_map(|lock| lock.read()),
                &physical_forces.as_map_ref().map_map(|lock| lock.read()),
                &exchange_forces.as_map_ref().map_map(|lock| lock.read()),
                momenta,
            )
            .map_err(VelocityVerletError::Thermostat)?;

        for zip_items!(momentum, &physical_force, &exchange_force) in
            zip_iterators!(momenta, physical_forces.read().read(), exchange_forces.read().read())
        {
            *momentum = *momentum + (physical_force + exchange_force) * self.step_size * 0.5.into()
        }

        {
            let mut positions_guard = positions.write();
            for zip_items!(position, &momentum) in zip_iterators!(&mut *positions_guard.write(), &*momenta) {
                *position = *position + momentum * self.step_size / self.mass;
            }
        }

        let physical_potential_energy = {
            let mut physical_forces_guard = physical_forces.write();
            physical_potential.calculate_energy_set_forces(
                &positions
                    .as_map_ref()
                    .map_map(|map| map.read())
                    .map_whole(|whole| whole.into()),
                &mut *physical_forces_guard.write(),
            )
        }
        .map_err(VelocityVerletError::PhysicalPotential)?;

        let exchange_potential_energy = {
            let group_index = positions.element_offset();
            let (type_in_prev_image, type_in_next_image) = {
                let types = positions.whole.get_whole().len();
                let type_index = positions.whole.element_offset();
                let (prev_image, next_image) = match (
                    positions.whole.get_whole().before(),
                    positions.whole.get_whole().after(),
                ) {
                    (&[], &[]) => Err(EmptyError)?,
                    (&[], rest) | (rest, &[]) => {
                        let prev_image_start = rest.len() - types;
                        (
                            rest.get(prev_image_start..).ok_or_else(|| {
                                InvalidRangeError::new((prev_image_start..rest.len()).into(), rest.len())
                            })?,
                            rest.get(..types)
                                .ok_or_else(|| InvalidRangeError::new((0..types).into(), rest.len()))?,
                        )
                    }
                    (before, after) => {
                        let prev_image_start = before.len() - types;
                        (
                            before.get(prev_image_start..).ok_or_else(|| {
                                InvalidRangeError::new((prev_image_start..before.len()).into(), before.len())
                            })?,
                            after
                                .get(..types)
                                .ok_or_else(|| InvalidRangeError::new((0..types).into(), after.len()))?,
                        )
                    }
                };
                (
                    MapInWhole::with_element_offset(prev_image, type_index)
                        .ok_or_else(|| InvalidIndexError::new(type_index, prev_image.len()))?,
                    MapInWhole::with_element_offset(next_image, type_index)
                        .ok_or_else(|| InvalidIndexError::new(type_index, next_image.len()))?,
                )
            };

            self.barrier.wait();
            let prev_image_type_guard = type_in_prev_image.read()?;
            let prev_image_positions = MapOutsideWhole {
                map: prev_image_type_guard
                    .get(group_index)
                    .ok_or_else(|| InvalidIndexError::new(group_index, prev_image_type_guard.len()))?,
                whole: type_in_prev_image,
            };

            let next_image_type_guard = type_in_next_image.read()?;
            let next_image_positions = MapOutsideWhole {
                map: next_image_type_guard
                    .get(group_index)
                    .ok_or_else(|| InvalidIndexError::new(group_index, next_image_type_guard.len()))?,
                whole: type_in_next_image,
            };

            let mut exchange_forces_guard = exchange_forces.write();
            exchange_potential.calculate_energy_set_forces(
                &prev_image_positions,
                &positions
                    .as_map_ref()
                    .map_map(|map| map.read())
                    .map_whole(|whole| whole.into()),
                &next_image_positions,
                &mut *exchange_forces_guard.write(),
            )
        }
        .map_err(VelocityVerletError::ExchangePotential)?;

        for zip_items!(momentum, &physical_force, &exchange_force) in
            zip_iterators!(momenta, physical_forces.read().read(), exchange_forces.read().read())
        {
            *momentum = *momentum + (physical_force + exchange_force) * self.step_size * 0.5.into()
        }

        heat = heat
            + thermostat
                .thermalize(
                    self.step_size * 0.5.into(),
                    step,
                    &positions.as_map_ref().map_map(|lock| lock.read()),
                    &physical_forces.as_map_ref().map_map(|lock| lock.read()),
                    &exchange_forces.as_map_ref().map_map(|lock| lock.read()),
                    momenta,
                )
                .map_err(VelocityVerletError::Thermostat)?;

        Ok((physical_potential_energy, exchange_potential_energy, heat))
    }
}
