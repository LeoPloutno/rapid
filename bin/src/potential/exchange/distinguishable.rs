use crate::core::constants::{BOLTZMANN_CONSTANT, REDUCED_PLANK_CONSTANT};
use lib::{
    core::{
        GroupInTypeInImage, Vector,
        marker::MeaningfulOutput,
        stat::Distinguishable,
        sync_ops::{SyncAddReceiver, SyncAddSender},
        zip_items, zip_iterators,
    },
    potential::exchange::ExchangePotential,
};
use std::ops::{Add, Mul};

pub struct DistinguishableExchangePotential<const N: usize, T, A> {
    adder: A,
    potential_prefactor: T,
}

impl<const N: usize, T, A> DistinguishableExchangePotential<N, T, A>
where
    T: PartialOrd + Mul<Output = T> + Clone + From<f32>,
{
    pub fn new(images: usize, mass: T, temperature: T, adder: A) -> Self {
        assert!(mass.clone() > 0.0.into(), "the mass must be positive");
        assert!(temperature.clone() > 0.0.into(), "the temperature must be positive");
        Self {
            potential_prefactor: T::from(
                0.5 * (images as f32) * BOLTZMANN_CONSTANT * BOLTZMANN_CONSTANT
                    / (REDUCED_PLANK_CONSTANT * REDUCED_PLANK_CONSTANT),
            ) * mass
                * temperature.clone()
                * temperature,
            adder,
        }
    }
}

impl<const N: usize, T, A> DistinguishableExchangePotential<N, T, A>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + From<f32>,
{
    #[inline]
    fn calculate_energy_group_contribution_set_forces<V>(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> T
    where
        V: Clone + Vector<Element = T>,
    {
        let mut energy = T::from(0.0);
        for zip_items!(force, position, prev_image_position, next_image_position) in zip_iterators!(
            forces,
            positions.read(),
            prev_image_positions.read(),
            next_image_positions.read()
        ) {
            let delta_prev = prev_image_position.clone() - position.clone();
            let delta_next = next_image_position.clone() - position.clone();
            energy = energy.clone() + delta_prev.clone().magnitude_squared() + delta_next.clone().magnitude_squared();
            *force = (delta_prev + delta_next) * self.potential_prefactor.clone() * 2.0.into();
        }
        // We multiply by a half to account for redunduncies across all threads.
        energy * self.potential_prefactor.clone() * 0.5.into()
    }

    #[inline]
    fn calculate_energy_group_contribution_add_forces<V>(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> T
    where
        V: Clone + Vector<Element = T>,
    {
        let mut energy = T::from(0.0);
        for zip_items!(force, position, prev_image_position, next_image_position) in zip_iterators!(
            forces,
            positions.read(),
            prev_image_positions.read(),
            next_image_positions.read()
        ) {
            let delta_prev = prev_image_position.clone() - position.clone();
            let delta_next = next_image_position.clone() - position.clone();
            energy = energy.clone() + delta_prev.clone().magnitude_squared() + delta_next.clone().magnitude_squared();
            *force = force.clone() + (delta_prev + delta_next) * self.potential_prefactor.clone() * 2.0.into();
        }
        // We multiply by a half to account for redunduncies across all threads.
        energy * self.potential_prefactor.clone() * 0.5.into()
    }

    #[inline]
    fn calculate_energy_group_contribution<V>(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
    ) -> T
    where
        V: Clone + Vector<Element = T>,
    {
        let mut energy = T::from(0.0);
        for zip_items!(position, prev_image_position, next_image_position) in zip_iterators!(
            positions.read(),
            prev_image_positions.read(),
            next_image_positions.read(),
        ) {
            energy = energy.clone()
                + (prev_image_position.clone() - position.clone()).magnitude_squared()
                + (next_image_position.clone() - position.clone()).magnitude_squared();
        }
        // We multiply by a half to account for redunduncies across all threads.
        energy * self.potential_prefactor.clone() * 0.5.into()
    }

    #[inline]
    fn set_forces<V>(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) where
        V: Clone + Vector<Element = T>,
    {
        for zip_items!(force, position, prev_image_position, next_image_position) in zip_iterators!(
            forces,
            positions.read(),
            prev_image_positions.read(),
            next_image_positions.read(),
        ) {
            *force = (prev_image_position.clone() + next_image_position.clone() - position.clone() * 2.0.into())
                * self.potential_prefactor.clone()
                * 2.0.into();
        }
    }

    #[inline]
    fn add_forces<V>(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) where
        V: Clone + Vector<Element = T>,
    {
        for zip_items!(force, position, prev_image_position, next_image_position) in zip_iterators!(
            forces,
            positions.read(),
            prev_image_positions.read(),
            next_image_positions.read(),
        ) {
            *force = force.clone()
                + (prev_image_position.clone() + next_image_position.clone() - position.clone() * 2.0.into())
                    * self.potential_prefactor.clone()
                    * 2.0.into();
        }
    }
}

impl<const N: usize, T, A> Distinguishable for DistinguishableExchangePotential<N, T, A> {}

impl<const N: usize, T, V, A> ExchangePotential<T, V, ()> for DistinguishableExchangePotential<N, T, A>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + From<f32>,
    V: Clone + Vector<Element = T>,
    A: SyncAddSender<T>,
{
    type Error = A::Error;

    fn calculate_energy_set_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        let energy = self.calculate_energy_group_contribution_set_forces(
            prev_image_positions,
            next_image_positions,
            positions,
            forces,
        );
        self.adder.send(energy)?;
        Ok(())
    }

    fn calculate_energy_add_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        let energy = self.calculate_energy_group_contribution_add_forces(
            prev_image_positions,
            next_image_positions,
            positions,
            forces,
        );
        self.adder.send(energy)?;
        Ok(())
    }

    fn calculate_energy(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<(), Self::Error> {
        let energy = self.calculate_energy_group_contribution(prev_image_positions, next_image_positions, positions);
        self.adder.send(energy)?;
        Ok(())
    }

    fn set_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        self.set_forces(prev_image_positions, next_image_positions, positions, forces);
        Ok(())
    }

    fn add_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        self.add_forces(prev_image_positions, next_image_positions, positions, forces);
        Ok(())
    }
}

impl<const N: usize, T, V, A> ExchangePotential<T, V, T> for DistinguishableExchangePotential<N, T, A>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + From<f32> + MeaningfulOutput,
    V: Clone + Vector<Element = T>,
    A: SyncAddReceiver<T>,
{
    type Error = A::Error;

    fn calculate_energy_set_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<T, Self::Error> {
        let image_energy = self.calculate_energy_group_contribution_set_forces(
            prev_image_positions,
            next_image_positions,
            positions,
            forces,
        );
        Ok(match self.adder.recv_sum()? {
            Some(other_images_energy) => other_images_energy + image_energy,
            None => image_energy,
        })
    }

    fn calculate_energy_add_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<T, Self::Error> {
        let image_energy = self.calculate_energy_group_contribution_add_forces(
            prev_image_positions,
            next_image_positions,
            positions,
            forces,
        );
        Ok(match self.adder.recv_sum()? {
            Some(other_images_energy) => other_images_energy + image_energy,
            None => image_energy,
        })
    }

    fn calculate_energy(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<T, Self::Error> {
        let image_energy =
            self.calculate_energy_group_contribution(prev_image_positions, next_image_positions, positions);
        Ok(match self.adder.recv_sum()? {
            Some(other_images_energy) => other_images_energy + image_energy,
            None => image_energy,
        })
    }

    fn set_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        self.set_forces(prev_image_positions, next_image_positions, positions, forces);
        Ok(())
    }

    fn add_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        self.add_forces(prev_image_positions, next_image_positions, positions, forces);
        Ok(())
    }
}
