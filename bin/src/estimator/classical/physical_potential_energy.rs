use lib::{
    core::{
        GroupInTypeInImageInSystem, Synchronizer,
        marker::MeaningfulOutput,
        sync_ops::{SyncAddReceiver, SyncAddSender},
    },
    estimator::classical::ClassicalEstimator,
};
use num::Float;

pub struct PhysicalPotentialEnergyTrailing;

impl<T, V, A, M> ClassicalEstimator<T, V, A, M, ()> for PhysicalPotentialEnergyTrailing
where
    A: SyncAddSender<T> + ?Sized,
    M: ?Sized,
{
    type Output = T;
    type Error = A::Error;

    fn calculate(
        &mut self,
        _system_synchronizer: &Synchronizer<T>,
        _image_synchronizer: &Synchronizer<T>,
        adder: &mut A,
        _multiplier: &mut M,
        _physical_potential_energy: T,
        _type_exchange_potential_energy: T,
        _group_heat: T,
        _positions: &GroupInTypeInImageInSystem<V>,
        _momenta: &GroupInTypeInImageInSystem<V>,
        _physical_forces: &GroupInTypeInImageInSystem<V>,
        _exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error> {
        adder.send_empty()?;
        Ok(())
    }
}

pub struct PhysicalPotentialEnergyImageLeader;

impl<T, V, A, M> ClassicalEstimator<T, V, A, M, ()> for PhysicalPotentialEnergyImageLeader
where
    A: SyncAddSender<T> + ?Sized,
    M: ?Sized,
{
    type Output = T;
    type Error = A::Error;

    fn calculate(
        &mut self,
        _system_synchronizer: &Synchronizer<T>,
        _image_synchronizer: &Synchronizer<T>,
        adder: &mut A,
        _multiplier: &mut M,
        physical_potential_energy: T,
        _type_exchange_potential_energy: T,
        _group_heat: T,
        _positions: &GroupInTypeInImageInSystem<V>,
        _momenta: &GroupInTypeInImageInSystem<V>,
        _physical_forces: &GroupInTypeInImageInSystem<V>,
        _exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error> {
        adder.send(physical_potential_energy)?;
        Ok(())
    }
}

pub struct PhysicalPotentialEnergyOutput;

impl<T, V, A, M> ClassicalEstimator<T, V, A, M, T> for PhysicalPotentialEnergyOutput
where
    T: Float + MeaningfulOutput,
    A: SyncAddReceiver<T> + ?Sized,
    M: ?Sized,
{
    type Error = A::Error;
    type Output = T;

    fn calculate(
        &mut self,
        _system_synchronizer: &Synchronizer<T>,
        _image_synchronizer: &Synchronizer<T>,
        adder: &mut A,
        _multiplier: &mut M,
        physical_potential_energy: T,
        _type_exchange_potential_energy: T,
        _group_heat: T,
        _positions: &GroupInTypeInImageInSystem<V>,
        _momenta: &GroupInTypeInImageInSystem<V>,
        _physical_forces: &GroupInTypeInImageInSystem<V>,
        _exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<T, Self::Error> {
        Ok(match adder.recv_sum()? {
            Some(other_images_energy) => other_images_energy + physical_potential_energy,
            None => physical_potential_energy,
        })
    }
}
