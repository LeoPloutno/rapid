use lib::{
    core::{
        GroupInTypeInImageInSystem, Synchronizer,
        marker::MeaningfulOutput,
        sync_ops::{SyncAddReceiver, SyncAddSender},
    },
    estimator::classical::ClassicalEstimator,
};
use num::Float;

pub struct ExchangePotentialEnergyTrailing;

impl<T, V, A, M> ClassicalEstimator<T, V, A, M, ()> for ExchangePotentialEnergyTrailing
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

pub struct ExchangePotentialEnergyTypeLeader;

impl<T, V, A, M> ClassicalEstimator<T, V, A, M, ()> for ExchangePotentialEnergyTypeLeader
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
        type_exchange_potential_energy: T,
        _group_heat: T,
        _positions: &GroupInTypeInImageInSystem<V>,
        _momenta: &GroupInTypeInImageInSystem<V>,
        _physical_forces: &GroupInTypeInImageInSystem<V>,
        _exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<(), Self::Error> {
        adder.send(type_exchange_potential_energy)?;
        Ok(())
    }
}

pub struct ExchangePotentialEnergyOutput;

impl<T, V, A, M> ClassicalEstimator<T, V, A, M, T> for ExchangePotentialEnergyOutput
where
    T: Float + MeaningfulOutput,
    A: SyncAddReceiver<T> + ?Sized,
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
        type_exchange_potential_energy: T,
        _group_heat: T,
        _positions: &GroupInTypeInImageInSystem<V>,
        _momenta: &GroupInTypeInImageInSystem<V>,
        _physical_forces: &GroupInTypeInImageInSystem<V>,
        _exchange_forces: &GroupInTypeInImageInSystem<V>,
    ) -> Result<T, Self::Error> {
        Ok(match adder.recv_sum()? {
            Some(other_types_exchange_energy) => other_types_exchange_energy + type_exchange_potential_energy,
            None => type_exchange_potential_energy,
        })
    }
}
