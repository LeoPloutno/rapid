use lib::{
    core::{GroupInTypeInImage, Synchronizer, marker::MeaningfulOutput},
    estimator::quantum::QuantumEstimator,
};
use std::convert::Infallible;

pub struct PotentialEnergy;

impl<T, V, A: ?Sized, M: ?Sized> QuantumEstimator<T, V, A, M, ()> for PotentialEnergy {
    type Output = T;
    type Error = Infallible;

    fn calculate(
        &mut self,
        _synchronizer: &Synchronizer<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        _physical_potential_energy: T,
        _type_exchange_potential_energy: T,
        _positions: &GroupInTypeInImage<V>,
        _physical_forces: &GroupInTypeInImage<V>,
        _exchange_forces: &GroupInTypeInImage<V>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<T: MeaningfulOutput, V, A: ?Sized, M: ?Sized> QuantumEstimator<T, V, A, M, T> for PotentialEnergy {
    type Output = T;
    type Error = Infallible;

    fn calculate(
        &mut self,
        _synchronizer: &Synchronizer<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        physical_potential_energy: T,
        _type_exchange_potential_energy: T,
        _positions: &GroupInTypeInImage<V>,
        _physical_forces: &GroupInTypeInImage<V>,
        _exchange_forces: &GroupInTypeInImage<V>,
    ) -> Result<T, Self::Error> {
        Ok(physical_potential_energy)
    }
}
