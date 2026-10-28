use crate::core::constants::BOLTZMANN_CONSTANT;
use lib::{
    core::{GroupInTypeInImage, Synchronizer, marker::MeaningfulOutput},
    estimator::quantum::QuantumEstimator,
};
use num::Float;
use std::convert::Infallible;

pub struct PrimitiveKineticEnergy<T> {
    constant: T,
}

impl<T> PrimitiveKineticEnergy<T>
where
    T: From<f32> + Float,
{
    pub fn new(dimension: usize, images: usize, atoms: usize, temperature: T) -> Self {
        assert!(temperature > T::zero(), "the temperature must be positive");

        Self {
            constant: temperature * (-0.5 * BOLTZMANN_CONSTANT * ((images * images * atoms * dimension) as f32)).into(),
        }
    }
}

impl<T, V, A: ?Sized, M: ?Sized> QuantumEstimator<T, V, A, M, ()> for PrimitiveKineticEnergy<T> {
    type Output = T;
    type Error = Infallible;

    fn calculate(
        &mut self,
        _synchronizer: &Synchronizer<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        _physical_potential_energy: T,
        _type_exchange_potential_energy: T,
        _positions: &GroupInTypeInImage<T, V>,
        _physical_forces: &GroupInTypeInImage<T, V>,
        _exchange_forces: &GroupInTypeInImage<T, V>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<T, V, A, M> QuantumEstimator<T, V, A, M, T> for PrimitiveKineticEnergy<T>
where
    T: Float + MeaningfulOutput,
    A: ?Sized,
    M: ?Sized,
{
    type Output = T;
    type Error = Infallible;

    fn calculate(
        &mut self,
        _synchronizer: &Synchronizer<T>,
        _adder: &mut A,
        _multiplier: &mut M,
        _physical_potential_energy: T,
        type_exchange_potential_energy: T,
        _positions: &GroupInTypeInImage<T, V>,
        _physical_forces: &GroupInTypeInImage<T, V>,
        _exchange_forces: &GroupInTypeInImage<T, V>,
    ) -> Result<T, Self::Error> {
        Ok(self.constant - type_exchange_potential_energy)
    }
}
