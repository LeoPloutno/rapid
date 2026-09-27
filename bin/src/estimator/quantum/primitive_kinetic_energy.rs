use std::{
    convert::Infallible,
    ops::{Mul, Sub},
};

use lib::{
    core::{GroupInTypeInImage, Synchronizer, marker::MeaningfulOutput},
    estimator::quantum::QuantumEstimator,
};

use crate::core::constants::BOLTZMANN_CONSTANT;

pub struct PrimitiveKineticEnergy<T> {
    constant: T,
}

impl<T> PrimitiveKineticEnergy<T>
where
    T: PartialOrd + Mul<Output = T> + Clone + From<f32>,
{
    pub fn new(dimension: usize, images: usize, atoms: usize, temperature: T) -> Self {
        assert!(temperature.clone() > 0.0.into(), "the temperature must be positive");
        Self {
            constant: temperature * (-0.5 * BOLTZMANN_CONSTANT * ((images * images * atoms * dimension) as f32)).into(),
        }
    }
}

impl<T, V, A, M> QuantumEstimator<T, V, A, M, ()> for PrimitiveKineticEnergy<T> {
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

impl<T, V, A, M> QuantumEstimator<T, V, A, M, T> for PrimitiveKineticEnergy<T>
where
    T: Sub<Output = T> + Clone + MeaningfulOutput,
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
        _positions: &GroupInTypeInImage<V>,
        _physical_forces: &GroupInTypeInImage<V>,
        _exchange_forces: &GroupInTypeInImage<V>,
    ) -> Result<T, Self::Error> {
        Ok(self.constant.clone() - type_exchange_potential_energy)
    }
}
