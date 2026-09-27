use lib::{
    core::{Vector, error::EmptyError},
    estimator::quantum::{AdditiveValueQuantumEstimator, AtomAdditiveQuantumEstimator},
};
use std::{
    convert::Infallible,
    ops::{Add, Mul},
};

pub struct VirialKineticEnergy<T> {
    prefactor: T,
}

impl<T: From<f32>> VirialKineticEnergy<T> {
    pub fn new(images: usize) -> AdditiveValueQuantumEstimator<Self> {
        AdditiveValueQuantumEstimator::new(Self {
            prefactor: T::from(-0.5 * (images as f32)),
        })
    }
}

impl<T, V> AtomAdditiveQuantumEstimator<T, V> for VirialKineticEnergy<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone,
    V: Clone + Vector<Element = T>,
{
    type Output = T;
    type AtomError = Infallible;
    type SystemError = EmptyError;

    fn calculate(
        &mut self,
        _atom_index: usize,
        _physical_potential_energy: T,
        _type_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        _exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError> {
        Ok(self.prefactor.clone() * position.clone().dot(physical_force.clone()))
    }
}
