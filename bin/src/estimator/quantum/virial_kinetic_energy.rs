use crate::estimator::EstimatorsError;
use lib::{
    core::Vector,
    estimator::quantum::{AdditiveValueQuantumEstimator, AtomAdditiveQuantumEstimator},
};
use num::Float;
use std::convert::Infallible;

pub struct VirialKineticEnergy<T, const N: usize> {
    prefactor: T,
}

impl<T: From<f32>, const N: usize> VirialKineticEnergy<T, N> {
    pub fn new(images: usize) -> AdditiveValueQuantumEstimator<Self> {
        AdditiveValueQuantumEstimator::new(Self {
            prefactor: T::from(-0.5 * (images as f32)),
        })
    }
}

impl<T, V, E, const N: usize> AtomAdditiveQuantumEstimator<T, V, E> for VirialKineticEnergy<T, N>
where
    T: Float,
    V: Vector<N, Element = T>,
    EstimatorsError: From<E>,
{
    type Output = T;
    type AtomError = Infallible;
    type SystemError = EstimatorsError;

    fn calculate(
        &mut self,
        _atom_index: usize,
        _physical_potential_energy: T,
        _type_exchange_potential_energy: T,
        position: &V,
        physical_force: &V,
        _exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError> {
        Ok(self.prefactor * position.dot(*physical_force))
    }
}
