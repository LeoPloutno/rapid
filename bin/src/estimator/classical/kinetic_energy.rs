use std::convert::Infallible;

use lib::{
    core::{Vector, error::EmptyError},
    estimator::classical::{AdditiveValueClassicalEstimator, AtomAdditiveClassicalEstimator},
};
use num::Float;

pub struct KineticEnergy<T> {
    mass: T,
}

impl<T: Float> KineticEnergy<T> {
    pub fn new(mass: T) -> AdditiveValueClassicalEstimator<Self> {
        assert!(mass > T::zero(), "the mass must be positive");

        AdditiveValueClassicalEstimator::new(Self { mass })
    }
}

impl<T, V> AtomAdditiveClassicalEstimator<T, V> for KineticEnergy<T>
where
    T: From<f32> + Float,
    V: Vector<Element = T>,
{
    type Output = T;
    type AtomError = Infallible;
    type SystemError = EmptyError;

    fn calculate(
        &mut self,
        _atom_index: usize,
        _physical_potential_energy: T,
        _type_exchange_potential_energy: T,
        _group_heat: T,
        _position: &V,
        momentum: &V,
        _physical_force: &V,
        _exchange_force: &V,
    ) -> Result<Self::Output, Self::AtomError> {
        Ok(momentum.magnitude_squared() / self.mass * 0.5.into())
    }
}
