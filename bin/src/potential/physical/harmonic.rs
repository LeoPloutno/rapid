use lib::{
    core::{Vector, error::AccessError},
    potential::physical::{AdditivePhysicalPotential, AtomAdditivePhysicalPotential},
};
use std::{
    convert::Infallible,
    ops::{Add, Div, Mul},
};

pub struct Harmonic<T> {
    potential_prefactor: T,
}

impl<T> Harmonic<T>
where
    T: Clone + From<f32> + PartialOrd + Div<Output = T>,
{
    pub fn new<A>(spring_constant: T, images: usize, adder: A) -> AdditivePhysicalPotential<Self, A> {
        assert!(
            spring_constant.clone() >= 0.0.into(),
            "spring constant must be non-negative"
        );
        AdditivePhysicalPotential::new(
            adder,
            Self {
                potential_prefactor: spring_constant / (images as f32).into(),
            },
        )
    }
}

impl<T, V> AtomAdditivePhysicalPotential<T, V> for Harmonic<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + From<f32>,
    V: Clone + Vector<Element = T>,
{
    type AtomError = Infallible;
    type SystemError = AccessError;

    #[inline]
    fn calculate_energy_and_force(&mut self, atom_index: usize, position: &V) -> Result<(T, V), Self::AtomError> {
        #[allow(deprecated)]
        Ok((
            self.calculate_energy(atom_index, position)?,
            self.calculate_force(atom_index, position)?,
        ))
    }

    #[inline]
    fn calculate_energy(&mut self, _atom_index: usize, position: &V) -> Result<T, Self::AtomError> {
        Ok(self.potential_prefactor.clone() * position.clone().magnitude_squared())
    }

    #[inline]
    fn calculate_force(&mut self, _atom_index: usize, position: &V) -> Result<V, Self::AtomError> {
        Ok(-position.clone() * 2.0.into() * self.potential_prefactor.clone())
    }
}
