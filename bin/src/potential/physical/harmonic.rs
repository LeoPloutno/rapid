use lib::{
    core::{Vector, error::AccessError},
    potential::physical::{AdditivePhysicalPotential, AtomAdditivePhysicalPotential},
};
use num::Float;
use std::convert::Infallible;

pub struct Harmonic<T> {
    potential_prefactor: T,
}

impl<T> Harmonic<T>
where
    T: From<f32> + Float,
{
    pub fn new<A>(images: usize, spring_constant: T, adder: A) -> AdditivePhysicalPotential<Self, A> {
        assert!(spring_constant >= T::zero(), "spring constant must be non-negative");

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
    T: From<f32> + Float,
    V: Vector<Element = T>,
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
        Ok(self.potential_prefactor * position.magnitude_squared())
    }

    #[inline]
    fn calculate_force(&mut self, _atom_index: usize, position: &V) -> Result<V, Self::AtomError> {
        Ok(-*position * self.potential_prefactor * 2.0.into())
    }
}
