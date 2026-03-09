mod harmonic {
    use std::ops::{Div, Mul};

    use lib::{
        core::{Vector, error::InvalidIndexError},
        potential::physical::AtomDecoupledPhysicalPotential,
    };

    pub struct Harmonic<const N: usize, T> {
        potential_prefactor: T,
    }

    impl<const N: usize, T> Harmonic<N, T>
    where
        T: Clone + From<f32> + PartialOrd + Div<Output = T>,
    {
        pub fn new(spring_constant: T, inner_images: usize) -> Self {
            assert!(
                spring_constant.clone() >= 0.0.into(),
                "spring constant must be non-negative"
            );
            Self {
                potential_prefactor: spring_constant / ((inner_images + 2) as f32).into(),
            }
        }
    }

    impl<const N: usize, T, V> AtomDecoupledPhysicalPotential<T, V> for Harmonic<N, T>
    where
        T: Clone + From<f32> + Mul<Output = T>,
        V: Vector<N, Element = T> + Clone,
    {
        type Error = InvalidIndexError;

        fn calculate_potential_set_force(
            &mut self,
            atom_index: usize,
            position: &V,
            force: &mut V,
        ) -> Result<T, Self::Error> {
            #![allow(deprecated)]
            self.set_force(atom_index, position, force)?;
            self.calculate_potential(atom_index, position)
        }

        fn calculate_potential_add_force(
            &mut self,
            atom_index: usize,
            position: &V,
            force: &mut V,
        ) -> Result<T, Self::Error> {
            #![allow(deprecated)]
            self.add_force(atom_index, position, force)?;
            self.calculate_potential(atom_index, position)
        }

        fn calculate_potential(&mut self, _atom_index: usize, position: &V) -> Result<T, Self::Error> {
            Ok(self.potential_prefactor.clone() * position.clone().magnitude_squared())
        }

        fn set_force(&mut self, _atom_index: usize, position: &V, force: &mut V) -> Result<(), Self::Error> {
            *force = -position.clone() * 2.0.into() * self.potential_prefactor.clone();
            Ok(())
        }

        fn add_force(&mut self, _atom_index: usize, position: &V, force: &mut V) -> Result<(), Self::Error> {
            *force += -position.clone() * 2.0.into() * self.potential_prefactor.clone();
            Ok(())
        }
    }
}

pub use harmonic::Harmonic;
