mod distinguishable {
    use std::ops::{Add, Mul, Range};

    use lib::{
        core::{
            Vector,
            error::{AccessError, EmptyIteratorError, InvalidRangeError},
            marker::{InnerIsLeading, InnerIsTrailing},
            stat::Distinguishable,
            zip_items, zip_iterators,
        },
        potential::exchange::InnerExchangePotential,
    };

    use crate::core::constants::{BOLTZMANN_CONSTANT, REDUCED_PLANK_CONSTANT};

    pub struct DistinguishableExchangePotential<const N: usize, T> {
        potential_prefactor: T,
        group_range: Range<usize>,
    }

    impl<const N: usize, T> DistinguishableExchangePotential<N, T>
    where
        T: Clone + From<f32> + PartialOrd + Mul<Output = T>,
    {
        pub fn new(mass: T, temperature: T, inner_images: usize, group_range: Range<usize>) -> Self {
            assert!(mass.clone() > 0.0.into(), "the mass must be positive");
            assert!(temperature.clone() > 0.0.into(), "the temperature must be positive");
            Self {
                potential_prefactor: T::from(
                    0.5 * ((inner_images + 2) as f32) * BOLTZMANN_CONSTANT * BOLTZMANN_CONSTANT
                        / (REDUCED_PLANK_CONSTANT * REDUCED_PLANK_CONSTANT),
                ) * mass
                    * temperature.clone()
                    * temperature,
                group_range,
            }
        }
    }

    impl<const N: usize, T> InnerIsLeading for DistinguishableExchangePotential<N, T> {}

    impl<const N: usize, T> InnerIsTrailing for DistinguishableExchangePotential<N, T> {}

    impl<const N: usize, T> Distinguishable for DistinguishableExchangePotential<N, T> {}

    impl<const N: usize, T, V> InnerExchangePotential<T, V> for DistinguishableExchangePotential<N, T>
    where
        T: Clone + From<f32> + Add<Output = T> + Mul<Output = T>,
        V: Vector<N, Element = T> + Clone,
    {
        type Error = AccessError;

        fn calculate_potential_set_forces(
            &mut self,
            type_positions_prev_image: &[V],
            type_positions_next_image: &[V],
            type_positions: &[V],
            group_forces: &mut [V],
        ) -> Result<T, Self::Error> {
            let mut iter = zip_iterators!(
                group_forces,
                type_positions
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions.len()))?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_prev_image.len()))?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_next_image.len()))?,
            )
            .map(
                |zip_items!(force, position, position_prev_image, position_next_image)| {
                    let connection_prev = position_prev_image.clone() - position.clone();
                    let connection_next = position_next_image.clone() - position.clone();
                    *force = (connection_prev.clone() + connection_next.clone())
                        * 2.0.into()
                        * self.potential_prefactor.clone();
                    self.potential_prefactor.clone()
                        * (connection_prev.clone().magnitude_squared() + connection_next.clone().magnitude_squared())
                },
            );
            let first = iter.next().ok_or(EmptyIteratorError)?;
            Ok(iter.fold(first, |accum, element| accum + element))
        }

        fn calculate_potential_add_forces(
            &mut self,
            type_positions_prev_image: &[V],
            type_positions_next_image: &[V],
            type_positions: &[V],
            group_forces: &mut [V],
        ) -> Result<T, Self::Error> {
            let mut iter = zip_iterators!(
                group_forces,
                type_positions
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions.len()))?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_prev_image.len()))?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_next_image.len()))?,
            )
            .map(
                |zip_items!(force, position, position_prev_image, position_next_image)| {
                    let connection_prev = position_prev_image.clone() - position.clone();
                    let connection_next = position_next_image.clone() - position.clone();
                    *force += (connection_prev.clone() + connection_next.clone())
                        * 2.0.into()
                        * self.potential_prefactor.clone();
                    self.potential_prefactor.clone()
                        * (connection_prev.clone().magnitude_squared() + connection_next.clone().magnitude_squared())
                },
            );
            let first = iter.next().ok_or(EmptyIteratorError)?;
            Ok(iter.fold(first, |accum, element| accum + element))
        }

        fn calculate_potential(
            &mut self,
            type_positions_prev_image: &[V],
            type_positions_next_image: &[V],
            type_positions: &[V],
        ) -> Result<T, Self::Error> {
            let mut iter = zip_iterators!(
                type_positions
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions.len()))?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_prev_image.len()))?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_next_image.len()))?,
            )
            .map(|zip_items!(position, position_prev_image, position_next_image)| {
                self.potential_prefactor.clone()
                    * ((position.clone() - position_prev_image.clone()).magnitude_squared()
                        + (position.clone() - position_next_image.clone()).magnitude_squared())
            });
            let first = iter.next().ok_or(EmptyIteratorError)?;
            Ok(iter.fold(first, |accum, element| accum + element))
        }

        fn set_forces(
            &mut self,
            type_positions_prev_image: &[V],
            type_positions_next_image: &[V],
            type_positions: &[V],
            group_forces: &mut [V],
        ) -> Result<(), Self::Error> {
            for zip_items!(force, position, position_prev_image, position_next_image) in zip_iterators!(
                group_forces,
                type_positions
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions.len()))?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_prev_image.len()))?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_next_image.len()))?,
            ) {
                *force = (position_prev_image.clone() + position_next_image.clone() - position.clone() * 2.0.into())
                    * 2.0.into()
                    * self.potential_prefactor.clone();
            }
            Ok(())
        }

        fn add_forces(
            &mut self,
            type_positions_prev_image: &[V],
            type_positions_next_image: &[V],
            type_positions: &[V],
            group_forces: &mut [V],
        ) -> Result<(), Self::Error> {
            for zip_items!(force, position, position_prev_image, position_next_image) in zip_iterators!(
                group_forces,
                type_positions
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions.len()))?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_prev_image.len()))?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or_else(|| InvalidRangeError::new(self.group_range.clone(), type_positions_next_image.len()))?,
            ) {
                *force += (position_prev_image.clone() + position_next_image.clone() - position.clone() * 2.0.into())
                    * 2.0.into()
                    * self.potential_prefactor.clone();
            }
            Ok(())
        }
    }
}

pub use distinguishable::DistinguishableExchangePotential;
