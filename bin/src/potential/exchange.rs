pub mod distinguishable {
    use std::{
        error::Error,
        fmt::{Display, Formatter, Result as FmtResult},
        ops::{Add, Mul, Range},
    };

    use lib::{
        core::{Vector, zip_items, zip_iterators},
        potential::exchange::InnerExchangePotential,
    };

    pub struct DistinguishableExchangePotential<const N: usize, T> {
        pub potential_prefix: T,
        pub group_range: Range<usize>,
    }

    impl<const N: usize, T, V> InnerExchangePotential<T, V> for DistinguishableExchangePotential<N, T>
    where
        T: From<f32> + Add<Output = T> + Mul<Output = T> + Clone,
        V: Vector<N, Element = T> + Clone,
    {
        type Error = Box<dyn Error>;

        fn calculate_potential_set_forces(
            &mut self,
            type_positions_prev_image: &[V],
            type_positions_next_image: &[V],
            type_positions: &[V],
            group_forces: &mut [V],
        ) -> Result<T, Self::Error> {
            let mut iter = zip_iterators!(
                group_forces,
                type_positions.get(self.group_range.clone()).ok_or(InvalidRangeError {
                    valid: 0..type_positions.len(),
                    invalid: self.group_range.clone(),
                })?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_prev_image.len(),
                        invalid: self.group_range.clone(),
                    })?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_next_image.len(),
                        invalid: self.group_range.clone(),
                    })?
            )
            .map(
                |zip_items!(force, position, position_prev_image, position_next_image)| {
                    let connection_prev = position_prev_image.clone() - position.clone();
                    let connection_next = position_next_image.clone() - position.clone();
                    *force = (connection_prev.clone() + connection_next.clone())
                        * T::from(2.0)
                        * self.potential_prefix.clone();
                    self.potential_prefix.clone()
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
                type_positions.get(self.group_range.clone()).ok_or(InvalidRangeError {
                    valid: 0..type_positions.len(),
                    invalid: self.group_range.clone(),
                })?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_prev_image.len(),
                        invalid: self.group_range.clone(),
                    })?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_next_image.len(),
                        invalid: self.group_range.clone(),
                    })?
            )
            .map(
                |zip_items!(force, position, position_prev_image, position_next_image)| {
                    let connection_prev = position_prev_image.clone() - position.clone();
                    let connection_next = position_next_image.clone() - position.clone();
                    *force += (connection_prev.clone() + connection_next.clone())
                        * T::from(2.0)
                        * self.potential_prefix.clone();
                    self.potential_prefix.clone()
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
                type_positions.get(self.group_range.clone()).ok_or(InvalidRangeError {
                    valid: 0..type_positions.len(),
                    invalid: self.group_range.clone(),
                })?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_prev_image.len(),
                        invalid: self.group_range.clone(),
                    })?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_next_image.len(),
                        invalid: self.group_range.clone(),
                    })?
            )
            .map(|zip_items!(position, position_prev_image, position_next_image)| {
                self.potential_prefix.clone()
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
                type_positions.get(self.group_range.clone()).ok_or(InvalidRangeError {
                    valid: 0..type_positions.len(),
                    invalid: self.group_range.clone(),
                })?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_prev_image.len(),
                        invalid: self.group_range.clone(),
                    })?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_next_image.len(),
                        invalid: self.group_range.clone(),
                    })?
            ) {
                *force = (position_prev_image.clone() + position_next_image.clone() - position.clone() * T::from(2.0))
                    * T::from(2.0)
                    * self.potential_prefix.clone();
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
                type_positions.get(self.group_range.clone()).ok_or(InvalidRangeError {
                    valid: 0..type_positions.len(),
                    invalid: self.group_range.clone(),
                })?,
                type_positions_prev_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_prev_image.len(),
                        invalid: self.group_range.clone(),
                    })?,
                type_positions_next_image
                    .get(self.group_range.clone())
                    .ok_or(InvalidRangeError {
                        valid: 0..type_positions_next_image.len(),
                        invalid: self.group_range.clone(),
                    })?
            ) {
                *force += (position_prev_image.clone() + position_next_image.clone() - position.clone() * T::from(2.0))
                    * T::from(2.0)
                    * self.potential_prefix.clone();
            }
            Ok(())
        }
    }

    #[derive(Clone, Debug)]
    pub struct InvalidRangeError {
        valid: Range<usize>,
        invalid: Range<usize>,
    }

    impl Display for InvalidRangeError {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            write!(
                f,
                "{}..{} is an invalid range in {}..{}",
                self.invalid.start, self.invalid.end, self.valid.start, self.valid.end
            )
        }
    }

    impl Error for InvalidRangeError {}

    #[derive(Clone, Copy, Debug)]
    pub struct EmptyIteratorError;

    impl Display for EmptyIteratorError {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            write!(f, "no items yielded")
        }
    }

    impl Error for EmptyIteratorError {}
}
