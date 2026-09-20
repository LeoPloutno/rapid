mod distinguishable {
    use std::{
        ops::{Add, Mul},
        range::Range,
    };

    use lib::core::{
        Vector,
        error::{AccessError, EmptyError, InvalidRangeError},
        stat::Distinguishable,
        zip_items, zip_iterators,
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
        pub fn new(
            mass: T,
            temperature: T,
            inner_images: usize,
            group_range: Range<usize>,
        ) -> Self {
            assert!(mass.clone() > 0.0.into(), "the mass must be positive");
            assert!(
                temperature.clone() > 0.0.into(),
                "the temperature must be positive"
            );
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

    impl<const N: usize, T> Distinguishable for DistinguishableExchangePotential<N, T> {}
}

pub use distinguishable::DistinguishableExchangePotential;
