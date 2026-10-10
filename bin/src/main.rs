#![allow(incomplete_features)]
#![feature(portable_simd, with_negative_coherence, generic_const_exprs)]

pub mod core;
pub mod estimator;
pub mod potential;
pub mod propagator;
pub mod thermostat;
pub mod vector;

fn main() {
    println!("Hello, world!");
}
