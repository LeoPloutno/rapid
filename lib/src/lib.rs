#![feature(ptr_metadata, negative_impls)]
#![allow(clippy::too_many_arguments)]
#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]
#![warn(missing_docs)]

//! This library defines the core simulation entities, such as propagators,
//! potentials, thermostats, etc.
//! To run a simulation, simply call `[run]` with the right arguments.

pub mod core;

pub mod estimator;

pub mod output;

pub mod potential;

pub mod thermostat;

pub mod propagator;

mod simulation;
pub use simulation::{PropagationOutput, Simulation};

mod stride;
mod stride_mut;
