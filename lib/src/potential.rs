//! Traits for updating the forces and calculating the different kinds of potential energies.

use crate::core::{AtomGroup, AtomTypeReaderLock, MapInWhole, MapOutsideWhole};

pub mod exchange;
pub mod physical;

pub type GroupInTypeInImage<'a, V> = MapOutsideWhole<
    &'a AtomGroup<V>,
    MapInWhole<&'a AtomTypeReaderLock<V>, &'a [AtomTypeReaderLock<V>]>,
>;
