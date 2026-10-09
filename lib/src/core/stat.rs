//! Types and traits meant to distinguish between different types
//! of ensemble statistics.

use std::ops::{Deref, DerefMut};

use crate::{
    core::{GroupInTypeInImage, marker::ValidOutput},
    potential::exchange::ExchangePotential,
};

/// An enum differentiating between distinguishable and bosonic statistics.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Stat<D, B> {
    /// Distinguishable statistics.
    Distinguishable(D),
    /// Bosonic statistics.
    Bosonic(B),
}

impl<D, B> Stat<D, B> {
    /// Converts from '&Stat<D, B>' to 'Stat<&D, &B>'.
    pub const fn as_ref(&self) -> Stat<&D, &B> {
        match self {
            Self::Distinguishable(dist) => Stat::Distinguishable(dist),
            Self::Bosonic(boson) => Stat::Bosonic(boson),
        }
    }

    /// Converts from '&mut Stat<D, B>' to 'Stat<&mut D, &mut B>'.
    pub const fn as_mut(&mut self) -> Stat<&mut D, &mut B> {
        match self {
            Self::Distinguishable(dist) => Stat::Distinguishable(dist),
            Self::Bosonic(boson) => Stat::Bosonic(boson),
        }
    }

    /// Converts from `Stat<D, B>` to
    /// `Stat<&D::Target, &B::Target>`.
    ///
    /// Leaves the original `Stat` in-place,
    /// creating a new one containing references to the inner types' `Deref::Target` types.
    pub fn as_deref(&self) -> Stat<&<D as Deref>::Target, &<B as Deref>::Target>
    where
        D: Deref,
        B: Deref,
    {
        match self {
            Self::Distinguishable(dist) => Stat::Distinguishable(dist),
            Self::Bosonic(boson) => Stat::Bosonic(boson),
        }
    }

    /// Converts from `Stat<D, B>` to
    /// `Stat<&mut D::Target, &mut B::Target>`.
    ///
    /// Leaves the original `Stat` in-place,
    /// creating a new one containing mutable references to the inner types' `Deref::Target` types.
    pub fn as_deref_mut(&mut self) -> Stat<&mut <D as Deref>::Target, &mut <B as Deref>::Target>
    where
        D: DerefMut,
        B: DerefMut,
    {
        match self {
            Self::Distinguishable(dist) => Stat::Distinguishable(dist),
            Self::Bosonic(boson) => Stat::Bosonic(boson),
        }
    }
}

impl<T, V, O, D, B> ExchangePotential<T, V, O> for Stat<D, B>
where
    O: ValidOutput<T>,
    D: ExchangePotential<T, V, O>,
    B: ExchangePotential<T, V, O>,
{
    type Error = Stat<D::Error, B::Error>;

    fn calculate_energy_set_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error> {
        Ok(match self {
            Self::Distinguishable(exchange_potential) => exchange_potential
                .calculate_energy_set_forces(prev_image_positions, next_image_positions, positions, forces)
                .map_err(Stat::Distinguishable)?,
            Self::Bosonic(exchange_potential) => exchange_potential
                .calculate_energy_set_forces(prev_image_positions, next_image_positions, positions, forces)
                .map_err(Stat::Bosonic)?,
        })
    }

    fn calculate_energy_add_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<O, Self::Error> {
        Ok(match self {
            Self::Distinguishable(exchange_potential) => exchange_potential
                .calculate_energy_add_forces(prev_image_positions, next_image_positions, positions, forces)
                .map_err(Stat::Distinguishable)?,
            Self::Bosonic(exchange_potential) => exchange_potential
                .calculate_energy_add_forces(prev_image_positions, next_image_positions, positions, forces)
                .map_err(Stat::Bosonic)?,
        })
    }

    fn calculate_energy(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &GroupInTypeInImage<V>,
        positions: &GroupInTypeInImage<V>,
    ) -> Result<O, Self::Error> {
        #[allow(deprecated)]
        Ok(match self {
            Self::Distinguishable(exchange_potential) => exchange_potential
                .calculate_energy(prev_image_positions, next_image_positions, positions)
                .map_err(Stat::Distinguishable)?,
            Self::Bosonic(exchange_potential) => exchange_potential
                .calculate_energy(prev_image_positions, next_image_positions, positions)
                .map_err(Stat::Bosonic)?,
        })
    }

    fn set_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &super::GroupInTypeInImage<V>,
        positions: &super::GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        Ok(match self {
            Self::Distinguishable(exchange_potential) => exchange_potential
                .set_forces(prev_image_positions, next_image_positions, positions, forces)
                .map_err(Stat::Distinguishable)?,
            Self::Bosonic(exchange_potential) => exchange_potential
                .set_forces(prev_image_positions, next_image_positions, positions, forces)
                .map_err(Stat::Bosonic)?,
        })
    }

    fn add_forces(
        &mut self,
        prev_image_positions: &GroupInTypeInImage<V>,
        next_image_positions: &super::GroupInTypeInImage<V>,
        positions: &super::GroupInTypeInImage<V>,
        forces: &mut [V],
    ) -> Result<(), Self::Error> {
        #[allow(deprecated)]
        Ok(match self {
            Self::Distinguishable(exchange_potential) => exchange_potential
                .add_forces(prev_image_positions, next_image_positions, positions, forces)
                .map_err(Stat::Distinguishable)?,
            Self::Bosonic(exchange_potential) => exchange_potential
                .add_forces(prev_image_positions, next_image_positions, positions, forces)
                .map_err(Stat::Bosonic)?,
        })
    }
}

/// A trait for marking exchange potentials of distinguishable particles.
pub trait Distinguishable {}

/// A trait for marking exchange potentials of bosons.
pub trait Bosonic {}
