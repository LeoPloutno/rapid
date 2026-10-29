//! Traits for exchange potentials expanded to the second order.

use super::ExchangePotential;
use crate::{
    core::{AtomType, marker::ValidOutput},
    stride::Stride,
};
use std::{
    error::Error,
    iter::FusedIterator,
    ops::{Deref, DerefMut},
};

/// A trait for exchange potential that may be expanded to second order.
pub trait QuadraticExpansionExchangePotential<'a, T, V, O: ValidOutput<T>> {
    /// The transformation that yields the modes such that
    /// the second order term is the sum over all modes squared times their
    /// respective eigenvalues.
    type QuadraticPotential: Transform<T, V>;
    /// The term left after the expansion to second order.
    /// Contains interactions of third order and higher.
    type ResidualPotential: ExchangePotential<T, V, O>;

    /// Treats `self` as a sum of harmonic oscillators - the modes -
    /// and a residual term of third order and beyond.
    fn as_quadratic_expansion(&'a mut self) -> (Self::QuadraticPotential, Self::ResidualPotential);
}

impl<'a, T, V, O, P> QuadraticExpansionExchangePotential<'a, T, V, O> for P
where
    O: ValidOutput<T>,
    P: DerefMut<Target: QuadraticExpansionExchangePotential<'a, T, V, O>> + ?Sized,
{
    type QuadraticPotential =
        <<Self as Deref>::Target as QuadraticExpansionExchangePotential<'a, T, V, O>>::QuadraticPotential;
    type ResidualPotential =
        <<Self as Deref>::Target as QuadraticExpansionExchangePotential<'a, T, V, O>>::ResidualPotential;

    #[inline]
    fn as_quadratic_expansion(&'a mut self) -> (Self::QuadraticPotential, Self::ResidualPotential) {
        (*self).as_quadratic_expansion()
    }
}

/// A trait for linear transformations between modes for coordinates.
pub trait Transform<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error: Error;

    /// Transforms all coordinates of a specific type into a subset of modes.
    ///
    /// The number of modes which are converted into by this transformation
    /// is the same as the number of atoms in the group allocated to it.
    fn transform(
        &mut self,
        images_type_coordinates: TypeAcrossImages<T, V>,
        group_modes: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Transforms all modes into the coordinates of this group.
    fn inverse_transform(
        &mut self,
        modes: TypeAcrossImages<T, V>,
        group_coordinates: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Sets `eigenvalues` to the eigenvalues of this transformation.
    ///
    /// Only the subset of all of the eigenvalues which corresponds to
    /// the group allocated to this transformation is set.
    fn eigenvalues(&self, eigenvalues: &mut [T]) -> Result<(), Self::Error>;
}

impl<T, V, U> Transform<T, V> for U
where
    U: DerefMut<Target: Transform<T, V>> + ?Sized,
{
    type Error = <<Self as Deref>::Target as Transform<T, V>>::Error;

    #[inline]
    fn transform(
        &mut self,
        images_type_coordinates: TypeAcrossImages<T, V>,
        group_modes: &mut [V],
    ) -> Result<(), Self::Error> {
        (*self).transform(images_type_coordinates, group_modes)
    }

    #[inline]
    fn inverse_transform(
        &mut self,
        modes: TypeAcrossImages<T, V>,
        group_coordinates: &mut [V],
    ) -> Result<(), Self::Error> {
        (*self).inverse_transform(modes, group_coordinates)
    }

    #[inline]
    fn eigenvalues(&self, eigenvalues: &mut [T]) -> Result<(), Self::Error> {
        (*self).eigenvalues(eigenvalues)
    }
}

/// An iterator over different instantiations of an atom type in all images.
#[derive(Debug)]
pub struct TypeAcrossImages<'a, T, V>(Stride<'a, AtomType<'a, T, V>>);

impl<'a, T, V> Clone for TypeAcrossImages<'a, T, V> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl<'a, T, V> Copy for TypeAcrossImages<'a, T, V> {}

impl<'a, T, V> Iterator for TypeAcrossImages<'a, T, V> {
    type Item = &'a AtomType<'a, T, V>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, T, V> DoubleEndedIterator for TypeAcrossImages<'a, T, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<'a, T, V> ExactSizeIterator for TypeAcrossImages<'a, T, V> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, T, V> FusedIterator for TypeAcrossImages<'a, T, V> {}
