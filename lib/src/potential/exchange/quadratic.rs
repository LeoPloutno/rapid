//! Traits for exchange potentials expanded to the second order.

use super::ExchangePotential;
use crate::{core::AtomTypeReaderLock, stride::Stride};
use std::iter::FusedIterator;

/// A trait for exchange potential that may be expanded to second order.
pub trait QuadraticExpansionExchangePotential<'a, T, V> {
    /// The transformation that yields the modes such that
    /// the second order term is the sum over all modes squared times their
    /// respective eigenvalues.
    type QuadraticPotential: Transform<T, V>;
    /// The term left after the expansion to second order.
    /// Contains interactions of third order and higher.
    type ResidualPotential: ExchangePotential<T, V>;

    /// Treats `self` as a sum of harmonic oscillators - the modes -
    /// and a residual term of third order and beyond.
    fn as_quadratic_expansion(&'a mut self) -> (Self::QuadraticPotential, Self::ResidualPotential);
}

/// A trait for linear transformations between modes for coordinates.
pub trait Transform<T, V> {
    /// The type associated with an error returned by the implementor.
    type Error;

    /// Transforms all coordinates of a specific type into a subset of modes.
    ///
    /// The number of modes which are converted into by this transformation
    /// is the same as the number of atoms in the group allocated to it.
    fn transform(
        &mut self,
        images_type_coordinates: TypeAcrossImages<V>,
        group_modes: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Transforms all modes into the coordinates of this group.
    fn inverse_transform(
        &mut self,
        modes: TypeAcrossImages<V>,
        group_coordinates: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Sets `eigenvalues` to the eigenvalues of this transformation.
    ///
    /// Only the subset of all of the eigenvalues which corresponds to
    /// the group allocated to this transformation is set.
    fn eigenvalues(&self, eigenvalues: &mut [T]) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub struct TypeAcrossImages<'a, V>(Stride<'a, AtomTypeReaderLock<V>>);

impl<'a, V> Clone for TypeAcrossImages<'a, V> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl<'a, V> Copy for TypeAcrossImages<'a, V> {}

impl<'a, V> Iterator for TypeAcrossImages<'a, V> {
    type Item = &'a AtomTypeReaderLock<V>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, V> DoubleEndedIterator for TypeAcrossImages<'a, V> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<'a, V> ExactSizeIterator for TypeAcrossImages<'a, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, V> FusedIterator for TypeAcrossImages<'a, V> {}
