//! Traits for exchange potentials expanded to the second order.

use crate::{core::GroupImageHandle, potential::exchange::ExchangePotential};
use arc_rw_lock::UniqueArcSliceRwLock;

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
        images_type_coordinates: &[GroupImageHandle<V>],
        group_modes: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Transforms all modes into the coordinates of this group.
    fn inverse_transform(
        &mut self,
        modes: &[UniqueArcSliceRwLock<V>],
        group_coordinates: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Sets `eigenvalues` to the eigenvalues of this transformation.
    ///
    /// Only the subset of all of the eigenvalues which corresponds to
    /// the group allocated to this transformation is set.
    fn eigenvalues(&self, eigenvalues: &mut [T]) -> Result<(), Self::Error>;
}
