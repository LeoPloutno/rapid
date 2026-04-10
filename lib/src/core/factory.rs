//! Traits for producing different yet connected types of objects.

use crate::core::atoms::AtomType;

/// A trait for "factories" that produce iterators for leading, inner and trailing images.
pub trait Factory<'a, T> {
    /// The object used in a leading thread.
    type Leading: 'a;
    /// The object used in an inner thread.
    type Inner: 'a;
    /// The object used in a trailing thread.
    type Trailing: 'a;
    /// The iterator producing `Leading` objects.
    type LeadingIter: ExactSizeIterator<Item = Self::Leading>;
    /// The iterator producing iterators that produce `Inner` objects.
    type InnerIter: ExactSizeIterator<Item: ExactSizeIterator<Item = Self::Inner>>;
    /// The iterator producing `Trailing` objects.
    type TrailingIter: ExactSizeIterator<Item = Self::Trailing>;

    /// Produces the iterators.
    fn produce(
        &'a mut self,
        inner_images: usize,
        atom_types: &[AtomType<T>],
    ) -> (Self::LeadingIter, Self::InnerIter, Self::TrailingIter);
}

/// A trait for "factories" that produce iterators for the main thread and leading, inner and trailing images.
pub trait FullFactory<'a, T> {
    /// The object used in the main thread.
    type Main: 'a;
    /// The object used in a leading thread.
    type Leading: 'a;
    /// The object used in an inner thread.
    type Inner: 'a;
    /// The object used in a trailing thread.
    type Trailing: 'a;
    /// The iterator producing `Leading` objects.
    type LeadingIter: ExactSizeIterator<Item = Self::Leading>;
    /// The iterator producing iterators that produce `Inner` objects.
    type InnerIter: ExactSizeIterator<Item: ExactSizeIterator<Item = Self::Inner>>;
    /// The iterator producing `Trailing` objects.
    type TrailingIter: ExactSizeIterator<Item = Self::Trailing>;

    /// Produces the main object and the iterators.
    fn produce(
        &'a mut self,
        inner_images: usize,
        atom_types: &[AtomType<T>],
    ) -> (Self::Main, Self::LeadingIter, Self::InnerIter, Self::TrailingIter);
}
