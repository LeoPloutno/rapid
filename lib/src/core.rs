//! Core functionalities used throughout the whole project.

use std::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Range, Sub, SubAssign},
    sync::LockResult,
};

use arc_rw_lock::{MappedRwLockReadWholeGuard, MappedRwLockWriteGuard, UniqueArcElementRwLock, UniqueArcSliceRwLock};

mod atoms;

pub use atoms::{AtomType, GroupSizes, GroupSizesIter, GroupsIter};

pub mod error;

pub mod marker {
    //! Marker traits for allowing default implementations.

    /// A marker trait for types that can implement `Leading[...]`
    /// traits by reusing their `Inner[...]` implementation.
    pub trait InnerIsLeading {}

    /// A marker trait for types that can implement `Trailing[...]`
    /// traits by reusing their `Inner[...]` implementation.
    pub trait InnerIsTrailing {}
}

pub mod stat;

pub mod sync_ops;

pub mod factory;

/// A macro that allows pattern-matching items of [zipped iterators](zip_iterators).
#[macro_export]
macro_rules! zip_items {
    ($item1:pat, $item2:pat $(,)?) => {
        ($item1, $item2)
    };
    ($item:pat, $($items:pat),+ $(,)?) => {
        ($item, zip_items!($($items),+))
    };
}
pub use zip_items;

/// A macro that automatically zips all provided iterators in a consistent manner.
#[macro_export]
macro_rules! zip_iterators {
    ($iter:expr) => {
        $iter
    };
    ($iter1:expr, $iter2:expr $(,)?) => {
        $iter1.into_iter().zip($iter2)
    };
    ($iter:expr, $($iters:expr),+ $(,)?) => {
        $iter.into_iter().zip(zip_iterators!($($iters),+))
    };
}
pub use zip_iterators;

/// A trait for objects that can be used as vectors.
pub trait Vector<const N: usize>:
    Sized
    + From<[Self::Element; N]>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Self::Element, Output = Self>
    + MulAssign<Self::Element>
    + Div<Self::Element, Output = Self>
    + DivAssign<Self::Element>
    + Neg<Output = Self>
{
    /// The type of the element of the vector.
    type Element;

    /// Converts to a reference to an array.
    fn as_array(&self) -> &[Self::Element; N];

    /// Converts to a mutable reference to an array.
    fn as_mut_array(&mut self) -> &mut [Self::Element; N];

    /// Calculates the square of the magnitude (length) of the vector.
    fn magnitude_squared(self) -> Self::Element;

    /// Calculates the dot product of `self` with `rhs`.
    fn dot(self, rhs: Self) -> Self::Element;
}

/// A lock which grants mutable access to a single group
/// or immutable access to all groups of the same type.
pub struct GroupTypeHandle<T>(UniqueArcSliceRwLock<T>);

impl<T> GroupTypeHandle<T> {
    /// Grants immutable access to the underlying group.
    pub fn read(&self) -> &[T] {
        self.0.as_ref().read()
    }

    /// Grants mutable access to the underlying group.
    pub fn write(&mut self) -> MappedRwLockWriteGuard<'_, [T]> {
        self.0.as_mut().write()
    }

    /// Grants immutable access to all groups allocated to a type of atoms,
    /// returning an error if a thread holding a mutable guard panicked.
    pub fn read_type(&self) -> LockResult<MappedRwLockReadWholeGuard<'_, [T]>> {
        self.0.as_ref().read_whole()
    }

    /// Returns the range of the subslice the underlying group
    /// represents amongst all atoms of the same type.
    pub fn range(&self) -> Range<usize> {
        self.0.as_ref().subslice_range()
    }
}

/// A lock which grants mutable access to a single image
/// or immutable access to all images.
pub struct GroupImageHandle<G>(UniqueArcElementRwLock<G>);

impl<G> GroupImageHandle<G> {
    /// Grants immutable access to the underlying image.
    pub fn read(&self) -> &G {
        self.0.as_ref().read()
    }

    /// Grants immutable access to the underlying image.
    pub fn write(&mut self) -> MappedRwLockWriteGuard<'_, G> {
        self.0.as_mut().write()
    }

    /// Grants immutable access to images,
    /// returning an error if a thread holding a mutable guard panicked.
    pub fn read_image(&self) -> LockResult<MappedRwLockReadWholeGuard<'_, [G]>> {
        self.0.as_ref().read_whole()
    }

    /// Returns the index of the group.
    pub fn index(&self) -> usize {
        self.0.as_ref().element_offset()
    }
}

/// Exchange potential expansion scheme.
#[derive(Clone, Copy, Debug)]
pub enum Scheme<T, U> {
    /// Regular, unexpanded.
    Regular(T),
    /// Expanded to the second order.
    QuadraticExpansion(U),
}

/// A struct which contains the two objects dependent on the expansion
/// scheme - the propagator and the exchange potential.
pub struct SchemeDependent<Prop, ExchPot> {
    /// The propagator.
    pub propagator: Prop,
    /// The exchange potential.
    pub exchange_potential: ExchPot,
}

/// A wrapper for implementors of `Decoupled` traits.
pub struct Decoupled<T: ?Sized>(pub T);
/// A wrapper for implementors of `Additive` traits.
pub struct Additive<T: ?Sized>(pub T);
/// A wrapper for implementors of `Multiplicative` traits.
pub struct Multiplicative<T: ?Sized>(pub T);
