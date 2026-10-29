//! Core functionalities used throughout the whole project.

use arc_rw_lock::{ArcSliceReaderLock, UniqueArcElementRwLock, UniqueArcSliceRwLock};
use std::{
    num::NonZeroUsize,
    ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub},
    sync::{Barrier, RwLock},
};

mod map_in_whole;
pub use map_in_whole::MapInWhole;

mod map_outside_whole;
pub use map_outside_whole::MapOutsideWhole;

pub mod stat;

pub mod sync_ops;

pub mod error;

pub mod marker;

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
pub trait Vector<const N: usize>
where
    Self: Sized
        + Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Self::Element, Output = Self>
        + Div<Self::Element, Output = Self>
        + Neg<Output = Self>,
{
    /// The type of the element of the vector.
    type Element;

    /// Calculates the square of the magnitude (length) of the vector.
    fn magnitude_squared(self) -> Self::Element {
        self.dot(self)
    }

    /// Calculates the dot product of `self` with `rhs`.
    fn dot(self, rhs: Self) -> Self::Element;

    /// Converts a reference to `Self` into a reference to an array.
    fn as_array(&self) -> &[Self::Element; N];

    /// Converts a mutable reference to `Self` into a mutable reference to an array.
    fn as_mut_array(&mut self) -> &mut [Self::Element; N];
}

pub type AtomGroup<V> = UniqueArcSliceRwLock<V>;

pub type AtomGroupRwLock<V> = UniqueArcElementRwLock<AtomGroup<V>>;

#[derive(Debug)]
pub struct AtomTypeCharacteristics<T> {
    pub mass: T,
    pub charge: T,
    pub name: String,
}

#[derive(Debug)]
pub struct AtomType<'a, T, V> {
    pub characteristics: &'a AtomTypeCharacteristics<T>,
    pub lock: ArcSliceReaderLock<AtomGroup<V>>,
}

pub type GroupInTypeInImageInSystem<'a, T, V> = MapOutsideWhole<
    &'a AtomGroup<V>,
    MapInWhole<&'a AtomType<'a, T, V>, MapInWhole<&'a [AtomType<'a, T, V>], &'a [AtomType<'a, T, V>]>>,
>;

pub type GroupInTypeInImage<'a, T, V> =
    MapOutsideWhole<&'a AtomGroup<V>, MapInWhole<&'a AtomType<'a, T, V>, &'a [AtomType<'a, T, V>]>>;

pub type GroupRwLockInTypeInImageInSystem<'a, T, V> = MapOutsideWhole<
    &'a mut AtomGroupRwLock<V>,
    MapInWhole<&'a AtomType<'a, T, V>, MapInWhole<&'a [AtomType<'a, T, V>], &'a [AtomType<'a, T, V>]>>,
>;

/// Exchange potential expansion scheme.
#[derive(Clone, Copy, Debug)]
pub enum Scheme<T, U> {
    /// Regular, unexpanded.
    Regular(T),
    /// Expanded to the second order.
    QuadraticExpansion(U),
}

impl<T: Deref, U: Deref> Scheme<T, U> {
    /// Converts from `Scheme<T, U>` to
    /// `Scheme<&T::Target, &U::Target>`.
    ///
    /// Leaves the original `Scheme` in-place,
    /// creating a new one containing references to the inner types' `Deref::Target` types.
    pub fn as_deref(&self) -> Scheme<&T::Target, &U::Target> {
        match self {
            Self::Regular(r) => Scheme::Regular(r),
            Self::QuadraticExpansion(r) => Scheme::QuadraticExpansion(r),
        }
    }
}

impl<T: DerefMut, U: DerefMut> Scheme<T, U> {
    /// Converts from `Scheme<T, U>` to
    /// `Scheme<&mut T::Target, &mut U::Target>`.
    ///
    /// Leaves the original `Scheme` in-place,
    /// creating a new one containing mutable references to the inner types' `Deref::Target` types.
    pub fn as_dere_mutf(&mut self) -> Scheme<&mut T::Target, &mut U::Target> {
        match self {
            Self::Regular(r) => Scheme::Regular(r),
            Self::QuadraticExpansion(r) => Scheme::QuadraticExpansion(r),
        }
    }
}

/// A struct which contains the two objects dependent on the expansion
/// scheme - the propagator and the exchange potential.
pub struct SchemeDependent<Prop, ExchPot> {
    /// The propagator.
    pub propagator: Prop,
    /// The exchange potential.
    pub exchange_potential: ExchPot,
}

/// An entity that allows synchronized sharing of data between threads.
pub struct Synchronizer<T> {
    pub(crate) lock: RwLock<T>,
    pub(crate) barrier: Barrier,
}

impl<T> Synchronizer<T> {
    /// Constructs a new synchronizer
    pub const fn new(lock: RwLock<T>, n: usize) -> Self {
        Self {
            lock,
            barrier: Barrier::new(n),
        }
    }
}

/// The type of and image.
#[derive(Clone, Copy, Debug)]
pub enum ImageType {
    /// The first image.
    Leading,
    /// Some image that is neither the first nor last.
    Inner(NonZeroUsize),
    /// The last image.
    Trailing(NonZeroUsize),
}
