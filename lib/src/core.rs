//! Core functionalities used throughout the whole project.

use arc_rw_lock::{ArcSliceReaderLock, UniqueArcSliceRwLock};
use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

mod map_in_whole {
    use std::{ops::Deref, ptr, range::Range, slice};

    #[derive(Clone, Copy, Debug)]
    pub struct MapInWhole<T, U> {
        map: T,
        whole: U,
    }

    impl<T, U> MapInWhole<T, U> {
        pub const fn as_map(&self) -> &T::Target
        where
            T: Deref,
        {
            &*self.map
        }

        pub const fn as_whole(&self) -> &U::Target
        where
            U: Deref,
        {
            &*&self.whole
        }

        pub const fn as_ref(&self) -> MapInWhole<&T::Target, &U::Target>
        where
            T: Deref,
            U: Deref,
        {
            MapInWhole {
                map: &*self.map,
                whole: &*self.whole,
            }
        }
    }

    impl<T: Deref, U> Deref for MapInWhole<T, U> {
        type Target = T::Target;

        /// Equivalent to [`MapInWhole::as_map`].
        fn deref(&self) -> &T::Target {
            &*self.map
        }
    }

    impl<T: Deref, U> AsRef<T::Target> for MapInWhole<T, U> {
        /// Equivalent to [`MapInWhole::as_map`].
        fn as_ref(&self) -> &T::Target {
            &*self.map
        }
    }

    impl<'a, T> MapInWhole<&'a T, &'a [T]> {
        pub const fn before(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole;
            }
            let slice_ptr = self.whole.as_ptr();
            let element_ptr = ptr::from_ref(self.map);
            unsafe {
                // SAFETY: - `slice_ptr` is derived from a reference.
                //         - The offset of an element - `self.subfield` - from the
                //           origin - `self.whole` if always less than the length of the slice.
                slice::from_raw_parts(
                    slice_ptr,
                    // SAFETY: By construction, `self.subfield` points to an element of `self.whole`,
                    //         so it always exceeds or is the start of the slice.
                    element_ptr.offset_from_unsigned(slice_ptr),
                )
            }
        }

        pub const fn after(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole;
            }
            let slice_end_ptr = self.whole.as_ptr_range().end;
            let element_ptr = ptr::from_ref(self.map);
            unsafe {
                // SAFETY: - By construction, `self.subfield` points to an element of `self.whole`.
                //         - `element_ptr + (slice_end_ptr - element_ptr) = slice_end_ptr`.
                slice::from_raw_parts(
                    element_ptr,
                    // SAFETY: By construction, `self.subfield` points to an element of `self.whole`,
                    //         so the it does not exceed the end of the slice.
                    slice_end_ptr.offset_from_unsigned(element_ptr),
                )
            }
        }

        pub const fn element_offset(&self) -> usize {
            if const { size_of::<T>() == 0 } {
                panic!("elements are zero-sized");
            }
            // SAFETY: By construction, `self.subfield` points to an element of `self.whole`,
            //         so the it does not exceed the end of the slice.
            unsafe { ptr::from_ref(self.map).offset_from_unsigned(self.whole.as_ptr()) }
        }
    }

    impl<'a, T, U> MapInWhole<&'a T, MapInWhole<&'a [T], U>> {
        pub const fn before(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole.map;
            }
            let slice_ptr = self.whole.as_ptr();
            let element_ptr = ptr::from_ref(self.map);
            unsafe {
                // SAFETY: - `slice_ptr` is derived from a reference.
                //         - The offset of an element - `self.subfield` - from the
                //           origin - `self.whole.subfield` if always less than the length of the slice.
                slice::from_raw_parts(
                    slice_ptr,
                    // SAFETY: By construction, `self.subfield` points to an element of `self.whole.subfield`,
                    //         so it always exceeds or is the start of the slice.
                    element_ptr.offset_from_unsigned(slice_ptr),
                )
            }
        }

        pub const fn after(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole.map;
            }
            let slice_end_ptr = self.whole.as_ptr_range().end;
            let element_ptr = ptr::from_ref(self.map);
            unsafe {
                // SAFETY: - By construction, `self.subfield` points to an element of `self.whole.subfield`.
                //         - `element_ptr + (slice_end_ptr - element_ptr) = slice_end_ptr`.
                slice::from_raw_parts(
                    element_ptr,
                    // SAFETY: By construction, `self.subfield` points to an element of `self.whole.subfield`,
                    //         so the it does not exceed the end of the slice.
                    slice_end_ptr.offset_from_unsigned(element_ptr),
                )
            }
        }

        pub const fn element_offset(&self) -> usize {
            if const { size_of::<T>() == 0 } {
                panic!("elements are zero-sized");
            }
            // SAFETY: By construction, `self.subfield` points to an element of `self.whole.subfield`,
            //         so the it does not exceed the end of the slice.
            unsafe { ptr::from_ref(self.map).offset_from_unsigned(self.whole.map.as_ptr()) }
        }
    }

    impl<'a, T, U> MapInWhole<MapInWhole<U, &'a T>, &'a [T]> {
        pub const fn before(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole;
            }
            let slice_ptr = self.whole.as_ptr();
            let element_ptr = ptr::from_ref(self.map.whole);
            unsafe {
                // SAFETY: - `slice_ptr` is derived from a reference.
                //         - The offset of an element - `self.subfield.whole` - from the
                //           origin - `self.whole` if always less than the length of the slice.
                slice::from_raw_parts(
                    slice_ptr,
                    // SAFETY: By construction, `self.subfield.whole` points to an element of `self.whole`,
                    //         so it always exceeds or is the start of the slice.
                    element_ptr.offset_from_unsigned(slice_ptr),
                )
            }
        }

        pub const fn after(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole;
            }
            let slice_end_ptr = self.whole.as_ptr_range().end;
            let element_ptr = ptr::from_ref(self.map.whole);
            unsafe {
                // SAFETY: - By construction, `self.subfield.whole` points to an element of `self.whole`.
                //         - `element_ptr + (slice_end_ptr - element_ptr) = slice_end_ptr`.
                slice::from_raw_parts(
                    element_ptr,
                    // SAFETY: By construction, `self.subfield.whole` points to an element of `self.whole`,
                    //         so the it does not exceed the end of the slice.
                    slice_end_ptr.offset_from_unsigned(element_ptr),
                )
            }
        }

        pub const fn element_offset(&self) -> usize {
            if const { size_of::<T>() == 0 } {
                panic!("elements are zero-sized");
            }
            // SAFETY: By construction, `self.subfield.whole` points to an element of `self.whole`,
            //         so the it does not exceed the end of the slice.
            unsafe { ptr::from_ref(self.map.whole).offset_from_unsigned(self.whole.as_ptr()) }
        }
    }

    impl<'a, T> MapInWhole<&'a [T], &'a [T]> {
        pub const fn before(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole;
            }
            let slice_ptr = self.whole.as_ptr();
            let subslice_ptr = self.map.as_ptr();
            unsafe {
                // SAFETY: - `slice_ptr` is derived from a reference.
                //         - The offset of a subslice - `self.subfield` from the
                //           origin - `self.whole` - is always less than or equal to the
                //           length of the slice.
                slice::from_raw_parts(
                    slice_ptr,
                    // SAFETY: By construction, `self.subfield` points to a subslice entirely within
                    //         `self.whole`, so its start always exceeds or is the slice's.
                    subslice_ptr.offset_from_unsigned(slice_ptr),
                )
            }
        }

        pub const fn after(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole;
            }
            let slice_end_ptr = self.whole.as_ptr_range().end;
            let subslice_end_ptr = self.map.as_ptr_range().end;
            unsafe {
                // SAFETY: - By construction, `self.subfield` points to a subslice entirely within
                //           `self.whole`. Thus, the end of said subslice also points within it.
                //         - `subslice_end_ptr + (slice_end_ptr - subslice_end_ptr) = slice_end_ptr`.
                slice::from_raw_parts(
                    subslice_end_ptr,
                    // SAFETY: By construction, `self.subfield` points to a subslice entirely within
                    //         `self.whole`, so the its end does not exceed the slice's.
                    slice_end_ptr.offset_from_unsigned(subslice_end_ptr),
                )
            }
        }

        pub const fn subslice_range(&self) -> Range<usize> {
            if const { size_of::<T>() == 0 } {
                panic!("elements are zero-sized");
            }
            let subslice_len = self.map.len();
            unsafe {
                // SAFETY: By construction, `self.subfield` points to a subslice entirely within `self.whole`,
                //         so its start does not preceed the slice's.
                let start = self.map.as_ptr().offset_from_unsigned(self.whole.as_ptr());
                Range {
                    start,
                    // SAFETY: Adding the length of a subslice to its start cannot overflow.
                    end: start.unchecked_add(subslice_len),
                }
            }
        }
    }

    impl<'a, T, U> MapInWhole<&'a [T], MapInWhole<&'a [T], U>> {
        pub const fn before(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole.map;
            }
            let slice_ptr = self.whole.map.as_ptr();
            let subslice_ptr = self.map.as_ptr();
            unsafe {
                // SAFETY: - `slice_ptr` is derived from a reference.
                //         - The offset of a subslice - `self.subfield` from the
                //           origin - `self.whole.subfield` - is always less than or equal to the
                //           length of the slice.
                slice::from_raw_parts(
                    slice_ptr,
                    // SAFETY: By construction, `self.subfield` points to a subslice entirely within
                    //         `self.whole.subfield`, so its start always exceeds or is the slice's.
                    subslice_ptr.offset_from_unsigned(slice_ptr),
                )
            }
        }

        pub const fn after(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole.map;
            }
            let slice_end_ptr = self.whole.as_ptr_range().end;
            let subslice_end_ptr = self.map.as_ptr_range().end;
            unsafe {
                // SAFETY: - By construction, `self.subfield` points to a subslice entirely within
                //           `self.whole.subfield`. Thus, the end of said subslice also points within it.
                //         - `subslice_end_ptr + (slice_end_ptr - subslice_end_ptr) = slice_end_ptr`.
                slice::from_raw_parts(
                    subslice_end_ptr,
                    // SAFETY: By construction, `self.subfield` points to a subslice entirely within
                    //         `self.whole.subfield`, so the its end does not exceed the slice's.
                    slice_end_ptr.offset_from_unsigned(subslice_end_ptr),
                )
            }
        }

        pub const fn subslice_range(&self) -> Range<usize> {
            if const { size_of::<T>() == 0 } {
                panic!("elements are zero-sized");
            }
            let subslice_len = self.map.len();
            unsafe {
                // SAFETY: By construction, `self.subfield` points to a subslice entirely within `self.whole.subfield`,
                //         so its start does not preceed the slice's.
                let start = self
                    .map
                    .as_ptr()
                    .offset_from_unsigned(self.whole.map.as_ptr());
                Range {
                    start,
                    // SAFETY: Adding the length of a subslice to its start cannot overflow.
                    end: start.unchecked_add(subslice_len),
                }
            }
        }
    }

    impl<'a, T, U> MapInWhole<MapInWhole<U, &'a [T]>, &'a [T]> {
        pub const fn before(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole;
            }
            let slice_ptr = self.whole.as_ptr();
            let subslice_ptr = self.map.whole.whole.as_ptr();
            unsafe {
                // SAFETY: - `slice_ptr` is derived from a reference.
                //         - The offset of a subslice - `self.subfield.whole` from the
                //           origin - `self.whole` - is always less than or equal to the
                //           length of the slice.
                slice::from_raw_parts(
                    slice_ptr,
                    // SAFETY: By construction, `self.subfield.whole` points to a subslice entirely within
                    //         `self.whole`, so its start always exceeds or is the slice's.
                    subslice_ptr.offset_from_unsigned(slice_ptr),
                )
            }
        }

        pub const fn after(&self) -> &[T] {
            if const { size_of::<T>() == 0 } {
                return self.whole;
            }
            let slice_end_ptr = self.whole.as_ptr_range().end;
            let subslice_end_ptr = self.map.whole.as_ptr_range().end;
            unsafe {
                // SAFETY: - By construction, `self.subfield.whole` points to a subslice entirely within
                //           `self.whole`. Thus, the end of said subslice also points within it.
                //         - `subslice_end_ptr + (slice_end_ptr - subslice_end_ptr) = slice_end_ptr`.
                slice::from_raw_parts(
                    subslice_end_ptr,
                    // SAFETY: By construction, `self.subfield.whole` points to a subslice entirely within
                    //         `self.whole`, so the its end does not exceed the slice's.
                    slice_end_ptr.offset_from_unsigned(subslice_end_ptr),
                )
            }
        }

        pub const fn subslice_range(&self) -> Range<usize> {
            if const { size_of::<T>() == 0 } {
                panic!("elements are zero-sized");
            }
            let subslice_len = self.map.whole.len();
            unsafe {
                // SAFETY: By construction, `self.subfield.whole` points to a subslice entirely within `self.whole`,
                //         so its start does not preceed the slice's.
                let start = self
                    .map
                    .whole
                    .as_ptr()
                    .offset_from_unsigned(self.whole.as_ptr());
                Range {
                    start,
                    // SAFETY: Adding the length of a subslice to its start cannot overflow.
                    end: start.unchecked_add(subslice_len),
                }
            }
        }
    }
}
pub use map_in_whole::MapInWhole;

mod map_outside_whole {
    use std::ops::{Deref, DerefMut};

    #[derive(Clone, Copy, Debug)]
    pub struct MapOutsideWhole<T, U> {
        map: T,
        whole: U,
    }

    impl<T, U> MapOutsideWhole<T, U> {
        pub const fn as_map(&self) -> &T::Target
        where
            T: Deref,
        {
            &*self.map
        }

        pub const fn as_map_mut(&mut self) -> &mut T::Target
        where
            T: DerefMut,
        {
            &mut *self.map
        }

        pub const fn as_whole(&self) -> &U::Target
        where
            U: Deref,
        {
            &*self.whole
        }

        pub const fn as_whole_mut(&mut self) -> &mut U::Target
        where
            U: DerefMut,
        {
            &mut *self.whole
        }

        pub const fn as_ref(&self) -> MapOutsideWhole<&T::Target, &U::Target>
        where
            T: Deref,
            U: Deref,
        {
            MapOutsideWhole {
                map: &*self.map,
                whole: &*self.whole,
            }
        }

        pub const fn as_mut(&mut self) -> MapOutsideWhole<&mut T::Target, &mut U::Target>
        where
            T: DerefMut,
            U: DerefMut,
        {
            MapOutsideWhole {
                map: &mut *self.map,
                whole: &mut *self.whole,
            }
        }
    }

    impl<T: Deref, U> Deref for MapOutsideWhole<T, U> {
        type Target = T::Target;

        /// Equivalent to [`MapOutsideWhole::as_map`].
        fn deref(&self) -> &Self::Target {
            &*self.map
        }
    }

    impl<T: DerefMut, U> DerefMut for MapOutsideWhole<T, U> {
        /// Equivalent to [`MapOutsideWhole::as_map_mut`].
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut *self.map
        }
    }

    impl<T: Deref, U> AsRef<T::Target> for MapOutsideWhole<T, U> {
        /// Equivalent to [`MapOutsideWhole::as_map`].
        fn as_ref(&self) -> &T::Target {
            &*self.map
        }
    }

    impl<T: DerefMut, U> AsMut<T::Target> for MapOutsideWhole<T, U> {
        /// Equivalent to [`MapOutsideWhole::as_map_mut`].
        fn as_mut(&mut self) -> &mut T::Target {
            &mut *self.map
        }
    }
}
pub use map_outside_whole::MapOutsideWhole;

pub type AtomGroup<V> = UniqueArcSliceRwLock<V>;

pub type AtomGroupRwLock<V> = UniqueArcSliceRwLock<AtomGroup<V>>;

pub type AtomTypeReaderLock<V> = ArcSliceReaderLock<AtomGroup<V>>;

pub type AtomType<V> = V;

pub type Image<V> = ArcSliceReaderLock<V>;

pub type GroupInTypeInImageInSystem<'a, V> = MapOutsideWhole<
    &'a AtomGroup<V>,
    MapInWhole<
        &'a AtomTypeReaderLock<V>,
        MapInWhole<&'a [AtomTypeReaderLock<V>], &'a [AtomTypeReaderLock<V>]>,
    >,
>;

// pub struct ImageInSystem<'a, V>(SubsliceInSlice<'a, AtomType<V>>);

// pub struct ImagesIter<'a, V> {
//     start: *const AtomType<V>,
//     end: *const AtomType<V>,
//     image_size: NonZeroUsize,
//     phantom: PhantomData<&'a [AtomType<V>]>,
// }

// impl<'a, V> ImageInSystem<'a, V> {
//     pub const fn this(&self) -> &[AtomType<V>] {
//         self.0.this()
//     }

//     pub const fn before(&self) -> ImagesIter<'a, V> {
//         let before = self.0.before();
//         let image_size = self.this().len();
//         assert!(image_size > 0);
//         // SAFETY: Checked above that `image_size > 0`.
//         let image_size = unsafe { NonZeroUsize::new_unchecked(image_size) };
//         assert!(matches!(before.len().checked_rem(image_size.get()), Some(0)));
//         let before_range = before.as_ptr_range();
//         ImagesIter {
//             start: before_range.start,
//             end: before_range.end,
//             image_size,
//             phantom: PhantomData,
//         }
//     }

//     pub const fn after(&self) -> ImagesIter<'a, V> {
//         let after = self.0.after();
//         let image_size = self.this().len();
//         assert!(image_size > 0);
//         // SAFETY: Checked above that `image_size > 0`.
//         let image_size = unsafe { NonZeroUsize::new_unchecked(image_size) };
//         assert!(matches!(after.len().checked_rem(image_size.get()), Some(0)));
//         let after_range = after.as_ptr_range();
//         ImagesIter {
//             start: after_range.start,
//             end: after_range.end,
//             image_size,
//             phantom: PhantomData,
//         }
//     }
// }

mod atoms;

pub use atoms::{AtomTypeInfo, GroupSizes, GroupSizesIter, GroupsIter};

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

#[cfg(feature = "monte_carlo")]
pub mod monte_carlo {
    pub enum ChangedGroup {
        This,
        Other(usize),
    }
}

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
    pub fn as_deref(&mut self) -> Scheme<&mut T::Target, &mut U::Target> {
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

/// A wrapper for implementors of `Decoupled` traits.
pub struct Decoupled<T: ?Sized>(pub(crate) T);

impl<T> Decoupled<T> {
    /// Wraps the provided value with `Decoupled`.
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}

/// A wrapper for implementors of `Additive` traits.
pub struct Additive<T: ?Sized>(pub(crate) T);

impl<T> Additive<T> {
    /// Wraps the provided value with `Additive`.
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}

/// A wrapper for implementors of `Multiplicative` traits.
pub struct Multiplicative<T: ?Sized>(pub(crate) T);

impl<T> Multiplicative<T> {
    /// Wraps the provided value with `Multiplicative`.
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}
