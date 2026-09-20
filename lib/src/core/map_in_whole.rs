use std::{ops::Deref, ptr, range::Range, slice};

#[derive(Clone, Copy, Debug)]
pub struct MapInWhole<T, U> {
    map: T,
    whole: U,
}

impl<T, U> MapInWhole<T, U> {
    pub fn as_map(&self) -> &T::Target
    where
        T: Deref,
    {
        &*self.map
    }

    pub fn as_whole(&self) -> &U::Target
    where
        U: Deref,
    {
        &*&self.whole
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

impl<T, U, V> From<MapInWhole<T, MapInWhole<U, V>>> for MapInWhole<T, U> {
    fn from(value: MapInWhole<T, MapInWhole<U, V>>) -> Self {
        Self {
            map: value.map,
            whole: value.whole.map,
        }
    }
}

impl<T, U, V> From<MapInWhole<MapInWhole<T, U>, V>> for MapInWhole<U, V> {
    fn from(value: MapInWhole<MapInWhole<T, U>, V>) -> Self {
        Self {
            map: value.map.whole,
            whole: value.whole,
        }
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
            //         - The offset of an element - `self.map` - from the
            //           origin - `self.whole` if always less than the length of the slice.
            slice::from_raw_parts(
                slice_ptr,
                // SAFETY: By construction, `self.map` points to an element of `self.whole`,
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
            // SAFETY: - By construction, `self.map` points to an element of `self.whole`.
            //         - `element_ptr + (slice_end_ptr - element_ptr) = slice_end_ptr`.
            slice::from_raw_parts(
                element_ptr,
                // SAFETY: By construction, `self.map` points to an element of `self.whole`,
                //         so the it does not exceed the end of the slice.
                slice_end_ptr.offset_from_unsigned(element_ptr),
            )
        }
    }

    pub const fn element_offset(&self) -> usize {
        if const { size_of::<T>() == 0 } {
            panic!("elements are zero-sized");
        }
        // SAFETY: By construction, `self.map` points to an element of `self.whole`,
        //         so the it does not exceed the end of the slice.
        unsafe { ptr::from_ref(self.map).offset_from_unsigned(self.whole.as_ptr()) }
    }
}

impl<'a, T, U> MapInWhole<&'a T, MapInWhole<&'a [T], U>> {
    pub const fn before(&self) -> &[T] {
        if const { size_of::<T>() == 0 } {
            return self.whole.map;
        }
        let slice_ptr = self.whole.map.as_ptr();
        let element_ptr = ptr::from_ref(self.map);
        unsafe {
            // SAFETY: - `slice_ptr` is derived from a reference.
            //         - The offset of an element - `self.map` - from the
            //           origin - `self.whole.map` if always less than the length of the slice.
            slice::from_raw_parts(
                slice_ptr,
                // SAFETY: By construction, `self.map` points to an element of `self.whole.map`,
                //         so it always exceeds or is the start of the slice.
                element_ptr.offset_from_unsigned(slice_ptr),
            )
        }
    }

    pub const fn after(&self) -> &[T] {
        if const { size_of::<T>() == 0 } {
            return self.whole.map;
        }
        let slice_end_ptr = self.whole.map.as_ptr_range().end;
        let element_ptr = ptr::from_ref(self.map);
        unsafe {
            // SAFETY: - By construction, `self.map` points to an element of `self.whole.map`.
            //         - `element_ptr + (slice_end_ptr - element_ptr) = slice_end_ptr`.
            slice::from_raw_parts(
                element_ptr,
                // SAFETY: By construction, `self.map` points to an element of `self.whole.map`,
                //         so the it does not exceed the end of the slice.
                slice_end_ptr.offset_from_unsigned(element_ptr),
            )
        }
    }

    pub const fn element_offset(&self) -> usize {
        if const { size_of::<T>() == 0 } {
            panic!("elements are zero-sized");
        }
        // SAFETY: By construction, `self.map` points to an element of `self.whole.map`,
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
            //         - The offset of an element - `self.map.whole` - from the
            //           origin - `self.whole` if always less than the length of the slice.
            slice::from_raw_parts(
                slice_ptr,
                // SAFETY: By construction, `self.map.whole` points to an element of `self.whole`,
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
            // SAFETY: - By construction, `self.map.whole` points to an element of `self.whole`.
            //         - `element_ptr + (slice_end_ptr - element_ptr) = slice_end_ptr`.
            slice::from_raw_parts(
                element_ptr,
                // SAFETY: By construction, `self.map.whole` points to an element of `self.whole`,
                //         so the it does not exceed the end of the slice.
                slice_end_ptr.offset_from_unsigned(element_ptr),
            )
        }
    }

    pub const fn element_offset(&self) -> usize {
        if const { size_of::<T>() == 0 } {
            panic!("elements are zero-sized");
        }
        // SAFETY: By construction, `self.map.whole` points to an element of `self.whole`,
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
            //         - The offset of a subslice - `self.map` from the
            //           origin - `self.whole` - is always less than or equal to the
            //           length of the slice.
            slice::from_raw_parts(
                slice_ptr,
                // SAFETY: By construction, `self.map` points to a subslice entirely within
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
            // SAFETY: - By construction, `self.map` points to a subslice entirely within
            //           `self.whole`. Thus, the end of said subslice also points within it.
            //         - `subslice_end_ptr + (slice_end_ptr - subslice_end_ptr) = slice_end_ptr`.
            slice::from_raw_parts(
                subslice_end_ptr,
                // SAFETY: By construction, `self.map` points to a subslice entirely within
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
            // SAFETY: By construction, `self.map` points to a subslice entirely within `self.whole`,
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
            //         - The offset of a subslice - `self.map` from the
            //           origin - `self.whole.map` - is always less than or equal to the
            //           length of the slice.
            slice::from_raw_parts(
                slice_ptr,
                // SAFETY: By construction, `self.map` points to a subslice entirely within
                //         `self.whole.map`, so its start always exceeds or is the slice's.
                subslice_ptr.offset_from_unsigned(slice_ptr),
            )
        }
    }

    pub const fn after(&self) -> &[T] {
        if const { size_of::<T>() == 0 } {
            return self.whole.map;
        }
        let slice_end_ptr = self.whole.map.as_ptr_range().end;
        let subslice_end_ptr = self.map.as_ptr_range().end;
        unsafe {
            // SAFETY: - By construction, `self.map` points to a subslice entirely within
            //           `self.whole.map`. Thus, the end of said subslice also points within it.
            //         - `subslice_end_ptr + (slice_end_ptr - subslice_end_ptr) = slice_end_ptr`.
            slice::from_raw_parts(
                subslice_end_ptr,
                // SAFETY: By construction, `self.map` points to a subslice entirely within
                //         `self.whole.map`, so the its end does not exceed the slice's.
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
            // SAFETY: By construction, `self.map` points to a subslice entirely within `self.whole.map`,
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
        let subslice_ptr = self.map.whole.as_ptr();
        unsafe {
            // SAFETY: - `slice_ptr` is derived from a reference.
            //         - The offset of a subslice - `self.map.whole` from the
            //           origin - `self.whole` - is always less than or equal to the
            //           length of the slice.
            slice::from_raw_parts(
                slice_ptr,
                // SAFETY: By construction, `self.map.whole` points to a subslice entirely within
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
            // SAFETY: - By construction, `self.map.whole` points to a subslice entirely within
            //           `self.whole`. Thus, the end of said subslice also points within it.
            //         - `subslice_end_ptr + (slice_end_ptr - subslice_end_ptr) = slice_end_ptr`.
            slice::from_raw_parts(
                subslice_end_ptr,
                // SAFETY: By construction, `self.map.whole` points to a subslice entirely within
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
            // SAFETY: By construction, `self.map.whole` points to a subslice entirely within `self.whole`,
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
