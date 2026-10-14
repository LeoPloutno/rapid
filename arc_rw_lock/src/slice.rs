use crate::{
    ArcMappedRwLock, ArcReaderLock, MappedRwLock, MappedRwLockGuard, ReaderLock, ReaderLockGuard, UniqueArcMappedRwLock,
};
use std::{alloc::Global, ptr::NonNull, range::Range};

mod iter;
pub use iter::Iter;

mod iter_mut;
pub use iter_mut::IterMut;

mod chunks_exact;
pub use chunks_exact::ChunksExact;

mod chunks_exact_mut;
pub use chunks_exact_mut::ChunksExactMut;

pub type ElementRwLock<T> = MappedRwLock<T, [T]>;

pub type ElementRwLockGuard<'a, T> = MappedRwLockGuard<'a, T>;

pub type ArcElementRwLock<T, A = Global> = ArcMappedRwLock<T, [T], A>;

pub type UniqueArcElementRwLock<T, A = Global> = UniqueArcMappedRwLock<T, [T], A>;

pub type SliceRwLock<T> = MappedRwLock<[T], [T]>;

pub type SliceRwLockGuard<'a, T> = MappedRwLockGuard<'a, [T]>;

pub type SliceReaderLock<T> = ReaderLock<[T]>;

pub type SliceReaderLockGuard<'a, T> = ReaderLockGuard<'a, [T]>;

pub type ArcSliceRwLock<T, A = Global> = ArcMappedRwLock<[T], [T], A>;

pub type ArcSliceReaderLock<T, A = Global> = ArcReaderLock<[T], A>;

pub type UniqueArcSliceRwLock<T, A = Global> = UniqueArcMappedRwLock<[T], [T], A>;

impl<T, U> MappedRwLock<U, [T]> {
    pub const fn len(&self) -> usize {
        // SAFETY: By construction, `inner` points to live and valid data.
        unsafe { &raw mut (*self.inner.as_ptr()).data }.len()
    }
}

impl<T> ElementRwLock<T> {
    pub const fn element_offset(&self) -> usize {
        // SAFETY: By construction, `inner` points to live and valid data.
        let (ptr_whole, _) = unsafe { &raw mut (*self.inner.as_ptr()).data }.to_raw_parts();
        // SAFETY: The offset of `data` guarantees `ptr` is non-null.
        let ptr_whole = unsafe { NonNull::new_unchecked(ptr_whole) }.cast::<T>();
        let (ptr, _) = self.subfield.to_raw_parts();
        let ptr = ptr.cast::<T>();
        // SAFETY: By construction, `ptr` points to a subslice of `ptr_whole`.
        unsafe { ptr.offset_from_unsigned(ptr_whole) }
    }
}

impl<T> SliceRwLock<T> {
    pub const fn subslice_range(&self) -> Range<usize> {
        // SAFETY: By construction, `inner` points to live and valid data.
        let (ptr_whole, _) = unsafe { &raw mut (*self.inner.as_ptr()).data }.to_raw_parts();
        // SAFETY: The offset of `data` guarantees `ptr` is non-null.
        let ptr_whole = unsafe { NonNull::new_unchecked(ptr_whole) }.cast::<T>();
        let (ptr, len) = self.subfield.to_raw_parts();
        let ptr = ptr.cast::<T>();
        // SAFETY: By construction, `ptr` points to a subslice of `ptr_whole`.
        let start = unsafe { ptr.offset_from_unsigned(ptr_whole) };
        Range {
            start,
            // SAFETY: By construction, `start + len` points within or right outside the allocation.
            end: unsafe { start.unchecked_add(len) },
        }
    }
}
