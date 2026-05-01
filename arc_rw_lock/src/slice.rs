use crate::{
    ArcMappedRwLock, ArcReaderLock, MappedRwLock, MappedRwLockGuard, ReaderLock, ReaderLockGuard,
    UniqueArcMappedRwLock,
    arc::InnerArc,
    slice::{iter::IterMut, iter_mut::Iter},
};
use std::{
    alloc::{Allocator, Global},
    mem,
    ops::Range,
    process,
    ptr::NonNull,
    sync::atomic::Ordering,
};

mod iter;
mod iter_mut;

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
        start
            ..(
                // SAFETY: By construction, `start + len` points within or right outside the allocation.
                unsafe { start.unchecked_add(len) }
            )
    }
}

impl<T, A: Allocator> UniqueArcSliceRwLock<T, A> {
    pub fn iter(self) -> Iter<T, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        // SAFETY: `lock.inner` has been allocated as a part of an `InnerArc`.
        let allocation = unsafe { InnerArc::from_lock(lock.inner).0 };
        unsafe {
            InnerArc::decrement_unique_counter(allocation, Ordering::Relaxed);
            if InnerArc::increment_shared_counter(allocation, Ordering::Release) {
                process::abort();
            }
        }
        Iter { lock, allocator }
    }

    pub fn iter_mut(self) -> IterMut<T, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let lock = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        IterMut { lock, allocator }
    }
}
