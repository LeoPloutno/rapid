use super::InnerArc;
use crate::lock::ReaderLock;
use std::{
    alloc::{Allocator, AllocatorClone, Global},
    borrow::Borrow,
    convert::AsRef,
    fmt::Debug,
    marker::Unsize,
    mem::{self, MaybeUninit, needs_drop},
    ops::{CoerceUnsized, Deref},
    process,
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};

pub struct ArcReaderLock<T: ?Sized, A: Allocator = Global> {
    pub(crate) lock: ReaderLock<T>,
    pub(crate) allocator: A,
}

impl<T: ?Sized, A: Allocator> ArcReaderLock<T, A> {
    pub const fn allocator(&self) -> &A {
        &self.allocator
    }
}

impl<T, A: Allocator> ArcReaderLock<MaybeUninit<T>, A> {
    pub const unsafe fn assume_init(self) -> ArcReaderLock<T, A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let ReaderLock(lock) = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        ArcReaderLock {
            lock: ReaderLock(lock.cast()),
            allocator,
        }
    }
}

impl<T, A: Allocator> ArcReaderLock<[MaybeUninit<T>], A> {
    pub const unsafe fn assume_init(self) -> ArcReaderLock<[T], A> {
        // SAFETY: All fields of `self` are forgotten immediately after
        //         reading them out of the pointers.
        let ReaderLock(lock) = unsafe { (&raw const self.lock).read() };
        let allocator = unsafe { (&raw const self.allocator).read() };
        mem::forget(self);
        ArcReaderLock {
            lock: ReaderLock({
                let (ptr, len) = lock.to_raw_parts();
                NonNull::from_raw_parts(ptr, len)
            }),
            allocator,
        }
    }
}

impl<T: ?Sized, A: Allocator> Drop for ArcReaderLock<T, A> {
    fn drop(&mut self) {
        // SAFETY: `self.lock.0` has been allocated as a part of an `InnerArc`.
        let (allocation, layout) = unsafe { InnerArc::from_lock(self.lock.0) };
        if unsafe { InnerArc::decrement_shared_counter(allocation, Ordering::Release) } {
            atomic::fence(Ordering::Acquire);
            if const { needs_drop::<InnerArc<T>>() } {
                // SAFETY: - By construction, `allocation` points to live and valid data.
                //         - Ensured this was the last handle to this allocation.
                unsafe {
                    allocation.drop_in_place();
                }
            }
            // SAFETY: By construction, this allocation has been allocated by this allocator.
            unsafe {
                self.allocator.deallocate(allocation.cast(), layout);
            }
        }
    }
}

impl<T: ?Sized, A: Allocator> Deref for ArcReaderLock<T, A> {
    type Target = ReaderLock<T>;

    fn deref(&self) -> &ReaderLock<T> {
        &self.lock
    }
}

impl<T: ?Sized, A: Allocator> AsRef<ReaderLock<T>> for ArcReaderLock<T, A> {
    fn as_ref(&self) -> &ReaderLock<T> {
        &self.lock
    }
}

impl<T: ?Sized, A: Allocator> Borrow<ReaderLock<T>> for ArcReaderLock<T, A> {
    fn borrow(&self) -> &ReaderLock<T> {
        &self.lock
    }
}

impl<T, U, A> CoerceUnsized<ArcReaderLock<U, A>> for ArcReaderLock<T, A>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
    A: Allocator,
{
}

unsafe impl<T, A> Send for ArcReaderLock<T, A>
where
    T: Send + Sync + ?Sized,
    A: Allocator + Send,
{
}

unsafe impl<T, A> Sync for ArcReaderLock<T, A>
where
    T: Send + Sync + ?Sized,
    A: Allocator + Sync,
{
}

impl<T: ?Sized, A: Allocator> Debug for ArcReaderLock<T, A> {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<T: ?Sized, A: AllocatorClone> Clone for ArcReaderLock<T, A> {
    fn clone(&self) -> Self {
        if unsafe {
            // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
            InnerArc::increment_shared_counter(
                // SAFETY: `self.lock.0` has been allocated as a part of an `InnerArc`.
                InnerArc::from_lock(self.lock.0).0,
                Ordering::Relaxed,
            )
        } {
            process::abort();
        }
        Self {
            lock: ReaderLock(self.lock.0),
            allocator: self.allocator.clone(),
        }
    }
}
