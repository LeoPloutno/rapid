use crate::{ArcMappedRwLock, MappedRwLock, arc::InnerArc};
use std::{
    alloc::{Allocator, AllocatorClone, Global},
    iter::FusedIterator,
    mem,
    num::NonZeroUsize,
    process,
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};

pub struct ChunksExact<T, U: ?Sized = dyn Send + Sync + 'static, A: Allocator = Global> {
    pub(crate) remainder_lock: MappedRwLock<[T], U>,
    pub(crate) slice: NonNull<[T]>,
    pub(crate) chunk_size: NonZeroUsize,
    pub(crate) allocator: A,
}

impl<T, U: ?Sized, A: Allocator> ChunksExact<T, U, A> {
    pub const fn remainder(&self) -> &MappedRwLock<[T], U> {
        &self.remainder_lock
    }
}

impl<T, U: ?Sized, A: Allocator> Drop for ChunksExact<T, U, A> {
    fn drop(&mut self) {
        // SAFETY: `self.remainder_lock.inner` has been allocated as a part of an `InnerArc`.
        let (allocation, layout) = unsafe { InnerArc::from_lock(self.remainder_lock.inner) };
        if unsafe { InnerArc::decrement_shared_counter(allocation, Ordering::Release) } {
            atomic::fence(Ordering::Acquire);
            if const { mem::needs_drop::<InnerArc<[T]>>() } {
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

unsafe impl<T, U, A> Send for ChunksExact<T, U, A>
where
    T: Send + Sync,
    U: Send + Sync + ?Sized,
    A: Allocator + Send,
{
}

unsafe impl<T, U, A> Sync for ChunksExact<T, U, A>
where
    T: Send + Sync,
    U: Send + Sync + ?Sized,
    A: Allocator + Sync,
{
}

impl<T, U: ?Sized, A: AllocatorClone> Iterator for ChunksExact<T, U, A> {
    type Item = ArcMappedRwLock<[T], U, A>;

    fn next(&mut self) -> Option<Self::Item> {
        let (ptr, len) = self.slice.to_raw_parts();
        if len > 0 {
            let ptr = ptr.cast::<T>();
            unsafe {
                self.slice = NonNull::from_raw_parts(
                    // SAFETY: By construction, 'ptr' points to a slice which contains at least `self.chunk_size` elements.
                    ptr.add(self.chunk_size.get()),
                    // SAFETY: Checked above that 'len > 0`.
                    //         By construction, `len` is a multiple of `self.chunk_size`.
                    len.unchecked_sub(self.chunk_size.get()),
                );
            }
            if crate::unlikely(unsafe {
                // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
                InnerArc::increment_shared_counter(
                    // SAFETY: `self.remainder_lock.inner` has been allocated as a part of an `InnerArc`.
                    InnerArc::from_lock(self.remainder_lock.inner).0,
                    Ordering::Release,
                )
            }) {
                process::abort()
            }
            Some(ArcMappedRwLock {
                lock: MappedRwLock {
                    inner: self.remainder_lock.inner,
                    subfield: NonNull::from_raw_parts(ptr, self.chunk_size.get()),
                },
                allocator: self.allocator.clone(),
            })
        } else {
            None
        }
    }
}

impl<T, U: ?Sized, A: AllocatorClone> DoubleEndedIterator for ChunksExact<T, U, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (ptr, mut len) = self.slice.to_raw_parts();
        if len > 0 {
            // SAFETY: Checked above that 'len > 0`.
            //         By construction, `len` is a multiple of `self.chunk_size`.
            len = unsafe { len.unchecked_sub(self.chunk_size.get()) };
            let ptr = ptr.cast::<T>();
            self.slice = NonNull::from_raw_parts(ptr, len);
            if crate::unlikely(unsafe {
                // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
                InnerArc::increment_shared_counter(
                    // SAFETY: `self.remainder_lock.inner` has been allocated as a part of an `InnerArc`.
                    InnerArc::from_lock(self.remainder_lock.inner).0,
                    Ordering::Release,
                )
            }) {
                process::abort()
            }
            Some(ArcMappedRwLock {
                lock: MappedRwLock {
                    inner: self.remainder_lock.inner,
                    subfield: NonNull::from_raw_parts(
                        // SAFETY: By construction, 'ptr' points to a slice which contains at least `len + self.chunk_size` elements.
                        unsafe { ptr.add(len) },
                        self.chunk_size.get(),
                    ),
                },
                allocator: self.allocator.clone(),
            })
        } else {
            None
        }
    }
}

impl<T, U: ?Sized, A: AllocatorClone> ExactSizeIterator for ChunksExact<T, U, A> {
    fn len(&self) -> usize {
        self.slice.len() / self.chunk_size
    }
}

impl<T, U: ?Sized, A: AllocatorClone> FusedIterator for ChunksExact<T, U, A> {}

impl<T, U: ?Sized, A: AllocatorClone> Clone for ChunksExact<T, U, A> {
    fn clone(&self) -> Self {
        if unsafe {
            // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
            InnerArc::increment_shared_counter(
                // SAFETY: `lock.lock.inner` has been allocated as a part of an `InnerArc`.
                InnerArc::from_lock(self.remainder_lock.inner).0,
                Ordering::Release,
            )
        } {
            process::abort();
        }
        Self {
            remainder_lock: MappedRwLock { ..self.remainder_lock },
            allocator: self.allocator.clone(),
            ..*self
        }
    }
}
