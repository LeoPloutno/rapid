use std::{
    alloc::{Allocator, Global},
    mem::needs_drop,
    process,
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};

use crate::{MappedRwLock, UniqueArcElementRwLock, arc::InnerArc, unlikely};

pub struct Iter<T, A: Allocator = Global> {
    pub(crate) lock: MappedRwLock<[T], [T]>,
    pub(crate) allocator: A,
}

impl<T, A: Allocator> Drop for Iter<T, A> {
    fn drop(&mut self) {
        // SAFETY: `self.lock.inner` has been allocated as a part of an `InnerArc`.
        let (allocation, layout) = unsafe { InnerArc::from_lock(self.lock.inner) };
        if unsafe { InnerArc::decrement_unique_counter(allocation, Ordering::Release) } {
            atomic::fence(Ordering::Acquire);
            if const { needs_drop::<InnerArc<[T]>>() } {
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

impl<T, A: Allocator + Clone> Iterator for Iter<T, A> {
    type Item = UniqueArcElementRwLock<T, A>;

    fn next(&mut self) -> Option<Self::Item> {
        let (ptr, len) = self.lock.subfield.to_raw_parts();
        if len > 0 {
            let ptr = ptr.cast::<T>();
            unsafe {
                self.lock.subfield = NonNull::from_raw_parts(
                    // SAFETY: `ptr` points to a slice which contains at least one element.
                    ptr.add(1),
                    len.unchecked_sub(1),
                );
            }
            if unlikely(unsafe {
                // SAFETY: By construction, the calculated pointer points to a valid and live instance of `InnerArc`.
                InnerArc::increment_shared_counter(
                    // SAFETY: `self.lock.inner` has been allocated as a part of an `InnerArc`.
                    InnerArc::from_lock(self.lock.inner).0,
                    Ordering::Release,
                )
            }) {
                process::abort()
            }
            Some(UniqueArcElementRwLock {
                lock: MappedRwLock {
                    inner: self.lock.inner,
                    subfield: ptr,
                },
                allocator: self.allocator.clone(),
            })
        } else {
            None
        }
    }
}

impl<T, A: Allocator + Clone> DoubleEndedIterator for Iter<T, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (ptr, mut len) = self.lock.subfield.to_raw_parts();
        if len > 0 {
            // SAFETY: Checked above that `len > 0`.
            len = unsafe { len.unchecked_sub(1) };
            let ptr = ptr.cast::<T>();
            self.lock.subfield = NonNull::from_raw_parts(ptr, len);
            if unlikely(unsafe {
                // SAFETY: By construction, the calculated pointer points to a valid and live instance of `InnerArc`.
                InnerArc::increment_shared_counter(
                    // SAFETY: `self.lock.inner` has been allocated as a part of an `InnerArc`.
                    InnerArc::from_lock(self.lock.inner).0,
                    Ordering::Release,
                )
            }) {
                process::abort()
            }
            Some(UniqueArcElementRwLock {
                lock: MappedRwLock {
                    inner: self.lock.inner,
                    // SAFETY: `ptr` points to a slice which contains at least `len + 1` elements.
                    subfield: unsafe { ptr.add(len) },
                },
                allocator: self.allocator.clone(),
            })
        } else {
            None
        }
    }
}
