use crate::{MappedRwLock, UniqueArcMappedRwLock, arc::InnerArc};
use std::{
    alloc::{Allocator, AllocatorClone, Global},
    iter::FusedIterator,
    mem, process,
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};

pub struct IterMut<T, U: ?Sized = dyn Send + Sync + 'static, A: Allocator = Global> {
    pub(crate) lock: MappedRwLock<[T], U>,
    pub(crate) allocator: A,
}

impl<T, U: ?Sized, A: Allocator> Drop for IterMut<T, U, A> {
    fn drop(&mut self) {
        // SAFETY: `self.lock.inner` has been allocated as a part of an `InnerArc`.
        let (allocation, layout) = unsafe { InnerArc::from_lock(self.lock.inner) };
        if unsafe { InnerArc::decrement_unique_counter(allocation, Ordering::Release) } {
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

unsafe impl<T, U, A> Send for IterMut<T, U, A>
where
    T: Send + Sync,
    U: Send + Sync + ?Sized,
    A: Allocator + Send,
{
}

unsafe impl<T, U, A> Sync for IterMut<T, U, A>
where
    T: Send + Sync,
    U: Send + Sync + ?Sized,
    A: Allocator + Sync,
{
}

impl<T, U: ?Sized, A: AllocatorClone> Iterator for IterMut<T, U, A> {
    type Item = UniqueArcMappedRwLock<T, U, A>;

    fn next(&mut self) -> Option<Self::Item> {
        let (ptr, len) = self.lock.subfield.to_raw_parts();
        if len > 0 {
            let ptr = ptr.cast::<T>();
            unsafe {
                self.lock.subfield = NonNull::from_raw_parts(
                    // SAFETY: By construction, `ptr` points to a slice which contains at least one element.
                    ptr.add(1),
                    // SAFETY: Checked above that `len > 0`.
                    len.unchecked_sub(1),
                );
            }
            if crate::unlikely(unsafe {
                // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
                InnerArc::increment_unique_counter(
                    // SAFETY: `self.lock.inner` has been allocated as a part of an `InnerArc`.
                    InnerArc::from_lock(self.lock.inner).0,
                    Ordering::Release,
                )
            }) {
                process::abort()
            }
            Some(UniqueArcMappedRwLock {
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

impl<T, U: ?Sized, A: AllocatorClone> DoubleEndedIterator for IterMut<T, U, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (ptr, mut len) = self.lock.subfield.to_raw_parts();
        if len > 0 {
            // SAFETY: Checked above that `len > 0`.
            len = unsafe { len.unchecked_sub(1) };
            let ptr = ptr.cast::<T>();
            self.lock.subfield = NonNull::from_raw_parts(ptr, len);
            if crate::unlikely(unsafe {
                // SAFETY: By construction, the pointer points to a valid and live instance of `InnerArc`.
                InnerArc::increment_unique_counter(
                    // SAFETY: `self.lock.inner` has been allocated as a part of an `InnerArc`.
                    InnerArc::from_lock(self.lock.inner).0,
                    Ordering::Release,
                )
            }) {
                process::abort()
            }
            Some(UniqueArcMappedRwLock {
                lock: MappedRwLock {
                    inner: self.lock.inner,
                    // SAFETY: By construction, `ptr` points to a slice which contains at least `len + 1` elements.
                    subfield: unsafe { ptr.add(len) },
                },
                allocator: self.allocator.clone(),
            })
        } else {
            None
        }
    }
}

impl<T, U: ?Sized, A: AllocatorClone> ExactSizeIterator for IterMut<T, U, A> {
    fn len(&self) -> usize {
        self.lock.subfield.len()
    }
}

impl<T, U: ?Sized, A: AllocatorClone> FusedIterator for IterMut<T, U, A> {}
