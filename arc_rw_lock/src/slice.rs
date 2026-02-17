use std::{alloc::Allocator, mem, process, sync::atomic::Ordering};

use crate::{
    UniqueArcSliceRwLock,
    arc::InnerArc,
    slice::{iter::IterMut, iter_mut::Iter},
};

mod iter;
mod iter_mut;

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
