use std::{
    ptr::NonNull,
    sync::{LockResult, PoisonError, TryLockError, TryLockResult, nonpoison::WouldBlock},
};

use crate::inner::{InnerRwLock, Lock, PoisonLock};

#[derive(Clone)]
pub struct MappedRwLock<T: ?Sized, U: ?Sized = dyn 'static + Send + Sync> {
    inner: NonNull<InnerRwLock<U>>,
    pub(crate) subfield: NonNull<T>,
}

pub struct MappedRwLockWriteGuard<'a, T: ?Sized> {
    lock: &'a PoisonLock,
    data: &'a mut T,
}

pub struct MappedRwLockReadWholeGuard<'a, T: ?Sized> {
    lock: &'a Lock,
    data: &'a T,
}

impl<T: ?Sized, U: ?Sized> MappedRwLock<T, U> {
    pub fn read(&self) -> &T {
        // SAFETY: - By construcion, `subfield` points to live and valid data.
        //         - By construcion, no other lock has mutable access to
        //           the subfield.
        unsafe { self.subfield.as_ref() }
    }

    pub fn write(&mut self) -> MappedRwLockWriteGuard<'_, T> {
        // SAFETY: By construction, `inner` points to live and valid data.
        let lock = unsafe { &(*self.inner.as_ptr()).lock };
        lock.lock.write();
        MappedRwLockWriteGuard {
            lock,
            // SAFETY: - By construction, `subfield` points to live and valid data.
            //         - Aliasing rules are enforced via synchronization.
            data: unsafe { self.subfield.as_mut() },
        }
    }

    pub fn try_write(&mut self) -> Result<MappedRwLockWriteGuard<'_, T>, WouldBlock> {
        // SAFETY: By construction, `inner` points to live and valid data.
        let lock = unsafe { &(*self.inner.as_ptr()).lock };
        if lock.lock.try_write() {
            Ok(MappedRwLockWriteGuard {
                lock,
                // SAFETY: - By construction, `subfield` points to live and valid data.
                //         - Aliasing rules are enforced via synchronization.
                data: unsafe { self.subfield.as_mut() },
            })
        } else {
            Err(WouldBlock)
        }
    }

    pub fn read_whole(&self) -> LockResult<MappedRwLockReadWholeGuard<'_, U>> {
        // SAFETY: By construction, `inner` points to live and valid data.
        let lock = unsafe { &(*self.inner.as_ptr()).lock };
        lock.lock.read_whole();
        let guard = MappedRwLockReadWholeGuard {
            lock: &lock.lock,
            // SAFETY: - By construction, `inner` points to live and valid data.
            //         - Aliasing rules are enforced via synchronization.
            data: unsafe { &(*self.inner.as_ptr()).data },
        };
        if lock.is_poisoned() {
            Ok(guard)
        } else {
            Err(PoisonError::new(guard))
        }
    }

    pub fn try_read_whole(&self) -> TryLockResult<MappedRwLockReadWholeGuard<'_, U>> {
        // SAFETY: By construction, `inner` points to live and valid data.
        let lock = unsafe { &(*self.inner.as_ptr()).lock };
        if lock.lock.try_read_whole() {
            let guard = MappedRwLockReadWholeGuard {
                lock: &lock.lock,
                // SAFETY: - By construction, `inner` points to live and valid data.
                //         - Aliasing rules are enforced via synchronization.
                data: unsafe { &(*self.inner.as_ptr()).data },
            };
            if lock.is_poisoned() {
                Err(TryLockError::Poisoned(PoisonError::new(guard)))
            } else {
                Ok(guard)
            }
        } else {
            Err(TryLockError::WouldBlock)
        }
    }
}
