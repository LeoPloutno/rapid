use atomic_wait;
use std::{
    hint, process,
    sync::atomic::{self, AtomicU32, Ordering},
};

use crate::unlikely;

pub(crate) struct Lock(AtomicU32);

impl Lock {
    const EMPTY: u32 = 0;
    const WRITE_FLAG: u32 = 1;
    const COUNTER_ONE: u32 = 1 << Self::WRITE_FLAG.trailing_ones();
    const COUNTER_MASK: u32 = !0 & !Self::WRITE_FLAG;
    const COUNTER_MAX: u32 = Self::COUNTER_MASK >> Self::COUNTER_MASK.trailing_zeros();

    /// Constructs an unlocked `Lock`.
    pub(crate) const fn new() -> Self {
        Self(AtomicU32::new(Self::EMPTY))
    }

    /// Blocks until there are no global readers and
    /// locks with subfield write access.
    pub(crate) fn write(&self) {
        let mut loaded = self.0.load(Ordering::Relaxed);
        loop {
            if loaded == Self::EMPTY {
                match self.0.compare_exchange_weak(
                    loaded,
                    Self::WRITE_FLAG | Self::COUNTER_ONE,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            } else if loaded & Self::WRITE_FLAG != 0 {
                if unlikely(loaded >> Self::COUNTER_MASK.trailing_zeros() == Self::COUNTER_MAX) {
                    process::abort();
                }
                match self.0.compare_exchange_weak(
                    loaded,
                    // SAFETY: Checked above that the counter will not overflow
                    // upon an increment.
                    unsafe { loaded.unchecked_add(Self::COUNTER_ONE) },
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            } else {
                atomic_wait::wait(&self.0, loaded);
                loaded = self.0.load(Ordering::Relaxed);
            }
        }
    }

    /// Attempts to lock with subfield write access without blocking
    /// and returns whether the operation succeeded.
    pub(crate) fn try_write(&self) -> bool {
        let mut loaded = self.0.load(Ordering::Relaxed);
        loop {
            if loaded == Self::EMPTY {
                match self.0.compare_exchange_weak(
                    loaded,
                    Self::WRITE_FLAG | Self::COUNTER_ONE,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return true,
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            } else if loaded & Self::WRITE_FLAG != 0 {
                if unlikely(loaded >> Self::COUNTER_MASK.trailing_zeros() == Self::COUNTER_MAX) {
                    process::abort();
                }
                match self.0.compare_exchange_weak(
                    loaded,
                    // SAFETY: Checked above that the counter will not overflow
                    // upon an increment.
                    unsafe { loaded.unchecked_add(Self::COUNTER_ONE) },
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return true,
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            } else {
                return false;
            }
        }
    }

    /// Blocks until there are no subfield writers and
    /// locks with global read access.
    pub(crate) fn read_whole(&self) {
        let mut loaded = self.0.load(Ordering::Relaxed);
        loop {
            if loaded == Self::EMPTY {
                match self
                    .0
                    .compare_exchange_weak(loaded, Self::COUNTER_ONE, Ordering::Acquire, Ordering::Relaxed)
                {
                    Ok(_) => return,
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            } else if loaded & Self::WRITE_FLAG == 0 {
                if unlikely(loaded >> Self::COUNTER_MASK.trailing_zeros() == Self::COUNTER_MAX) {
                    process::abort();
                }
                match self.0.compare_exchange_weak(
                    loaded,
                    // SAFETY: Checked above that the counter will not overflow
                    // upon an increment.
                    unsafe { loaded.unchecked_add(Self::COUNTER_ONE) },
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            } else {
                atomic_wait::wait(&self.0, loaded);
                loaded = self.0.load(Ordering::Relaxed);
            }
        }
    }

    /// Attempts to lock with global read access without blocking
    /// and returns whether the operation succeeded.
    pub(crate) fn try_read_whole(&self) -> bool {
        let mut loaded = self.0.load(Ordering::Relaxed);
        loop {
            if loaded == Self::EMPTY {
                match self
                    .0
                    .compare_exchange_weak(loaded, Self::COUNTER_ONE, Ordering::Acquire, Ordering::Relaxed)
                {
                    Ok(_) => return true,
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            } else if loaded & Self::WRITE_FLAG == 0 {
                if unlikely(loaded >> Self::COUNTER_MASK.trailing_zeros() == Self::COUNTER_MAX) {
                    process::abort();
                }
                match self.0.compare_exchange_weak(
                    loaded,
                    // SAFETY: Checked above that the counter will not overflow
                    // upon an increment.
                    unsafe { loaded.unchecked_add(Self::COUNTER_ONE) },
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return true,
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            } else {
                return false;
            }
        }
    }

    /// Decrements the writers counter assuming it
    /// is non-zero.
    ///
    /// # Safety
    ///
    /// The writers counter must be non-zero.
    pub(crate) unsafe fn drop_writer_unchecked(&self) {
        let mut loaded = self.0.load(Ordering::Relaxed);
        loop {
            let counter = loaded >> Self::COUNTER_MASK.trailing_zeros();
            if counter == 0 {
                // SAFETY: User-upheld invariant.
                unsafe {
                    hint::unreachable_unchecked();
                }
            } else if counter == 1 {
                match self
                    .0
                    .compare_exchange_weak(loaded, Self::EMPTY, Ordering::Release, Ordering::Relaxed)
                {
                    Ok(_) => {
                        atomic_wait::wake_all(&self.0);
                        return;
                    }
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            } else {
                match self.0.compare_exchange_weak(
                    loaded,
                    // SAFETY: Cheched above that the counter is non-zero.
                    unsafe { loaded.unchecked_sub(Self::COUNTER_ONE) },
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(current) => {
                        hint::spin_loop();
                        loaded = current;
                    }
                }
            }
        }
    }

    /// Decrements the readers counter assuming it
    /// is non-zero.
    ///
    /// # Sefety
    ///
    /// The readers counter must be non-zero.
    pub(crate) unsafe fn drop_whole_reader_unchecked(&self) {
        if self.0.fetch_sub(Self::COUNTER_ONE, Ordering::Release) == Self::COUNTER_ONE {
            atomic::fence(Ordering::Acquire);
            atomic_wait::wake_all(&self.0);
        }
    }
}
