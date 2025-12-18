pub(crate) use lock::Lock;
use std::sync::atomic::{AtomicBool, Ordering};

mod lock;

pub(crate) struct PoisonLock {
    pub(crate) lock: Lock,
    poison: AtomicBool,
}

impl PoisonLock {
    /// Creates a new unlocked lock without poison.
    pub(crate) const fn new() -> Self {
        Self {
            lock: Lock::new(),
            poison: AtomicBool::new(false),
        }
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poison.load(Ordering::Acquire)
    }

    pub(crate) fn poison(&self) {
        self.poison.store(true, Ordering::Release);
    }

    pub(crate) fn remove_poison(&self) {
        self.poison.store(false, Ordering::Release);
    }
}

pub(crate) struct InnerRwLock<T: ?Sized> {
    pub(crate) lock: PoisonLock,
    pub(crate) data: T,
}
