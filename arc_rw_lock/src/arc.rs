use std::{
    alloc::{Allocator, Global},
    sync::atomic::AtomicUsize,
};

use crate::{inner::InnerRwLock, lock::MappedRwLock};

struct InnerArc<T: ?Sized> {
    counter: AtomicUsize,
    lock: InnerRwLock<T>,
}

pub struct ArcMappedRwLock<T: ?Sized, U: ?Sized = dyn 'static + Send + Sync, A: Allocator = Global>
{
    pub(crate) lock: MappedRwLock<T, U>,
    pub(crate) allocator: A,
}
