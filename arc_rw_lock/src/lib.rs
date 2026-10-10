#![allow(dead_code)]
#![feature(allocator_ext, ptr_metadata, sync_nonpoison, unsize, coerce_unsized)]

mod alloc;
mod arc;
pub use arc::{ArcMappedRwLock, ArcReaderLock, UniqueArcMappedRwLock};
mod lock;
pub use lock::{MappedRwLock, MappedRwLockGuard, ReaderLock, ReaderLockGuard};
mod slice;
pub use slice::{
    ArcElementRwLock, ArcSliceReaderLock, ArcSliceRwLock, ElementRwLock, ElementRwLockGuard, SliceReaderLock,
    SliceReaderLockGuard, SliceRwLock, UniqueArcElementRwLock, UniqueArcSliceRwLock,
};
mod unique_arc;

#[cold]
fn unlikely<T>(value: T) -> T {
    value
}

#[cold]
fn cold_path() {}
