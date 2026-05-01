#![allow(dead_code)]
#![feature(allocator_api, ptr_metadata, layout_for_ptr, sync_nonpoison)]

mod alloc;
mod arc;
pub use arc::{ArcMappedRwLock, ArcReaderLock, UniqueArcMappedRwLock};
mod lock;
pub use lock::{MappedRwLock, MappedRwLockGuard, ReaderLock, ReaderLockGuard};
mod slice;
pub use slice::{
    ArcElementRwLock, ArcSliceReaderLock, ArcSliceRwLock, ElementRwLock, ElementRwLockGuard,
    SliceReaderLock, SliceReaderLockGuard, SliceRwLock, UniqueArcElementRwLock,
    UniqueArcSliceRwLock,
};
mod unique_arc;

#[cold]
fn unlikely<T>(value: T) -> T {
    value
}

#[cold]
fn cold_path() {}
