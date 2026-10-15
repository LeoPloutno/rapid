use crate::{
    ArcMappedRwLock, ArcReaderLock, MappedRwLock, MappedRwLockGuard, ReaderLock, ReaderLockGuard, UniqueArcMappedRwLock,
};
use std::alloc::Global;

mod iter;
pub use iter::Iter;

mod iter_mut;
pub use iter_mut::IterMut;

mod chunks_exact;
pub use chunks_exact::ChunksExact;

mod chunks_exact_mut;
pub use chunks_exact_mut::ChunksExactMut;

pub type ElementRwLock<T> = MappedRwLock<T>;

pub type ElementRwLockGuard<'a, T> = MappedRwLockGuard<'a, T>;

pub type ArcElementRwLock<T, A = Global> = ArcMappedRwLock<T, [T], A>;

pub type UniqueArcElementRwLock<T, A = Global> = UniqueArcMappedRwLock<T, [T], A>;

pub type SliceRwLock<T> = MappedRwLock<[T]>;

pub type SliceRwLockGuard<'a, T> = MappedRwLockGuard<'a, [T]>;

pub type SliceReaderLock<T> = ReaderLock<[T]>;

pub type SliceReaderLockGuard<'a, T> = ReaderLockGuard<'a, [T]>;

pub type ArcSliceRwLock<T, A = Global> = ArcMappedRwLock<[T], [T], A>;

pub type ArcSliceReaderLock<T, A = Global> = ArcReaderLock<[T], A>;

pub type UniqueArcSliceRwLock<T, A = Global> = UniqueArcMappedRwLock<[T], [T], A>;
