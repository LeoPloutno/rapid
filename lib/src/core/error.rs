//! Common errors.
use std::{
    convert::Infallible,
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    ops::Range,
};

/// An error that represents invalid indexing with indices.
#[derive(Clone, Copy, Debug)]
pub struct InvalidIndexError {
    index: usize,
    len: usize,
}

impl InvalidIndexError {
    /// Constructs a new `InvalidIndexError`.
    pub fn new(index: usize, len: usize) -> Self {
        Self { index, len }
    }
}

impl From<Infallible> for InvalidIndexError {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

impl Display for InvalidIndexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "{} is an invalid index in a slice of length {}",
            self.index, self.len
        )
    }
}

impl Error for InvalidIndexError {}

/// An error that represents invalid indexing with ranges.
#[derive(Clone, Copy, Debug)]
pub struct InvalidRangeError {
    start: usize,
    end: usize,
    len: usize,
}

impl InvalidRangeError {
    /// Constructs a new `InvalidRangeError`.
    pub fn new(range: Range<usize>, len: usize) -> Self {
        Self {
            start: range.start,
            end: range.end,
            len,
        }
    }
}

impl From<Infallible> for InvalidRangeError {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

impl Display for InvalidRangeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "{}..{} is an invalid range in a slice of length {}",
            self.start, self.end, self.len
        )
    }
}

impl Error for InvalidRangeError {}

/// An error representing an attempt to access an empty container.
#[derive(Clone, Copy, Debug)]
pub struct EmptyError;

impl From<Infallible> for EmptyError {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

impl Display for EmptyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "attempted to access an empty container")
    }
}

impl Error for EmptyError {}

/// An error that represents invalid access.
#[derive(Clone, Debug)]
pub enum AccessError {
    /// Invalid indexing.
    Index(InvalidIndexError),
    /// Invalid range indexing.
    Range(InvalidRangeError),
    /// Accessing an empty container.
    Empty(EmptyError),
}

impl From<Infallible> for AccessError {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

impl From<InvalidIndexError> for AccessError {
    fn from(value: InvalidIndexError) -> Self {
        Self::Index(value)
    }
}

impl From<InvalidRangeError> for AccessError {
    fn from(value: InvalidRangeError) -> Self {
        Self::Range(value)
    }
}

impl From<EmptyError> for AccessError {
    fn from(value: EmptyError) -> Self {
        Self::Empty(value)
    }
}

impl Display for AccessError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Index(err) => write!(f, "invalid index: {}", err),
            Self::Range(err) => write!(f, "invalid range: {}", err),
            Self::Empty(_) => write!(f, "empty container"),
        }
    }
}

impl Error for AccessError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Index(err) => Some(err),
            Self::Range(err) => Some(err),
            Self::Empty(err) => Some(err),
        }
    }
}

/// A miscellaneous error used by [`run`].
///
/// [`run`]: crate::run
#[derive(Clone, Debug)]
pub enum CommError {
    /// The error arose in the main thread.
    Main,
    /// The error arose in a leading thread.
    Leading {
        /// The index of the group the thread is assigned to.
        group: usize,
    },
    /// The error arose in an inner thread.
    Inner {
        /// The image the thread is assigned to.
        image: usize,
        /// The index of the group the thread is assigned to.
        group: usize,
    },
    /// The error arose in a trailing thread.
    Trailing {
        /// The index of the group the thread is assigmed to.
        group: usize,
    },
}

impl From<Infallible> for CommError {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

impl Display for CommError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Main => write!(f, "something happened in the main thread"),
            Self::Leading { group } => write!(
                f,
                "something happened in a thread dedicated to group #{} in the first image",
                group
            ),
            Self::Inner { image, group } => write!(
                f,
                "something happened in a thread dedicated to group #{} in image #{}",
                group, image
            ),
            Self::Trailing { group } => write!(
                f,
                "something happened in a thread dedicated to group #{} in the last image",
                group
            ),
        }
    }
}

impl Error for CommError {}
