use std::{iter::FusedIterator, marker::PhantomData, num::NonZero, ptr::NonNull};

#[cold]
fn unlikely<T>(value: T) -> T {
    value
}

#[cold]
fn cold_path() {}

pub(crate) struct StrideMut<'a, T> {
    start: NonNull<T>,
    end: NonNull<T>,
    stride: NonZero<usize>,
    phantom: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for StrideMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            // SAFETY: By construction, `start` points to valid and live data.
            //         The offseting guarantees no other iterator has access to this element.
            let ret = Some(unsafe { self.start.as_mut() });
            // SAFETY: `end` is a multiple of `stride` apart from `start` and is within or right outside the allocation.
            //         Checked above that `start < end`.
            self.start = unsafe { self.start.add(self.stride.into()) };
            ret
        } else {
            cold_path();
            None
        }
    }
}

impl<'a, T> DoubleEndedIterator for StrideMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            // SAFETY: `end` is a multiple of `stride` apart from `start` and is within or right outside the allocation.
            //         Checked above that `start < end`.
            self.end = unsafe { self.end.sub(self.stride.into()) };
            // SAFETY: By construction, `end` points to or right outside valid and live data.
            //         Moved it in-bounds above.
            //         The offseting guarantees no other iterator has access to this element.
            Some(unsafe { self.end.as_mut() })
        } else {
            cold_path();
            None
        }
    }
}

impl<'a, T> ExactSizeIterator for StrideMut<'a, T> {
    fn len(&self) -> usize {
        if self.start < self.end {
            // SAFETY: - Checked above that `start < end`.
            //         - By construction, both pointers are derived from the same allocation.
            unsafe { self.end.offset_from_unsigned(self.start) / self.stride }
        } else {
            cold_path();
            0
        }
    }
}

impl<'a, T> FusedIterator for StrideMut<'a, T> {}

pub(crate) struct StridesMut<'a, T> {
    start: NonNull<T>,
    end: NonNull<T>,
    remainder: &'a mut [T],
    stride: NonZero<usize>,
}

impl<'a, T> StridesMut<'a, T> {
    pub fn into_remainder(self) -> &'a mut [T] {
        self.remainder
    }

    pub fn from_slice(mut s: &'a mut [T], stride: usize) -> Self {
        let stride = NonZero::new(stride).expect("stride must be non-zero");
        let start = NonNull::from(&*s).to_raw_parts().0.cast();
        let n = s.len() / stride;
        if n > 0 {
            // SAFETY: Checked above that `n * stride <= s.len()`.
            let remainder = unsafe { s.split_off_mut(n.unchecked_mul(stride.into())..).unwrap_unchecked() };
            Self {
                start,
                // SAFETY: Checked above that `n * stride <= s.len()` and `1 <= n`.
                end: unsafe { start.add(stride.into()) },
                remainder,
                stride,
            }
        } else {
            Self {
                start,
                end: start,
                remainder: s,
                stride,
            }
        }
    }
}

impl<'a, T> Iterator for StridesMut<'a, T> {
    type Item = StrideMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let chunks = unsafe {
                NonNull::from(&*self.remainder)
                    .to_raw_parts()
                    .0
                    .cast()
                    // SAFETY: - By construction, `end` preceeds `&remainder`.
                    //         - By construction, both pointers are derived from the same allocation.
                    .offset_from_unsigned(self.end)
            } / self.stride;
            let ret = Some(StrideMut {
                start: self.start,
                end: unsafe { self.start.add(chunks) },
                stride: self.stride,
                phantom: PhantomData,
            });
            // SAFETY: Checked above that `start < end`.
            unsafe {
                self.start = self.start.add(1);
            }
            ret
        } else {
            cold_path();
            None
        }
    }
}

impl<'a, T> DoubleEndedIterator for StridesMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            // SAFETY: Checked above that `start < end`.
            unsafe {
                self.end = self.end.sub(1);
            }
            let chunks = unsafe {
                NonNull::from(&*self.remainder)
                    .to_raw_parts()
                    .0
                    .cast()
                    // SAFETY: - By construction, `end` preceeds `&remainder`.
                    //         - By construction, both pointers are derived from the same allocation.
                    .offset_from_unsigned(self.end)
            } / self.stride;
            Some(StrideMut {
                start: self.end,
                end: unsafe { self.end.add(chunks) },
                stride: self.stride,
                phantom: PhantomData,
            })
        } else {
            cold_path();
            None
        }
    }
}

impl<'a, T> ExactSizeIterator for StridesMut<'a, T> {
    fn len(&self) -> usize {
        // SAFETY: By construction, `start` cannot exceed `end`
        unsafe { self.end.offset_from_unsigned(self.start) }
    }
}

impl<'a, T> FusedIterator for StridesMut<'a, T> {}
