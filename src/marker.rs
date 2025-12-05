/// A marker trait for types that can implement `Leading[...]`
/// traits by reusing their `Inner[...]` implementation.
pub trait InnerIsLeading {}

/// A marker trait for types that can implement `Trailing[...]`
/// traits by reusing their `Inner[...]` implementation.
pub trait InnerIsTrailing {}
