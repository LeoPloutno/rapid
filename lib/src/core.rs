use std::range::Range;

/// Information about atoms of the same type.
#[derive(Clone, Debug)]
pub struct AtomGroupInfo<T> {
    /// Unique identifier.
    pub id: usize,
    /// The range of indices corresponding to this group.
    pub span: Range<usize>,
    /// The mass of a single atom of this group.
    pub mass: T,
    /// Atomic symbol
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct CommError<T> {
    pub replica: usize,
    pub group: AtomGroupInfo<T>,
}

pub trait Factory {
    type Leading;
    type Inner: Iterator;
    type Trailing;

    fn produce(&mut self) -> (Self::Leading, Self::Inner, Self::Trailing);
}

// pub trait FullFactory {
//     type Main;
//     type Leading;
//     type Inner: Iterator;
//     type Trailing;

//     fn produce(&mut self) -> (Self::Main, Self::Leading, Self::Inner, Self::Trailing);
// }

pub type DynFactory<L, I, T> = Box<dyn Factory<Leading = L, Inner = Box<dyn Iterator<Item = I>>, Trailing = T>>;

// pub type DynFullFactory<M, L, I, T> =
//     Box<dyn FullFactory<Main = M, Leading = L, Inner = Box<dyn Iterator<Item = I>>, Trailing = T>>;

pub trait GroupRecord<T> {
    fn group(&self) -> AtomGroupInfo<T>;

    fn group_idx(&self) -> usize;
}

pub trait ReplicasFactory<T> {
    type Leading: Iterator;
    type Inner: Iterator<Item: Iterator>;
    type Trailing: Iterator;

    fn produce(&mut self, inner_replicas: usize, groups: T) -> (Self::Leading, Self::Inner, Self::Trailing);
}

pub trait FullFactory<T> {
    type Main;
    type Leading: Iterator;
    type Inner: Iterator<Item: Iterator>;
    type Trailing: Iterator;

    fn produce(&mut self, inner_replicas: usize, groups: T)
    -> (Self::Main, Self::Leading, Self::Inner, Self::Trailing);
}

pub type DynReplicasFactory<T, Leading, Inner, Trailing> = dyn ReplicasFactory<
        T,
        Leading = Box<dyn Iterator<Item = Leading>>,
        Inner = Box<dyn Iterator<Item = Box<dyn Iterator<Item = Inner>>>>,
        Trailing = Box<dyn Iterator<Item = Trailing>>,
    >;

pub type DynFullFactory<T, Main, Leading, Inner, Trailing> = dyn FullFactory<
        T,
        Main = Main,
        Leading = Box<dyn Iterator<Item = Leading>>,
        Inner = Box<dyn Iterator<Item = Box<dyn Iterator<Item = Inner>>>>,
        Trailing = Box<dyn Iterator<Item = Trailing>>,
    >;
