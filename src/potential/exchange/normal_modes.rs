use crate::core::AtomGroupInfo;
use crate::marker::{InnerIsLeading, InnerIsTrailing};

/// A trait for normal mode transformations that transform
/// the positions of a group in all replicas into the first `n` normal modes
/// and all normal modes into the positions of the group in the first replica,
/// where `n` is the number of atoms in the group.
///
/// The transformation deals with the modes whose indices are `[0, n)`.
pub trait LeadingNormalModesTransform<'a, T, V>
where
    T: 'a,
    V: 'a,
{
    /// Transforms the positions of this group in all replicas
    /// into the normal modes delegated to the first replica.
    fn cartesian_to_normal_modes<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_replicas: I,
        normal_modes: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone;

    /// Transforms all normal modes of this group into the positions
    /// of this group in the first replica.
    fn normal_modes_to_cartesian<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        normal_modes_replicas: I,
        position: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to the first replica.
    fn eigenvalues(&mut self, group: &AtomGroupInfo<T>, eigenvalues: &mut [T]);
}

/// A trait for normal mode transformations that transform
/// the positions of a group in all replicas into `n` normal modes
/// and all normal modes into the positions of the group in a replica,
/// where `n` is the number of atoms in the group.
///
/// Given the index of the replica `replica`, the transformation
/// deals with the modes whose indices are `[replica * n, replica * (n + 1))`.
pub trait InnerNormalModesTransform<'a, T, V>
where
    T: 'a,
    V: 'a,
{
    /// Transforms the positions of this group in all replicas
    /// into the normal modes delegated to this replica.
    fn cartesian_to_normal_modes<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        replica: usize,
        positions_replicas: I,
        normal_modes: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone;

    /// Transforms all normal modes of this group into the positions
    /// of this group in this replica.
    fn normal_modes_to_cartesian<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        replica: usize,
        normal_modes_replicas: I,
        position: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to this replica.
    fn eigenvalues(&mut self, group: &AtomGroupInfo<T>, replica: usize, eigenvalues: &mut [T]);
}

/// A trait for normal mode transformations that transform
/// the positions of a group in all replicas into the last `n` normal modes
/// and all normal modes into the positions of the group in the last replica,
/// where `n` is the number of atoms in the group.
///
/// Given the index of the last replica `last_replica`, the transformation
/// deals with the modes whose indices are `[last_replica * n, last_replica * (n + 1))`.
pub trait TrailingNormalModesTransform<'a, T, V>
where
    T: 'a,
    V: 'a,
{
    /// Transforms the positions of this group in all replicas
    /// into the normal modes delegated to the last replica.
    fn cartesian_to_normal_modes<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_replicas: I,
        normal_modes: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone;

    /// Transforms all normal modes of this group into the positions
    /// of this group in the last replica.
    fn normal_modes_to_cartesian<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        normal_modes_replicas: I,
        position: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to the last replica.
    fn eigenvalues(&mut self, group: &AtomGroupInfo<T>, last_replica: usize, eigenvalues: &mut [T]);
}

impl<'a, T, V, U> LeadingNormalModesTransform<'a, T, V> for U
where
    T: 'a,
    V: 'a,
    U: InnerNormalModesTransform<'a, T, V> + InnerIsLeading,
{
    fn cartesian_to_normal_modes<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        positions_replicas: I,
        normal_modes: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone,
    {
        InnerNormalModesTransform::cartesian_to_normal_modes(
            self,
            group,
            0,
            positions_replicas,
            normal_modes,
        );
    }

    fn normal_modes_to_cartesian<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        normal_modes_replicas: I,
        position: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone,
    {
        InnerNormalModesTransform::normal_modes_to_cartesian(
            self,
            group,
            0,
            normal_modes_replicas,
            position,
        );
    }

    fn eigenvalues(&mut self, group: &AtomGroupInfo<T>, eigenvalues: &mut [T]) {
        InnerNormalModesTransform::eigenvalues(self, group, 0, eigenvalues);
    }
}

impl<'a, T, V, U> TrailingNormalModesTransform<'a, T, V> for U
where
    T: 'a,
    V: 'a,
    U: InnerNormalModesTransform<'a, T, V> + InnerIsTrailing,
{
    fn cartesian_to_normal_modes<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        positions_replicas: I,
        normal_modes: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone,
    {
        InnerNormalModesTransform::cartesian_to_normal_modes(
            self,
            group,
            last_replica,
            positions_replicas,
            normal_modes,
        );
    }

    fn normal_modes_to_cartesian<I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        normal_modes_replicas: I,
        position: &mut [V],
    ) where
        I: Iterator<Item = &'a [V]> + Clone,
    {
        InnerNormalModesTransform::normal_modes_to_cartesian(
            self,
            group,
            last_replica,
            normal_modes_replicas,
            position,
        );
    }

    fn eigenvalues(
        &mut self,
        group: &AtomGroupInfo<T>,
        last_replica: usize,
        eigenvalues: &mut [T],
    ) {
        InnerNormalModesTransform::eigenvalues(self, group, last_replica, eigenvalues);
    }
}
