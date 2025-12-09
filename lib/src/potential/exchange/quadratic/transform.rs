use crate::core::AtomGroupInfo;
use crate::marker::{InnerIsLeading, InnerIsTrailing};

/// A trait for normal mode transformations that transform
/// the positions of a group in all replicas into the first `n` normal modes
/// and all normal modes into the positions of the group in the first replica,
/// where `n` is the number of atoms in the group.
///
/// The transformation deals with the modes whose indices are `[0, n)`.
pub trait LeadingNormalModesTransform<T, V> {
    /// Transforms the positions of this group in all replicas
    /// into the normal modes delegated to the first replica.
    fn cartesian_to_normal_modes<'a, I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        replicas_group_positions: I,
        replica_normal_modes: &mut [V],
    ) where
        V: 'a,
        I: Iterator<Item = &'a [V]> + Clone;

    /// Transforms all normal modes of this group into the positions
    /// of this group in the first replica.
    fn normal_modes_to_cartesian<'a, I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        replicas_normal_modes: I,
        replica_group_position: &mut [V],
    ) where
        V: 'a,
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
pub trait InnerNormalModesTransform<T, V> {
    /// Transforms the positions of this group in all replicas
    /// into the normal modes delegated to this replica.
    fn cartesian_to_normal_modes<'a, I>(
        &mut self,
        replica: usize,
        group: &AtomGroupInfo<T>,
        replicas_group_positions: I,
        replica_normal_modes: &mut [V],
    ) where
        V: 'a,
        I: Iterator<Item = &'a [V]> + Clone;

    /// Transforms all normal modes of this group into the positions
    /// of this group in this replica.
    fn normal_modes_to_cartesian<'a, I>(
        &mut self,
        replica: usize,
        group: &AtomGroupInfo<T>,
        replicas_normal_modes: I,
        replica_group_position: &mut [V],
    ) where
        V: 'a,
        I: Iterator<Item = &'a [V]> + Clone;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to this replica.
    fn eigenvalues(&mut self, replica: usize, group: &AtomGroupInfo<T>, eigenvalues: &mut [T]);
}

/// A trait for normal mode transformations that transform
/// the positions of a group in all replicas into the last `n` normal modes
/// and all normal modes into the positions of the group in the last replica,
/// where `n` is the number of atoms in the group.
///
/// Given the index of the last replica `last_replica`, the transformation
/// deals with the modes whose indices are `[last_replica * n, last_replica * (n + 1))`.
pub trait TrailingNormalModesTransform<T, V> {
    /// Transforms the positions of this group in all replicas
    /// into the normal modes delegated to the last replica.
    fn cartesian_to_normal_modes<'a, I>(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        replicas_group_positions: I,
        replica_normal_modes: &mut [V],
    ) where
        V: 'a,
        I: Iterator<Item = &'a [V]> + Clone;

    /// Transforms all normal modes of this group into the positions
    /// of this group in the last replica.
    fn normal_modes_to_cartesian<'a, I>(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        replica_normal_modes: I,
        replica_group_position: &mut [V],
    ) where
        V: 'a,
        I: Iterator<Item = &'a [V]> + Clone;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to the last replica.
    fn eigenvalues(&mut self, last_replica: usize, group: &AtomGroupInfo<T>, eigenvalues: &mut [T]);
}

impl<T, V, U> LeadingNormalModesTransform<T, V> for U
where
    U: InnerNormalModesTransform<T, V> + InnerIsLeading,
{
    fn cartesian_to_normal_modes<'a, I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        replicas_group_positions: I,
        replica_normal_modes: &mut [V],
    ) where
        V: 'a,
        I: Iterator<Item = &'a [V]> + Clone,
    {
        InnerNormalModesTransform::cartesian_to_normal_modes(
            self,
            0,
            group,
            replicas_group_positions,
            replica_normal_modes,
        );
    }

    fn normal_modes_to_cartesian<'a, I>(
        &mut self,
        group: &AtomGroupInfo<T>,
        replicas_normal_modes: I,
        replica_group_position: &mut [V],
    ) where
        V: 'a,
        I: Iterator<Item = &'a [V]> + Clone,
    {
        InnerNormalModesTransform::normal_modes_to_cartesian(
            self,
            0,
            group,
            replicas_normal_modes,
            replica_group_position,
        );
    }

    fn eigenvalues(&mut self, group: &AtomGroupInfo<T>, eigenvalues: &mut [T]) {
        InnerNormalModesTransform::eigenvalues(self, 0, group, eigenvalues);
    }
}

impl<T, V, U> TrailingNormalModesTransform<T, V> for U
where
    U: InnerNormalModesTransform<T, V> + InnerIsTrailing,
{
    fn cartesian_to_normal_modes<'a, I>(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        replicas_group_positions: I,
        replica_normal_modes: &mut [V],
    ) where
        V: 'a,
        I: Iterator<Item = &'a [V]> + Clone,
    {
        InnerNormalModesTransform::cartesian_to_normal_modes(
            self,
            last_replica,
            group,
            replicas_group_positions,
            replica_normal_modes,
        );
    }

    fn normal_modes_to_cartesian<'a, I>(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        replicas_normal_modes: I,
        replica_group_position: &mut [V],
    ) where
        V: 'a,
        I: Iterator<Item = &'a [V]> + Clone,
    {
        InnerNormalModesTransform::normal_modes_to_cartesian(
            self,
            last_replica,
            group,
            replicas_normal_modes,
            replica_group_position,
        );
    }

    fn eigenvalues(
        &mut self,
        last_replica: usize,
        group: &AtomGroupInfo<T>,
        eigenvalues: &mut [T],
    ) {
        InnerNormalModesTransform::eigenvalues(self, last_replica, group, eigenvalues);
    }
}
