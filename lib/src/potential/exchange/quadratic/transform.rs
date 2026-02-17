use arc_rw_lock::{ElementRwLock, UniqueArcSliceRwLock};

use crate::marker::{InnerIsLeading, InnerIsTrailing};

/// A trait for normal mode transformations that transform
/// the positions of a group in all replicas into the first `n` normal modes
/// and all normal modes into the positions of the group in the first replica,
/// where `n` is the number of atoms in the group.
///
/// The transformation deals with the modes whose indices are `[0, n)`.
pub trait LeadingNormalModesTransform<T, V> {
    type Error;

    /// Transforms the positions of this group in all replicas
    /// into the normal modes delegated to the first replica.
    #[must_use]
    fn cartesian_to_normal_modes(
        &mut self,
        replicas_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Transforms all normal modes of this group into the positions
    /// of this group in the first replica.
    #[must_use]
    fn normal_modes_to_cartesian(
        &mut self,
        replicas_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to the first replica.
    fn eigenvalues(&self, eigenvalues: &mut [T]);
}

/// A trait for normal mode transformations that transform
/// the positions of a group in all replicas into `n` normal modes
/// and all normal modes into the positions of the group in a replica,
/// where `n` is the number of atoms in the group.
///
/// Given the index of the replica `replica`, the transformation
/// deals with the modes whose indices are `[replica * n, replica * (n + 1))`.
pub trait InnerNormalModesTransform<T, V> {
    type Error;

    /// Transforms the positions of this group in all replicas
    /// into the normal modes delegated to this replica.
    #[must_use]
    fn cartesian_to_normal_modes(
        &mut self,
        replicas_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Transforms all normal modes of this group into the positions
    /// of this group in this replica.
    #[must_use]
    fn normal_modes_to_cartesian(
        &mut self,
        replicas_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to this replica.
    fn eigenvalues(&self, eigenvalues: &mut [T]);
}

/// A trait for normal mode transformations that transform
/// the positions of a group in all replicas into the last `n` normal modes
/// and all normal modes into the positions of the group in the last replica,
/// where `n` is the number of atoms in the group.
///
/// Given the index of the last replica `last_replica`, the transformation
/// deals with the modes whose indices are `[last_replica * n, last_replica * (n + 1))`.
pub trait TrailingNormalModesTransform<T, V> {
    type Error;

    /// Transforms the positions of this group in all replicas
    /// into the normal modes delegated to the last replica.
    #[must_use]
    fn cartesian_to_normal_modes(
        &mut self,
        replicas_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Transforms all normal modes of this group into the positions
    /// of this group in the last replica.
    #[must_use]
    fn normal_modes_to_cartesian(
        &mut self,
        replicas_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to the last replica.
    fn eigenvalues(&self, eigenvalues: &mut [T]);
}

impl<T, V, U> LeadingNormalModesTransform<T, V> for U
where
    U: InnerNormalModesTransform<T, V> + InnerIsLeading,
{
    type Error = <Self as InnerNormalModesTransform<T, V>>::Error;

    fn cartesian_to_normal_modes(
        &mut self,
        replicas_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error> {
        InnerNormalModesTransform::cartesian_to_normal_modes(self, replicas_positions, normal_modes)
    }

    fn normal_modes_to_cartesian(
        &mut self,
        replicas_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error> {
        InnerNormalModesTransform::normal_modes_to_cartesian(self, replicas_normal_modes, group_position)
    }

    fn eigenvalues(&self, eigenvalues: &mut [T]) {
        InnerNormalModesTransform::eigenvalues(self, eigenvalues);
    }
}

impl<T, V, U> TrailingNormalModesTransform<T, V> for U
where
    U: InnerNormalModesTransform<T, V> + InnerIsTrailing,
{
    type Error = <Self as InnerNormalModesTransform<T, V>>::Error;

    fn cartesian_to_normal_modes(
        &mut self,
        replicas_positions: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error> {
        InnerNormalModesTransform::cartesian_to_normal_modes(self, replicas_positions, normal_modes)
    }

    fn normal_modes_to_cartesian(
        &mut self,
        replicas_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error> {
        InnerNormalModesTransform::normal_modes_to_cartesian(self, replicas_normal_modes, group_position)
    }

    fn eigenvalues(&self, eigenvalues: &mut [T]) {
        InnerNormalModesTransform::eigenvalues(self, eigenvalues);
    }
}
