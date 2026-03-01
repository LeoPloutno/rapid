use arc_rw_lock::{ElementRwLock, UniqueArcSliceRwLock};

use crate::{
    core::GroupImageHandle,
    marker::{InnerIsLeading, InnerIsTrailing},
};

/// A trait for normal mode transformations that transform
/// the positions of a group in all images into `m` normal modes
/// amongst the first `n` modes, where `n` is the number of atoms of this type
/// and `m` is the number of atoms in this group.
pub trait LeadingNormalModesTransform<T, V> {
    type Error;

    /// Transforms the positions of the type in all images
    /// into the normal modes delegated to the first image.
    fn cartesian_to_normal_modes(
        &mut self,
        images_type_positions: &ElementRwLock<GroupImageHandle<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Transforms all normal modes of the type into the positions
    /// of this group in the first image.
    fn normal_modes_to_cartesian(
        &mut self,
        images_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to the first image.
    fn eigenvalues(&self, eigenvalues: &mut [T]);
}

/// A trait for normal mode transformations that transform
/// the positions of a group in all images into `m` normal modes
/// amongst some `n` modes, where `n` is the number of atoms of this type
/// and `m` is the number of atoms in this group.
pub trait InnerNormalModesTransform<T, V> {
    type Error;

    /// Transforms the positions of this group in all images
    /// into the normal modes delegated to this image.
    fn cartesian_to_normal_modes(
        &mut self,
        images_type_positions: &ElementRwLock<GroupImageHandle<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Transforms all normal modes of this group into the positions
    /// of this group in this image.
    fn normal_modes_to_cartesian(
        &mut self,
        images_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to this image.
    fn eigenvalues(&self, eigenvalues: &mut [T]);
}

/// A trait for normal mode transformations that transform
/// the positions of a group in all images into `m` normal modes
/// amongst the last `n` modes, where `n` is the number of atoms of this type
/// and `m` is the number of atoms in this group.
pub trait TrailingNormalModesTransform<T, V> {
    type Error;

    /// Transforms the positions of this group in all images
    /// into the normal modes delegated to the last image.
    fn cartesian_to_normal_modes(
        &mut self,
        images_type_positions: &ElementRwLock<GroupImageHandle<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Transforms all normal modes of this group into the positions
    /// of this group in the last image.
    fn normal_modes_to_cartesian(
        &mut self,
        images_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error>;

    /// Sets `eigenvalues` to contain the eigenvalues of the normal modes
    /// of this group delegated to the last image.
    fn eigenvalues(&self, eigenvalues: &mut [T]);
}

impl<T, V, U> LeadingNormalModesTransform<T, V> for U
where
    U: InnerNormalModesTransform<T, V> + InnerIsLeading,
{
    type Error = <Self as InnerNormalModesTransform<T, V>>::Error;

    fn cartesian_to_normal_modes(
        &mut self,
        images_type_positions: &ElementRwLock<GroupImageHandle<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error> {
        InnerNormalModesTransform::cartesian_to_normal_modes(self, images_type_positions, normal_modes)
    }

    fn normal_modes_to_cartesian(
        &mut self,
        images_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error> {
        InnerNormalModesTransform::normal_modes_to_cartesian(self, images_normal_modes, group_position)
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
        images_type_positions: &ElementRwLock<GroupImageHandle<V>>,
        normal_modes: &mut [V],
    ) -> Result<(), Self::Error> {
        InnerNormalModesTransform::cartesian_to_normal_modes(self, images_type_positions, normal_modes)
    }

    fn normal_modes_to_cartesian(
        &mut self,
        images_normal_modes: &ElementRwLock<UniqueArcSliceRwLock<V>>,
        group_position: &mut [V],
    ) -> Result<(), Self::Error> {
        InnerNormalModesTransform::normal_modes_to_cartesian(self, images_normal_modes, group_position)
    }

    fn eigenvalues(&self, eigenvalues: &mut [T]) {
        InnerNormalModesTransform::eigenvalues(self, eigenvalues);
    }
}
