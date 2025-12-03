use crate::vector::Vector;

/// A trait for transformations between cartesian coordinates of the first replica
/// and normal modes.
pub trait LeadingNormalModesTransform<'a, T, const N: usize, V>
where
    T: 'a,
    V: 'a + Vector<N, Element = T>,
{
    /// Transforms the positions of all replicas into the normal modes whose
    /// indices are multiples of `n_replicas`.
    ///
    /// Interprets the items of `noral_modes` as `(mode, eigenvalue)` and sets
    /// them accordingly.
    fn cartesian_to_normal_modes<CartIter, NmIter>(
        &mut self,
        n_replicas: usize,
        replica_size: usize,
        cartesian_replicas: CartIter,
        normal_modes: NmIter,
    ) where
        CartIter: Iterator<Item = &'a [V]> + Clone,
        NmIter: Iterator<Item = (&'a mut V, &'a mut T)>;

    /// Transforms all normal modes into the positions of the first replica.
    fn normal_modes_to_cartesian<NmIter, CartIter>(
        &mut self,
        n_replicas: usize,
        replica_size: usize,
        normal_modes: NmIter,
        cartesian_replica: CartIter,
    ) where
        NmIter: Iterator<Item = &'a V> + Clone,
        CartIter: Iterator<Item = &'a mut V>;
}

/// A trait for transformations between cartesian coordinates of an inner replica
/// and normal modes.
pub trait InnerNormalModesTransform<'a, T, V>
where
    T: 'a,
    V: 'a,
{
    /// Transforms the positions of all replicas into the normal modes whose
    /// indices are of the form `mode_idx + n * n_replicas`, where `n` is a whole number
    /// ranging from zero to `replica_size - 1`.
    ///
    /// Interprets the items of `noral_modes` as `(mode, eigenvalue)` and sets
    /// them accordingly.
    fn cartesian_to_normal_modes<CartIter, NmIter>(
        &mut self,
        n_replicas: usize,
        replica_size: usize,
        mode_idx: usize,
        cartesian_replicas: CartIter,
        normal_modes: NmIter,
    ) where
        CartIter: Iterator<Item = &'a [V]> + Clone,
        NmIter: Iterator<Item = (&'a mut V, &'a mut T)>;

    /// Transforms all normal modes into the positions of the `replica_idx`-th replica.
    fn normal_modes_to_cartesian<NmIter, CartIter>(
        &mut self,
        n_replicas: usize,
        replica_size: usize,
        replica_idx: usize,
        normal_modes: NmIter,
        cartesian_replica: CartIter,
    ) where
        NmIter: Iterator<Item = &'a V> + Clone,
        CartIter: Iterator<Item = &'a mut V>;
}

/// A trait for transformations between cartesian coordinates of the last replica
/// and normal modes.
pub trait TrailingNormalModesTransform<'a, T, const N: usize, V>
where
    T: 'a,
    V: 'a + Vector<N, Element = T>,
{
    /// Transforms the positions of all replicas into the normal modes whose
    /// indices are of the form `replica_size - 1 + n * n_replicas`, where `n` is a whole number
    /// ranging from zero to `replica_size - 1`.
    ///
    /// Interprets the items of `noral_modes` as `(mode, eigenvalue)` and sets
    /// them accordingly.
    fn cartesian_to_normal_modes<CartIter, NmIter>(
        &mut self,
        n_replicas: usize,
        replica_size: usize,
        cartesian_replicas: CartIter,
        normal_modes: NmIter,
    ) where
        CartIter: Iterator<Item = &'a [V]> + Clone,
        NmIter: Iterator<Item = (&'a mut V, &'a mut T)>;

    /// Transforms all normal modes into the positions of the last replica.
    fn normal_modes_to_cartesian<NmIter, CartIter>(
        &mut self,
        n_replicas: usize,
        replica_size: usize,
        normal_modes: NmIter,
        cartesian_replica: CartIter,
    ) where
        NmIter: Iterator<Item = &'a V> + Clone,
        CartIter: Iterator<Item = &'a mut V>;
}
