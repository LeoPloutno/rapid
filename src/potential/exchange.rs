use crate::{
    core::{
        adder::{SyncAddError, SyncAdderReciever, SyncAdderSender},
        marker::TrailingIsInner,
    },
    vector::Vector,
};
use std::ops::Add;

/// A trait for exchange potentials of the first replica.
pub trait LeadingExchangePotential<T, const N: usize, V, Adder>
where
    T: Add<Output = T>,
    V: Vector<N, Element = T>,
    Adder: ?Sized + SyncAdderReciever<T>,
{
    /// Calculates the exchange potential energy of a group
    /// and sets the forces of that group in the first replica accordingly.
    ///
    /// Returns the energy. If a synchronized summation failed, returns the error.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        adder: &mut Adder,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> Result<T, SyncAddError>;

    /// Calculates the exchange potential energy of a group.
    ///
    /// Returns the energy. If a synchronized summation failed, returns the error.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        adder: &mut Adder,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
    ) -> Result<T, SyncAddError>;

    /// Sets the forces of a group in the first replica.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        positions_prev_replica: &[V],
        position_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    );
}

/// A trait for exchange potentials of an inner replica.
pub trait InnerExchangePotential<T, const N: usize, V, Adder>
where
    T: Add<Output = T>,
    V: Vector<N, Element = T>,
    Adder: ?Sized + SyncAdderSender<T>,
{
    /// Calculates the exchange potential energy of a group
    /// and sets the forces of that group in the `replica_idx`-th replica accordingly.
    ///
    /// Returns the energy. If a synchronized summation failed, returns the error.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        adder: &mut Adder,
        positions_prev_replica: &[V],
        positions_next_replica: &[V],
        replica_idx: usize,
        positions: &[V],
        forces: &mut [V],
    ) -> Result<(), SyncAddError>;

    /// Calculates the exchange potential energy of a group.
    ///
    /// Returns the energy. If a synchronized summation failed, returns the error.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        adder: &mut Adder,
        positions_prev_replica: &[V],
        positions_next_replica: &[V],
        replica_idx: usize,
        positions: &[V],
    ) -> Result<(), SyncAddError>;

    /// Sets the forces of a group in the `replica_idx`-th replica.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        positions_prev_replica: &[V],
        positions_next_replica: &[V],
        replica_idx: usize,
        positions: &[V],
        forces: &mut [V],
    );
}

/// A trait for exchange potentials the last replica.
pub trait TrailingExchangePotential<T, const N: usize, V, Adder>
where
    T: Add<Output = T>,
    V: Vector<N, Element = T>,
    Adder: ?Sized + SyncAdderSender<T>,
{
    /// Calculates the exchange potential energy of a group
    /// and sets the forces of that group in the last replica accordingly.
    ///
    /// Returns the energy. If a synchronized summation failed, returns the error.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        adder: &mut Adder,
        positions_prev_replica: &[V],
        positions_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    ) -> Result<(), SyncAddError>;

    /// Calculates the exchange potential energy of a group.
    ///
    /// Returns the energy. If a synchronized summation failed, returns the error.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        adder: &mut Adder,
        positions_prev_replica: &[V],
        positions_next_replica: &[V],
        positions: &[V],
    ) -> Result<(), SyncAddError>;

    /// Sets the forces of a group in the last replica.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        positions_prev_replica: &[V],
        positions_next_replica: &[V],
        positions: &[V],
        forces: &mut [V],
    );
}

pub mod normal_modes {
    use std::ops::Add;

    use crate::{
        core::{adder::SyncAdderReciever, marker::LeadingIsInner},
        vector::Vector,
    };

    use super::LeadingExchangePotential;

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
    pub trait InnerNormalModesTransform<'a, T, const N: usize, V>
    where
        T: 'a,
        V: 'a + Vector<N, Element = T>,
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

    impl<'a, T, const N: usize, V, U> LeadingNormalModesTransform<'a, T, N, V> for U
    where
        T: 'a,
        V: Vector<N, Element = T> + 'a,
        U: LeadingNormalModesTransform<'a, T, N, V> + LeadingIsInner,
    {
        fn cartesian_to_normal_modes<CartIter, NmIter>(
            &mut self,
            n_replicas: usize,
            replica_size: usize,
            cartesian_replicas: CartIter,
            normal_modes: NmIter,
        ) where
            CartIter: Iterator<Item = &'a [V]> + Clone,
            NmIter: Iterator<Item = (&'a mut V, &'a mut T)>,
        {
        }

        fn normal_modes_to_cartesian<NmIter, CartIter>(
            &mut self,
            n_replicas: usize,
            replica_size: usize,
            normal_modes: NmIter,
            cartesian_replica: CartIter,
        ) where
            NmIter: Iterator<Item = &'a V> + Clone,
            CartIter: Iterator<Item = &'a mut V>,
        {
        }
    }

    pub trait LeadingExchangePotentialResidue<T, const N: usize, V, Adder>:
        LeadingExchangePotential<T, N, V, Adder>
    where
        T: Add<Output = T>,
        V: Vector<N, Element = T>,
        Adder: ?Sized + SyncAdderReciever<T>,
    {
        type Transform: for<'a> LeadingNormalModesTransform<'a, T, N, V>;

        fn second_order_potential(&mut self) -> Self::Transform;
    }
}
