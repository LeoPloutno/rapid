use crate::{
    core::{
        AtomGroupInfo,
        adder::{SyncAddError, SyncAdderReciever, SyncAdderSender},
    },
    vector::Vector,
};
use std::ops::Add;

/// A trait for physical potentials that yield the total potential energy
/// of a replica.
pub trait LeadingPhysicalPotential<
    'a,
    T,
    const N: usize,
    V,
    Adder = dyn SyncAdderReciever<T>,
    PosIter = dyn Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
> where
    T: 'a + Add<Output = T>,
    V: Vector<N, Element = T> + 'a,
    Adder: ?Sized + SyncAdderReciever<T>,
    PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
{
    /// Calculates the total physical potential energy of a replica
    /// and sets the forces of this group accordingly.
    ///
    /// Returns the energy. If a synchronized summation failed, returns the error.
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential_set_forces(
        &mut self,
        adder: &mut Adder,
        groups_positions: &mut PosIter,
        group: AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> Result<T, SyncAddError>;

    /// Calculates the total physical potential energy of a replica.
    ///
    /// Returns the energy. If a synchronized summation failed, returns the error.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
    fn calculate_potential(
        &mut self,
        adder: &mut Adder,
        groups_positions: &mut PosIter,
        group: AtomGroupInfo<T>,
        positions: &[V],
    ) -> Result<T, SyncAddError>;

    /// Sets the forces of this group.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        groups_positions: &mut PosIter,
        group: AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    );
}

/// A trait for physical potentials that assist a `LeadingPhysicalPotential`
/// yield the total potential energy of a replica.
pub trait TrailingPhysicalPotential<
    'a,
    T,
    const N: usize,
    V,
    Adder = dyn SyncAdderSender<T>,
    PosIter = dyn Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
> where
    T: 'a + Add<Output = T>,
    V: Vector<N, Element = T> + 'a,
    Adder: ?Sized + SyncAdderSender<T>,
    PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
{
    /// Calculates the total physical potential energy of a replica
    /// and sets the forces of this group accordingly.
    ///
    /// If a synchronized summation failed, returns the error.
    fn calculate_potential_set_forces(
        &mut self,
        adder: &mut Adder,
        groups_positions: &mut PosIter,
        group: AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    ) -> Result<(), SyncAddError>;

    /// Calculates the total physical potential energy of a replica.
    ///
    /// If a synchronized summation failed, returns the error.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn calculate_potential(
        &mut self,
        adder: &mut Adder,
        groups_positions: &mut PosIter,
        group: AtomGroupInfo<T>,
        positions: &[V],
    ) -> Result<(), SyncAddError>;

    /// Sets the forces of this group.
    #[deprecated = "Consider using `calculate_potential_set_forces` as a more efficient alternative"]
    fn set_forces(
        &mut self,
        groups_positions: &mut PosIter,
        group: AtomGroupInfo<T>,
        positions: &[V],
        forces: &mut [V],
    );
}

#[cfg(feature = "monte_carlo")]
mod monte_carlo {
    use super::{LeadingPhysicalPotential, TrailingPhysicalPotential};
    use crate::{
        core::{
            AtomGroupInfo, ChangedPosition, ContainerOption,
            adder::{SyncAddError, SyncAdderReciever, SyncAdderSender},
        },
        vector::Vector,
    };
    use std::ops::Add;

    /// A trait for leading physical potentials that may be used in a Monte-Carlo algorithm.
    pub trait MonteCarloLeadingPhysicalPotential<
        'a,
        T,
        const N: usize,
        V,
        Adder = dyn SyncAdderReciever<T>,
        PosIter = dyn Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
        ChngPosIter = dyn Iterator<Item = ChangedPosition<V>>,
    >: LeadingPhysicalPotential<'a, T, N, V, Adder, PosIter>
    where
        T: 'a + Add<Output = T>,
        V: Vector<N, Element = T> + 'a,
        Adder: ?Sized + SyncAdderReciever<T>,
        PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
        ChngPosIter: ?Sized + Iterator<Item = ContainerOption<ChangedPosition<V>>>,
    {
        /// Calculates the change in total physical potential energy of a replica after a change
        /// in the position of a single atom and sets the forces of this group accordingly.
        ///
        /// Returns the change in energy. If a synchronized summation failed, returns the error.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff_set_forces(
            &mut self,
            adder: &mut Adder,
            changed_positions: &mut ChngPosIter,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> Result<T, SyncAddError>;

        /// Calculates the change in total physical potential energy of a replica after a change
        /// in the position of a single atom.
        ///
        /// Returns the change in energy. If a synchronized summation failed, returns the error.
        #[deprecated = "Consider using `calculate_potential_diff_set_forces` as a more efficient alternative"]
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_diff(
            &mut self,
            adder: &mut Adder,
            changed_positions: &mut ChngPosIter,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
        ) -> Result<T, SyncAddError>;
    }

    /// A trait for trailing physical potentials that may be used in a Monte-Carlo algorithm.
    pub trait MonteCarloTrailingPhysicalPotential<
        'a,
        T,
        const N: usize,
        V,
        Adder = dyn SyncAdderSender<T>,
        PosIter = dyn Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
        ChngPosIter = dyn Iterator<Item = ChangedPosition<V>>,
    >: TrailingPhysicalPotential<'a, T, N, V, Adder, PosIter>
    where
        T: 'a + Add<Output = T>,
        V: Vector<N, Element = T> + 'a,
        Adder: ?Sized + SyncAdderSender<T>,
        PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
        ChngPosIter: ?Sized + Iterator<Item = ContainerOption<ChangedPosition<V>>>,
    {
        /// Calculates the change in total physical potential energy of a replica after a change
        /// in the position of a single atom and sets the forces of this group accordingly.
        ///
        /// If a synchronized summation failed, returns the error.
        fn calculate_potential_diff_set_forces(
            &mut self,
            adder: &mut Adder,
            changed_positions: &mut ChngPosIter,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> Result<(), SyncAddError>;

        /// Calculates the change in total physical potential energy of a replica after a change
        /// in the position of a single atom.
        ///
        /// If a synchronized summation failed, returns the error.
        #[deprecated = "Consider using `calculate_potential_diff_set_forces` as a more efficient alternative"]
        fn calculate_potential_diff(
            &mut self,
            adder: &mut Adder,
            changed_positions: &mut ChngPosIter,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
        ) -> Result<(), SyncAddError>;
    }
}

mod group_additive {
    use super::{LeadingPhysicalPotential, TrailingPhysicalPotential};
    use crate::{
        core::{
            AtomGroupInfo,
            adder::{SyncAddError, SyncAdderReciever, SyncAdderSender},
        },
        vector::Vector,
    };
    use std::ops::Add;

    /// A trait for physical potentials that can be expressed as a sum
    /// of contributions of all atom groups.
    pub trait GroupAdditivePhysicalPotential<
        'a,
        T,
        const N: usize,
        V,
        PosIter = dyn Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
    >
    where
        T: 'a,
        V: Vector<N, Element = T> + 'a,
        PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
    {
        /// Calculates the contribution of this group to the total physical potential energy
        /// of a replica and sets the forces of this group accordingly.
        ///
        /// Returns the contribution to the total energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_contribution_set_forces(
            &mut self,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T;

        /// Calculates the contribution of this group to the total physical potential energy
        /// of a replica.
        ///
        /// Returns the contribution to the total energy.
        #[deprecated = "Consider using `calculate_potential_contribution_set_forces` as a more efficient alternative"]
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_potential_contribution(
            &mut self,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
        ) -> T;

        /// Sets the forces of this group.
        #[deprecated = "Consider using `calculate_potential_contribution_set_forces` as a more efficient alternative"]
        fn set_forces(
            &mut self,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        );
    }

    impl<
        'a,
        T,
        const N: usize,
        V,
        Adder,
        PosIter,
        P: GroupAdditivePhysicalPotential<'a, T, N, V, PosIter>,
    > LeadingPhysicalPotential<'a, T, N, V, Adder, PosIter> for P
    where
        T: 'a + Add<Output = T>,
        V: Vector<N, Element = T> + 'a,
        Adder: ?Sized + SyncAdderReciever<T>,
        PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
    {
        fn calculate_potential_set_forces(
            &mut self,
            adder: &mut Adder,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> Result<T, SyncAddError> {
            let contribution = self.calculate_potential_contribution_set_forces(
                groups_positions,
                group,
                positions,
                forces,
            );
            Ok(adder.recieve()? + contribution)
        }

        fn calculate_potential(
            &mut self,
            adder: &mut Adder,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
        ) -> Result<T, SyncAddError> {
            #[allow(deprecated)]
            let contribution =
                self.calculate_potential_contribution(groups_positions, group, positions);
            Ok(adder.recieve()? + contribution)
        }

        fn set_forces(
            &mut self,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) {
            #[allow(deprecated)]
            GroupAdditivePhysicalPotential::set_forces(
                self,
                groups_positions,
                group,
                positions,
                forces,
            );
        }
    }

    impl<
        'a,
        T,
        const N: usize,
        V,
        Adder,
        PosIter,
        P: GroupAdditivePhysicalPotential<'a, T, N, V, PosIter>,
    > TrailingPhysicalPotential<'a, T, N, V, Adder, PosIter> for P
    where
        T: 'a + Add<Output = T>,
        V: Vector<N, Element = T> + 'a,
        Adder: ?Sized + SyncAdderSender<T>,
        PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
    {
        fn calculate_potential_set_forces(
            &mut self,
            adder: &mut Adder,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> Result<(), SyncAddError> {
            let contribution = self.calculate_potential_contribution_set_forces(
                groups_positions,
                group,
                positions,
                forces,
            );
            adder.send(contribution)?;
            Ok(())
        }

        fn calculate_potential(
            &mut self,
            adder: &mut Adder,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
        ) -> Result<(), SyncAddError> {
            #[allow(deprecated)]
            let contribution =
                self.calculate_potential_contribution(groups_positions, group, positions);
            adder.send(contribution)?;
            Ok(())
        }

        fn set_forces(
            &mut self,
            groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) {
            #[allow(deprecated)]
            GroupAdditivePhysicalPotential::set_forces(
                self,
                groups_positions,
                group,
                positions,
                forces,
            );
        }
    }

    #[cfg(feature = "monte_carlo")]
    pub(super) mod monte_carlo {
        use super::{
            super::{MonteCarloLeadingPhysicalPotential, MonteCarloTrailingPhysicalPotential},
            GroupAdditivePhysicalPotential,
        };
        use crate::{
            core::{
                AtomGroupInfo, ChangedPosition, ContainerOption,
                adder::{SyncAddError, SyncAdderReciever, SyncAdderSender},
            },
            vector::Vector,
        };
        use std::ops::Add;

        /// A trait for group-additive physical potentials that may be used in a Monte-Carlo algorithm.
        pub trait MonteCarloGroupAdditivePhysicalPotential<
            'a,
            T,
            const N: usize,
            V,
            PosIter = dyn Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
            ChngPosIter = dyn Iterator<Item = ChangedPosition<V>>,
        >: GroupAdditivePhysicalPotential<'a, T, N, V, PosIter>
        where
            T: 'a,
            V: Vector<N, Element = T> + 'a,
            PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
            ChngPosIter: ?Sized + Iterator<Item = ContainerOption<ChangedPosition<V>>>,
        {
            /// Calculates the contribution of this group to the change in total physical potential energy
            /// of a replica and sets the forces of this group accordingly.
            ///
            /// Returns the contribution to the change in total energy.
            #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
            fn calculate_potential_diff_contribution_set_forces(
                &mut self,
                changed_positions: &mut ChngPosIter,
                groups_positions: &mut PosIter,
                group: AtomGroupInfo<T>,
                positions: &[V],
                forces: &mut [V],
            ) -> T;

            /// Calculates the contribution of this group to the change in total physical potential energy
            /// of a replica.
            ///
            /// Returns the contribution to the change in total energy.
            #[deprecated = "Consider using `calculate_potential_diff_contribution_set_forces` as a more efficient alternative"]
            #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
            fn calculate_potential_diff_contribution(
                &mut self,
                changed_positions: &mut ChngPosIter,
                groups_positions: &mut PosIter,
                group: AtomGroupInfo<T>,
                positions: &[V],
            ) -> T;
        }

        impl<'a, T, const N: usize, V, Adder, PosIter, ChngPosIter, P>
            MonteCarloLeadingPhysicalPotential<'a, T, N, V, Adder, PosIter, ChngPosIter> for P
        where
            T: 'a + Add<Output = T>,
            V: Vector<N, Element = T> + 'a,
            Adder: ?Sized + SyncAdderReciever<T>,
            PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
            ChngPosIter: ?Sized + Iterator<Item = ContainerOption<ChangedPosition<V>>>,
            P: MonteCarloGroupAdditivePhysicalPotential<'a, T, N, V, PosIter, ChngPosIter>,
        {
            fn calculate_potential_diff_set_forces(
                &mut self,
                adder: &mut Adder,
                changed_positions: &mut ChngPosIter,
                groups_positions: &mut PosIter,
                group: AtomGroupInfo<T>,
                positions: &[V],
                forces: &mut [V],
            ) -> Result<T, SyncAddError> {
                let contribution = self.calculate_potential_diff_contribution_set_forces(
                    changed_positions,
                    groups_positions,
                    group,
                    positions,
                    forces,
                );
                Ok(adder.recieve()? + contribution)
            }

            fn calculate_potential_diff(
                &mut self,
                adder: &mut Adder,
                changed_positions: &mut ChngPosIter,
                groups_positions: &mut PosIter,
                group: AtomGroupInfo<T>,
                positions: &[V],
            ) -> Result<T, SyncAddError> {
                #[allow(deprecated)]
                let contribution = self.calculate_potential_diff_contribution(
                    changed_positions,
                    groups_positions,
                    group,
                    positions,
                );
                Ok(adder.recieve()? + contribution)
            }
        }

        impl<'a, T, const N: usize, V, Adder, PosIter, ChngPosIter, P>
            MonteCarloTrailingPhysicalPotential<'a, T, N, V, Adder, PosIter, ChngPosIter> for P
        where
            T: 'a + Add<Output = T>,
            V: Vector<N, Element = T> + 'a,
            Adder: ?Sized + SyncAdderSender<T>,
            PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
            ChngPosIter: ?Sized + Iterator<Item = ContainerOption<ChangedPosition<V>>>,
            P: MonteCarloGroupAdditivePhysicalPotential<'a, T, N, V, PosIter, ChngPosIter>,
        {
            fn calculate_potential_diff_set_forces(
                &mut self,
                adder: &mut Adder,
                changed_positions: &mut ChngPosIter,
                groups_positions: &mut PosIter,
                group: AtomGroupInfo<T>,
                positions: &[V],
                forces: &mut [V],
            ) -> Result<(), SyncAddError> {
                let contribution = self.calculate_potential_diff_contribution_set_forces(
                    changed_positions,
                    groups_positions,
                    group,
                    positions,
                    forces,
                );
                adder.send(contribution)?;
                Ok(())
            }

            fn calculate_potential_diff(
                &mut self,
                adder: &mut Adder,
                changed_positions: &mut ChngPosIter,
                groups_positions: &mut PosIter,
                group: AtomGroupInfo<T>,
                positions: &[V],
            ) -> Result<(), SyncAddError> {
                #[allow(deprecated)]
                let contribution = self.calculate_potential_diff_contribution(
                    changed_positions,
                    groups_positions,
                    group,
                    positions,
                );
                adder.send(contribution)?;
                Ok(())
            }
        }
    }
}

pub use group_additive::GroupAdditivePhysicalPotential;

mod group_uncoupled {
    use super::GroupAdditivePhysicalPotential;
    use crate::{core::AtomGroupInfo, vector::Vector};

    /// A trait for physical potentials that can be expressed as a sum
    /// of potentials, each depending on the parameters and positions of a single group.
    pub trait GroupUncoupledPhysicalPotential<T, const N: usize, V>
    where
        V: Vector<N, Element = T>,
    {
        /// Calculates the contribution of this group to the total physical potential energy
        /// of a replica and sets the forces of this group accordingly.
        ///
        /// Returns the contribution to the total energy.
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_group_potential_set_forces(
            &mut self,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T;

        /// Calculates the contribution of this group to the total physical potential energy
        /// of a replica.
        ///
        /// Returns the contribution to the total energy.
        #[deprecated = "Consider using `calculate_group_potential_set_forces` as a more efficient alternative"]
        #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
        fn calculate_group_potential(&mut self, group: AtomGroupInfo<T>, positions: &[V]) -> T;

        /// Sets the forces of this group.
        #[deprecated = "Consider using `calculate_potential_contribution_set_forces` as a more efficient alternative"]
        fn set_forces(&mut self, group: AtomGroupInfo<T>, positions: &[V], forces: &mut [V]);
    }

    impl<'a, T, const N: usize, V, PosIter, P> GroupAdditivePhysicalPotential<'a, T, N, V, PosIter>
        for P
    where
        T: 'a,
        V: Vector<N, Element = T> + 'a,
        PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
        P: GroupUncoupledPhysicalPotential<T, N, V>,
    {
        fn calculate_potential_contribution_set_forces(
            &mut self,
            _groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) -> T {
            self.calculate_group_potential_set_forces(group, positions, forces)
        }

        fn calculate_potential_contribution(
            &mut self,
            _groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
        ) -> T {
            #[allow(deprecated)]
            self.calculate_group_potential(group, positions)
        }

        fn set_forces(
            &mut self,
            _groups_positions: &mut PosIter,
            group: AtomGroupInfo<T>,
            positions: &[V],
            forces: &mut [V],
        ) {
            #[allow(deprecated)]
            GroupUncoupledPhysicalPotential::set_forces(self, group, positions, forces);
        }
    }

    #[cfg(feature = "monte_carlo")]
    pub(super) mod monte_carlo {
        use super::{
            super::MonteCarloGroupAdditivePhysicalPotential, GroupUncoupledPhysicalPotential,
        };
        use crate::{
            core::{AtomGroupInfo, ChangedPosition, ContainerOption},
            vector::Vector,
        };

        /// A trait for group-uncoupled potentials that may be used in a Monte-Carlo algorithm.
        pub trait MonteCarloGroupUncoupledPhysicalPotential<T, const N: usize, V>:
            GroupUncoupledPhysicalPotential<T, N, V>
        where
            V: Vector<N, Element = T>,
        {
            /// Calculates the contribution of this group to the change in total physical potential energy
            /// of a replica and sets the forces of this group accordingly.
            ///
            /// Returns the contribution to the change in total energy.
            #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
            fn calculate_group_potential_diff_set_forces(
                &mut self,
                changed_position: ChangedPosition<V>,
                group: AtomGroupInfo<T>,
                positions: &[V],
                forces: &mut [V],
            ) -> T;

            /// Calculates the contribution of this group to the change in total physical potential energy
            /// of a replica.
            ///
            /// Returns the contribution to the change in total energy.
            #[deprecated = "Consider using `calculate_group_potential_diff_set_forces` as a more efficient alternative"]
            #[must_use = "Discarding the result of a potentially heavy computation is wasteful"]
            fn calculate_group_potential_diff(
                &mut self,
                changed_position: ChangedPosition<V>,
                group: AtomGroupInfo<T>,
                positions: &[V],
            ) -> T;
        }

        impl<'a, T, const N: usize, V, PosIter, ChngPosIter, P>
            MonteCarloGroupAdditivePhysicalPotential<'a, T, N, V, PosIter, ChngPosIter> for P
        where
            T: 'a + Default,
            V: Vector<N, Element = T> + 'a,
            PosIter: ?Sized + Iterator<Item = (&'a AtomGroupInfo<T>, &'a [V])>,
            ChngPosIter: ?Sized + Iterator<Item = ContainerOption<ChangedPosition<V>>>,
            P: MonteCarloGroupUncoupledPhysicalPotential<T, N, V>,
        {
            fn calculate_potential_diff_contribution_set_forces(
                &mut self,
                changed_positions: &mut ChngPosIter,
                _groups_positions: &mut PosIter,
                group: AtomGroupInfo<T>,
                positions: &[V],
                forces: &mut [V],
            ) -> T {
                for changed_position in changed_positions {
                    if let ContainerOption::This(changed_position) = changed_position {
                        return self.calculate_group_potential_diff_set_forces(
                            changed_position,
                            group,
                            positions,
                            forces,
                        );
                    }
                }
                T::default()
            }

            fn calculate_potential_diff_contribution(
                &mut self,
                changed_positions: &mut ChngPosIter,
                _groups_positions: &mut PosIter,
                group: AtomGroupInfo<T>,
                positions: &[V],
            ) -> T {
                for changed_position in changed_positions {
                    if let ContainerOption::This(changed_position) = changed_position {
                        #[allow(deprecated)]
                        return self.calculate_group_potential_diff(
                            changed_position,
                            group,
                            positions,
                        );
                    }
                }
                T::default()
            }
        }
    }
}

pub use group_uncoupled::GroupUncoupledPhysicalPotential;

#[cfg(feature = "monte_carlo")]
pub use self::{
    group_additive::monte_carlo::MonteCarloGroupAdditivePhysicalPotential,
    group_uncoupled::monte_carlo::MonteCarloGroupUncoupledPhysicalPotential,
    monte_carlo::{MonteCarloLeadingPhysicalPotential, MonteCarloTrailingPhysicalPotential},
};
