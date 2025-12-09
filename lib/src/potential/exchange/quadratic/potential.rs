use crate::marker::{InnerIsLeading, InnerIsTrailing};

use super::{
    super::{InnerExchangePotential, LeadingExchangePotential, TrailingExchangePotential},
    transform::{
        InnerNormalModesTransform, LeadingNormalModesTransform, TrailingNormalModesTransform,
    },
};

/// A trait for leading exchange potentials that may be expanded to second order.
pub trait LeadingQuadraticExpansionExchangePotential<T, V>: LeadingExchangePotential<T, V> {
    /// The transformation that yields the normal modes such that
    /// the second order term is the sum over all modes squared times their
    /// respective eigenvalues.
    type QuadraticPotential<'a>: LeadingNormalModesTransform<T, V>
    where
        Self: 'a;

    /// The term left after the expansion to second order.
    /// Contains interactions of third order and higher.
    type ResiduePotential<'a>: LeadingExchangePotential<T, V>
    where
        Self: 'a;

    /// Treats `self` as a sum of harmonic oscillators - the normal modes -
    /// and a residual term of third order and beyond.
    fn as_quadratic_expansion(
        &mut self,
    ) -> (Self::QuadraticPotential<'_>, Self::ResiduePotential<'_>);
}

/// A trait for inner exchange potentials that may be expanded to second order.
pub trait InnerQuadraticExpansionExchangePotential<T, V>: InnerExchangePotential<T, V> {
    /// The transformation that yields the normal modes such that
    /// the second order term is the sum over all modes squared times their
    /// respective eigenvalues.
    type QuadraticPotential<'a>: for<'b> InnerNormalModesTransform<T, V>
    where
        Self: 'a;

    /// The term left after the expansion to second order.
    /// Contains interactions of third order and higher.
    type ResiduePotential<'a>: InnerExchangePotential<T, V>
    where
        Self: 'a;

    /// Treats `self` as a sum of harmonic oscillators - the normal modes -
    /// and a residual term of third order and beyond.
    fn as_quadratic_expansion(
        &mut self,
    ) -> (Self::QuadraticPotential<'_>, Self::ResiduePotential<'_>);
}

/// A trait for trailing exchange potentials that may be expanded to second order.
pub trait TrailingQuadraticExpansionExchangePotential<T, V>:
    TrailingExchangePotential<T, V>
{
    /// The transformation that yields the normal modes such that
    /// the second order term is the sum over all modes squared times their
    /// respective eigenvalues.
    type QuadraticPotential<'a>: for<'b> TrailingNormalModesTransform<T, V>
    where
        Self: 'a;

    /// The term left after the expansion to second order.
    /// Contains interactions of third order and higher.
    type ResiduePotential<'a>: TrailingExchangePotential<T, V>
    where
        Self: 'a;

    /// Treats `self` as a sum of harmonic oscillators - the normal modes -
    /// and a residual term of third order and beyond.
    fn as_quadratic_expansion(
        &mut self,
    ) -> (Self::QuadraticPotential<'_>, Self::ResiduePotential<'_>);
}

impl<T, V, U> LeadingQuadraticExpansionExchangePotential<T, V> for U
where
    U: InnerQuadraticExpansionExchangePotential<T, V> + InnerIsLeading,
    for<'a> U::QuadraticPotential<'a>:
        LeadingNormalModesTransform<T, V> + InnerNormalModesTransform<T, V>,
    for<'a> U::ResiduePotential<'a>: LeadingExchangePotential<T, V> + InnerExchangePotential<T, V>,
{
    type QuadraticPotential<'a>
        = <Self as InnerQuadraticExpansionExchangePotential<T, V>>::QuadraticPotential<'a>
    where
        Self: 'a;

    type ResiduePotential<'a>
        = <Self as InnerQuadraticExpansionExchangePotential<T, V>>::ResiduePotential<'a>
    where
        Self: 'a;

    fn as_quadratic_expansion(
        &mut self,
    ) -> (Self::QuadraticPotential<'_>, Self::ResiduePotential<'_>) {
        InnerQuadraticExpansionExchangePotential::as_quadratic_expansion(self)
    }
}

impl<T, V, U> TrailingQuadraticExpansionExchangePotential<T, V> for U
where
    U: InnerQuadraticExpansionExchangePotential<T, V> + InnerIsTrailing,
    for<'a> U::QuadraticPotential<'a>:
        TrailingNormalModesTransform<T, V> + InnerNormalModesTransform<T, V>,
    for<'a> U::ResiduePotential<'a>: TrailingExchangePotential<T, V> + InnerExchangePotential<T, V>,
{
    type QuadraticPotential<'a>
        = <Self as InnerQuadraticExpansionExchangePotential<T, V>>::QuadraticPotential<'a>
    where
        Self: 'a;

    type ResiduePotential<'a>
        = <Self as InnerQuadraticExpansionExchangePotential<T, V>>::ResiduePotential<'a>
    where
        Self: 'a;

    fn as_quadratic_expansion(
        &mut self,
    ) -> (Self::QuadraticPotential<'_>, Self::ResiduePotential<'_>) {
        InnerQuadraticExpansionExchangePotential::as_quadratic_expansion(self)
    }
}
