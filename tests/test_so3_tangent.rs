use approx::abs_diff_eq;
use cardo::prelude::*;
use proptest::prelude::*;
use std::f64::consts::PI;

struct A;
struct B;

const EPS: f64 = 1e-10;

fn tangent_approx_eq(v1: &SO3Tangent<A, B, A>, v2: &SO3Tangent<A, B, A>) -> bool {
    abs_diff_eq!(v1.x(), v2.x(), epsilon = EPS)
        && abs_diff_eq!(v1.y(), v2.y(), epsilon = EPS)
        && abs_diff_eq!(v1.z(), v2.z(), epsilon = EPS)
}

fn mat3_approx_eq(a: &Mat3, b: &Mat3, eps: f64) -> bool {
    (0..3).all(|i| (0..3).all(|j| abs_diff_eq!(a.data[i][j], b.data[i][j], epsilon = eps)))
}

// Strategy for generating small tangent vectors (tests small angle approximations)
fn arb_tangent_small() -> impl Strategy<Value = SO3Tangent<A, B, A>> {
    (-1e-10..1e-10f64, -1e-10..1e-10f64, -1e-10..1e-10f64)
        .prop_map(|(x, y, z)| SO3Tangent::new(x, y, z))
}

// Strategy for generating tangent vectors with magnitude bounded by π
fn arb_tangent() -> impl Strategy<Value = SO3Tangent<A, B, A>> {
    (arb_unit_axis(), -PI..PI).prop_map(|(axis, angle)| {
        SO3Tangent::new(axis.x() * angle, axis.y() * angle, axis.z() * angle)
    })
}

// Strategy for generating unit axis vectors
fn arb_unit_axis() -> impl Strategy<Value = Vector3<A>> {
    (-1.0..1.0f64, -1.0..1.0f64, -1.0..1.0f64)
        .prop_filter_map("non-zero axis", |(x, y, z)| {
            let v: Vector3<A> = Vector3::new(x, y, z);
            if v.norm_squared() > 0.01 {
                Some(v.normalized())
            } else {
                None
            }
        })
}

proptest! {
    #[test]
    fn log_exp_roundtrip(v in arb_tangent()) {
        // log(exp(v)) ≈ v
        let r = v.exp();
        let v2 = r.log();
        prop_assert!(tangent_approx_eq(&v, &v2));
    }

    #[test]
    fn exp_small_angle_produces_unit_quaternion(v in arb_tangent_small()) {
        let r = v.exp();
        prop_assert!(abs_diff_eq!(r.quat.norm_squared(), 1.0, epsilon = EPS));
    }

    #[test]
    fn exp_large_angle_produces_unit_quaternion(v in arb_tangent()) {
        let r = v.exp();
        prop_assert!(abs_diff_eq!(r.quat.norm_squared(), 1.0, epsilon = EPS));
    }

    #[test]
    fn zero_exp_is_identity(_dummy in 0..1i32) {
        let v: SO3Tangent<A, B, A> = SO3Tangent::new(0.0, 0.0, 0.0);
        let r = v.exp();
        let id: SO3<A, B> = SO3::identity();
        prop_assert!(abs_diff_eq!(r.quat.norm_squared(), 1.0, epsilon = EPS));
        let delta = r.between(id).log();
        prop_assert!(abs_diff_eq!(delta.norm(), 0.0, epsilon = EPS));
    }

    // ljac

    #[test]
    fn ljac_at_zero_is_identity(_dummy in 0..1i32) {
        let jl = SO3Tangent::<A, B, A>::zero().ljac();
        prop_assert!(mat3_approx_eq(&jl, &Mat3::identity(), EPS));
    }

    #[test]
    fn ljac_preserves_own_tangent(tau in arb_tangent()) {
        // Jl(τ) * τ.data == τ.data because [τ]× τ = 0
        let jl = tau.ljac();
        let result = jl * tau.data;
        prop_assert!(abs_diff_eq!(result[0], tau.data[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[1], tau.data[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[2], tau.data[2], epsilon = EPS));
    }

    #[test]
    fn ljac_numerical_left_derivative(tau in arb_tangent()) {
        // Each column of Jl(τ) ≈ Log(Exp(τ + h·eᵢ) · Exp(τ)⁻¹).data / h
        // O(h) truncation error from finite differences limits accuracy to ~1e-7
        let eps = 1e-5;
        let jl = tau.ljac();
        let h = 1e-7;
        let r_tau: SO3<A, B> = tau.exp();
        let r_tau_inv: SO3<B, A> = r_tau.inverse();

        for i in 0..3 {
            let mut perturbed = tau.data;
            perturbed[i] += h;
            let tau_plus: SO3Tangent<A, B, A> = SO3Tangent::from_data(perturbed);
            let r_plus: SO3<A, B> = tau_plus.exp();
            let delta: SO3<B, B> = r_plus.compose(r_tau_inv);
            let log_delta: SO3Tangent<B, B, B> = delta.log();
            for row in 0..3 {
                let numerical = log_delta.data[row] / h;
                prop_assert!(
                    abs_diff_eq!(jl.data[row][i], numerical, epsilon = eps),
                    "mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jl.data[row][i], numerical
                );
            }
        }
    }

    #[test]
    fn ljac_small_angle_near_identity(tau in arb_tangent_small()) {
        // For small τ, Jl(τ) ≈ I + ½[τ]×
        // First-order approximation; higher-order terms are small but nonzero
        let eps = 1e-5;
        let jl = tau.ljac();
        let expected = Mat3::identity() + tau.hat() * 0.5;
        prop_assert!(mat3_approx_eq(&jl, &expected, eps));
    }

    // rjac

    #[test]
    fn rjac_at_zero_is_identity(_dummy in 0..1i32) {
        let jr = SO3Tangent::<A, B, A>::zero().rjac();
        prop_assert!(mat3_approx_eq(&jr, &Mat3::identity(), EPS));
    }

    #[test]
    fn rjac_preserves_own_tangent(tau in arb_tangent()) {
        // Jr(τ) * τ.data == τ.data because [τ]× τ = 0
        let jr = tau.rjac();
        let result = jr * tau.data;
        prop_assert!(abs_diff_eq!(result[0], tau.data[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[1], tau.data[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[2], tau.data[2], epsilon = EPS));
    }

    #[test]
    fn rjac_numerical_right_derivative(tau in arb_tangent()) {
        // Each column of Jr(τ) ≈ Log(Exp(τ)⁻¹ · Exp(τ + h·eᵢ)).data / h
        // O(h) truncation error from finite differences limits accuracy to ~1e-7
        let eps = 1e-5;
        let jr = tau.rjac();
        let h = 1e-7;
        let r_tau: SO3<A, B> = tau.exp();
        let r_tau_inv: SO3<B, A> = r_tau.inverse();

        for i in 0..3 {
            let mut perturbed = tau.data;
            perturbed[i] += h;
            let tau_plus: SO3Tangent<A, B, A> = SO3Tangent::from_data(perturbed);
            let r_plus: SO3<A, B> = tau_plus.exp();
            let delta: SO3<A, A> = r_tau_inv.compose(r_plus);
            let log_delta: SO3Tangent<A, A, A> = delta.log();
            for row in 0..3 {
                let numerical = log_delta.data[row] / h;
                prop_assert!(
                    abs_diff_eq!(jr.data[row][i], numerical, epsilon = eps),
                    "mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jr.data[row][i], numerical
                );
            }
        }
    }

    #[test]
    fn rjac_small_angle_near_identity(tau in arb_tangent_small()) {
        // For small τ, Jr(τ) ≈ I - ½[τ]×
        // First-order approximation; higher-order terms are small but nonzero
        let eps = 1e-5;
        let jr = tau.rjac();
        let expected = Mat3::identity() - tau.hat() * 0.5;
        prop_assert!(mat3_approx_eq(&jr, &expected, eps));
    }

    // ljacinv

    #[test]
    fn ljacinv_at_zero_is_identity(_dummy in 0..1i32) {
        let jlinv = SO3Tangent::<A, B, A>::zero().ljacinv();
        prop_assert!(mat3_approx_eq(&jlinv, &Mat3::identity(), EPS));
    }

    #[test]
    fn ljacinv_numerical_left_derivative(tau in arb_tangent()) {
        // Jl⁻¹ maps numerical Jl columns back to basis vectors
        // O(h) truncation error from finite differences limits accuracy to ~1e-7
        let eps = 1e-5;
        let jlinv = tau.ljacinv();
        let h = 1e-7;
        let r_tau: SO3<A, B> = tau.exp();
        let r_tau_inv: SO3<B, A> = r_tau.inverse();

        for i in 0..3 {
            let mut perturbed = tau.data;
            perturbed[i] += h;
            let tau_plus: SO3Tangent<A, B, A> = SO3Tangent::from_data(perturbed);
            let r_plus: SO3<A, B> = tau_plus.exp();
            let delta: SO3<B, B> = r_plus.compose(r_tau_inv);
            let log_delta: SO3Tangent<B, B, B> = delta.log();
            // Jl maps eᵢ to log_delta/h; so Jl⁻¹ maps log_delta/h back to eᵢ
            let recovered = jlinv * log_delta.data;
            for row in 0..3 {
                let expected = if row == i { h } else { 0.0 };
                prop_assert!(
                    abs_diff_eq!(recovered[row], expected, epsilon = eps),
                    "mismatch at ({}, {}): recovered={}, expected={}",
                    row, i, recovered[row], expected
                );
            }
        }
    }

    #[test]
    fn ljacinv_preserves_own_tangent(tau in arb_tangent()) {
        // Jl⁻¹(τ) * τ.data == τ.data because [τ]× τ = 0
        let jlinv = tau.ljacinv();
        let result = jlinv * tau.data;
        prop_assert!(abs_diff_eq!(result[0], tau.data[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[1], tau.data[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[2], tau.data[2], epsilon = EPS));
    }

    #[test]
    fn ljacinv_small_angle_near_identity(tau in arb_tangent_small()) {
        // For small τ, Jl⁻¹(τ) ≈ I - ½[τ]×
        // First-order approximation; higher-order terms are small but nonzero
        let eps = 1e-5;
        let jlinv = tau.ljacinv();
        let expected = Mat3::identity() - tau.hat() * 0.5;
        prop_assert!(mat3_approx_eq(&jlinv, &expected, eps));
    }

    #[test]
    fn ljacinv_is_inverse_of_ljac(tau in arb_tangent()) {
        // Jl⁻¹(τ) · Jl(τ) = I
        let product = tau.ljacinv() * tau.ljac();
        prop_assert!(mat3_approx_eq(&product, &Mat3::identity(), EPS));
    }

    // rjacinv

    #[test]
    fn rjacinv_at_zero_is_identity(_dummy in 0..1i32) {
        let jrinv = SO3Tangent::<A, B, A>::zero().rjacinv();
        prop_assert!(mat3_approx_eq(&jrinv, &Mat3::identity(), EPS));
    }

    #[test]
    fn rjacinv_preserves_own_tangent(tau in arb_tangent()) {
        // Jr⁻¹(τ) * τ.data == τ.data because [τ]× τ = 0
        let jrinv = tau.rjacinv();
        let result = jrinv * tau.data;
        prop_assert!(abs_diff_eq!(result[0], tau.data[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[1], tau.data[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[2], tau.data[2], epsilon = EPS));
    }

    #[test]
    fn rjacinv_numerical_right_derivative(tau in arb_tangent()) {
        // Jr⁻¹ maps numerical Jr columns back to basis vectors
        // O(h) truncation error from finite differences limits accuracy to ~1e-7
        let eps = 1e-5;
        let jrinv = tau.rjacinv();
        let h = 1e-7;
        let r_tau: SO3<A, B> = tau.exp();
        let r_tau_inv: SO3<B, A> = r_tau.inverse();

        for i in 0..3 {
            let mut perturbed = tau.data;
            perturbed[i] += h;
            let tau_plus: SO3Tangent<A, B, A> = SO3Tangent::from_data(perturbed);
            let r_plus: SO3<A, B> = tau_plus.exp();
            let delta: SO3<A, A> = r_tau_inv.compose(r_plus);
            let log_delta: SO3Tangent<A, A, A> = delta.log();
            let recovered = jrinv * log_delta.data;
            for row in 0..3 {
                let expected = if row == i { h } else { 0.0 };
                prop_assert!(
                    abs_diff_eq!(recovered[row], expected, epsilon = eps),
                    "mismatch at ({}, {}): recovered={}, expected={}",
                    row, i, recovered[row], expected
                );
            }
        }
    }

    #[test]
    fn rjacinv_small_angle_near_identity(tau in arb_tangent_small()) {
        // For small τ, Jr⁻¹(τ) ≈ I + ½[τ]×
        // First-order approximation; higher-order terms are small but nonzero
        let eps = 1e-5;
        let jrinv = tau.rjacinv();
        let expected = Mat3::identity() + tau.hat() * 0.5;
        prop_assert!(mat3_approx_eq(&jrinv, &expected, eps));
    }

    #[test]
    fn rjacinv_is_inverse_of_rjac(tau in arb_tangent()) {
        // Jr⁻¹(τ) · Jr(τ) = I
        let product = tau.rjacinv() * tau.rjac();
        prop_assert!(mat3_approx_eq(&product, &Mat3::identity(), EPS));
    }

    // ad

    // compose (BCH)

    #[test]
    fn compose_zero_lhs(tau in arb_tangent()) {
        // 0.bch_compose(τ) ≈ τ
        let zero: SO3Tangent<A, B, A> = SO3Tangent::zero();
        let result = zero.bch_compose(tau);
        prop_assert!(abs_diff_eq!(result.x(), tau.x(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.y(), tau.y(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.z(), tau.z(), epsilon = EPS));
    }

    #[test]
    fn compose_zero_rhs(tau in arb_tangent()) {
        // τ.bch_compose(0) ≈ τ
        let zero: SO3Tangent<A, B, A> = SO3Tangent::zero();
        let result = tau.bch_compose(zero);
        prop_assert!(abs_diff_eq!(result.x(), tau.x(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.y(), tau.y(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.z(), tau.z(), epsilon = EPS));
    }

    #[test]
    fn compose_approximates_exact(tau1 in arb_tangent(), tau2 in arb_tangent()) {
        // BCH should approximate log(exp(τ₁) · exp(τ₂))
        // Accuracy is O(|τ|³), so scale down for a tighter bound
        let s = 0.1;
        let t1: SO3Tangent<A, A, A> = SO3Tangent::from_data(std::array::from_fn(|i| tau1.data[i] * s));
        let t2: SO3Tangent<A, A, A> = SO3Tangent::from_data(std::array::from_fn(|i| tau2.data[i] * s));
        let bch = t1.bch_compose(t2);
        let exact = t1.exp().compose(t2.exp()).log();
        // O(|τ|³) ≈ 0.1³ = 1e-3, so eps = 1e-2 is conservative
        let eps = 1e-2;
        prop_assert!(abs_diff_eq!(bch.x(), exact.x(), epsilon = eps));
        prop_assert!(abs_diff_eq!(bch.y(), exact.y(), epsilon = eps));
        prop_assert!(abs_diff_eq!(bch.z(), exact.z(), epsilon = eps));
    }

    #[test]
    fn compose_with_negation_near_zero(tau in arb_tangent()) {
        // τ.bch_compose(-τ) ≈ 0 (for small τ, exact at τ = 0)
        let s = 0.1;
        let t: SO3Tangent<A, A, A> = SO3Tangent::from_data(std::array::from_fn(|i| tau.data[i] * s));
        let neg: SO3Tangent<A, A, A> = SO3Tangent::from_data(std::array::from_fn(|i| -t.data[i]));
        let result = t.bch_compose(neg);
        let eps = 1e-2;
        prop_assert!(abs_diff_eq!(result.norm(), 0.0, epsilon = eps));
    }

    // ad

    #[test]
    fn ad_equals_hat(tau in arb_tangent()) {
        // For SO3, ad(ω) = [ω]× = hat(ω)
        let ad = tau.ad();
        let hat = tau.hat();
        prop_assert!(mat3_approx_eq(&ad, &hat, EPS));
    }

    #[test]
    fn ad_zero_is_zero(_dummy in 0..1i32) {
        let zero: SO3Tangent<A, B, A> = SO3Tangent::zero();
        let ad = zero.ad();
        prop_assert!(mat3_approx_eq(&ad, &Mat3::zeros(), EPS));
    }

    #[test]
    fn ad_self_bracket_is_zero(tau in arb_tangent()) {
        // ad(τ) · τ = [τ, τ] = 0 (cross product of a vector with itself)
        let result = tau.ad() * tau.data;
        prop_assert!(abs_diff_eq!(result[0], 0.0, epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[1], 0.0, epsilon = EPS));
        prop_assert!(abs_diff_eq!(result[2], 0.0, epsilon = EPS));
    }

    #[test]
    fn ad_antisymmetric(tau1 in arb_tangent(), tau2 in arb_tangent()) {
        // ad(τ₁) · τ₂ = -ad(τ₂) · τ₁
        let lhs = tau1.ad() * tau2.data;
        let rhs = tau2.ad() * tau1.data;
        prop_assert!(abs_diff_eq!(lhs[0], -rhs[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(lhs[1], -rhs[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(lhs[2], -rhs[2], epsilon = EPS));
    }

    #[test]
    fn ad_jacobi_identity(tau1 in arb_tangent(), tau2 in arb_tangent(), tau3 in arb_tangent()) {
        // ad(τ₁)·(ad(τ₂)·τ₃) + ad(τ₂)·(ad(τ₃)·τ₁) + ad(τ₃)·(ad(τ₁)·τ₂) = 0
        let a = tau1.ad() * (tau2.ad() * tau3.data);
        let b = tau2.ad() * (tau3.ad() * tau1.data);
        let c = tau3.ad() * (tau1.ad() * tau2.data);
        for i in 0..3 {
            prop_assert!(abs_diff_eq!(a[i] + b[i] + c[i], 0.0, epsilon = EPS));
        }
    }
}

#[test]
fn so3_tangent_debug() {
    let v: SO3Tangent<A, B, A> = SO3Tangent::new(1.0, 2.0, 3.0);
    let s = format!("{:?}", v);
    assert!(s.contains("SO3Tangent"));
}

#[test]
fn so3_tangent_clone() {
    let v: SO3Tangent<A, B, A> = SO3Tangent::new(1.0, 2.0, 3.0);
    let v2 = v.clone();
    assert!(abs_diff_eq!(v.x(), v2.x(), epsilon = EPS));
    assert!(abs_diff_eq!(v.y(), v2.y(), epsilon = EPS));
    assert!(abs_diff_eq!(v.z(), v2.z(), epsilon = EPS));
}
