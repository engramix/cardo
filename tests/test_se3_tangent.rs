use approx::abs_diff_eq;
use cardo::prelude::*;
use proptest::prelude::*;
use std::f64::consts::PI;

struct A;
struct B;

const EPS: f64 = 1e-10;

fn tangent_approx_eq(a: &SE3Tangent<A, B, A>, b: &SE3Tangent<A, B, A>) -> bool {
    (0..6).all(|i| abs_diff_eq!(a.data[i], b.data[i], epsilon = EPS))
}

fn mat6_approx_eq(a: &Mat6, b: &Mat6, eps: f64) -> bool {
    (0..6).all(|i| (0..6).all(|j| abs_diff_eq!(a.data[i][j], b.data[i][j], epsilon = eps)))
}

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

fn arb_se3_tangent() -> impl Strategy<Value = SE3Tangent<A, B, A>> {
    (arb_unit_axis(), -PI..PI, -10.0..10.0f64, -10.0..10.0f64, -10.0..10.0f64)
        .prop_map(|(axis, angle, vx, vy, vz)| {
            SE3Tangent::from_lin_ang(
                [vx, vy, vz],
                [axis.x() * angle, axis.y() * angle, axis.z() * angle],
            )
        })
}

fn arb_se3_tangent_small() -> impl Strategy<Value = SE3Tangent<A, B, A>> {
    (
        -1e-10..1e-10f64, -1e-10..1e-10f64, -1e-10..1e-10f64,
        -1e-10..1e-10f64, -1e-10..1e-10f64, -1e-10..1e-10f64,
    )
        .prop_map(|(vx, vy, vz, wx, wy, wz)| {
            SE3Tangent::from_data([vx, vy, vz, wx, wy, wz])
        })
}

#[test]
fn tangent_debug() {
    let v: SE3Tangent<A, B, A> = SE3Tangent::from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let s = format!("{:?}", v);
    assert!(s.contains("SE3Tangent"));
}

#[test]
fn tangent_clone() {
    let v: SE3Tangent<A, B, A> = SE3Tangent::from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v2 = v.clone();
    assert_eq!(v.data, v2.data);
}

#[test]
fn tangent_zero() {
    let v: SE3Tangent<A, B, A> = SE3Tangent::zero();
    assert_eq!(v.data, [0.0; 6]);
}

#[test]
fn tangent_lin_ang_accessors() {
    let v: SE3Tangent<A, B, A> = SE3Tangent::from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(v.lin(), [1.0, 2.0, 3.0]);
    assert_eq!(v.ang(), [4.0, 5.0, 6.0]);
}

#[test]
fn tangent_from_lin_ang() {
    let v: SE3Tangent<A, B, A> = SE3Tangent::from_lin_ang([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    assert_eq!(v.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn tangent_hat() {
    // hat([ρ; θ]) = [[θ]×  ρ]
    //               [ 0    0]
    let v: SE3Tangent<A, B, A> = SE3Tangent::from_lin_ang([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let h = v.hat();
    // [θ]× block (top-left 3x3)
    assert_eq!(h[0][0], 0.0); assert_eq!(h[0][1], -6.0); assert_eq!(h[0][2], 5.0);
    assert_eq!(h[1][0], 6.0); assert_eq!(h[1][1], 0.0);  assert_eq!(h[1][2], -4.0);
    assert_eq!(h[2][0], -5.0); assert_eq!(h[2][1], 4.0); assert_eq!(h[2][2], 0.0);
    // ρ column (top-right)
    assert_eq!(h[0][3], 1.0);
    assert_eq!(h[1][3], 2.0);
    assert_eq!(h[2][3], 3.0);
    // bottom row is zero
    for j in 0..4 {
        assert_eq!(h[3][j], 0.0);
    }
}

proptest! {
    #[test]
    fn log_exp_roundtrip(tau in arb_se3_tangent()) {
        // log(exp(τ)) ≈ τ
        let t = tau.exp();
        let tau2 = t.log();
        prop_assert!(tangent_approx_eq(&tau, &tau2));
    }

    #[test]
    fn exp_small_tangent_produces_unit_quaternion(tau in arb_se3_tangent_small()) {
        let t = tau.exp();
        prop_assert!(abs_diff_eq!(t.quat.norm_squared(), 1.0, epsilon = EPS));
    }

    #[test]
    fn exp_large_tangent_produces_unit_quaternion(tau in arb_se3_tangent()) {
        let t = tau.exp();
        prop_assert!(abs_diff_eq!(t.quat.norm_squared(), 1.0, epsilon = EPS));
    }

    #[test]
    fn zero_exp_is_identity(_dummy in 0..1i32) {
        let tau: SE3Tangent<A, B, A> = SE3Tangent::zero();
        let t = tau.exp();
        prop_assert!(abs_diff_eq!(t.quat.norm_squared(), 1.0, epsilon = EPS));
        let roundtrip = t.log();
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(roundtrip.data[i], 0.0, epsilon = EPS));
        }
    }

    // ljac

    #[test]
    fn ljac_at_zero_is_identity(_dummy in 0..1i32) {
        let jl = SE3Tangent::<A, B, A>::zero().ljac();
        prop_assert!(mat6_approx_eq(&jl, &Mat6::identity(), EPS));
    }

    #[test]
    fn ljac_numerical_left_derivative(tau in arb_se3_tangent()) {
        // Each column of Jl(τ) ≈ Log(Exp(τ + h·eᵢ) · Exp(τ)⁻¹).data / h
        let eps = 1e-5;
        let jl = tau.ljac();
        let h = 1e-7;
        let r_tau: SE3<A, B> = tau.exp();
        let r_tau_inv: SE3<B, A> = r_tau.inverse();

        for i in 0..6 {
            let mut perturbed = tau.data;
            perturbed[i] += h;
            let tau_plus: SE3Tangent<A, B, A> = SE3Tangent::from_data(perturbed);
            let r_plus: SE3<A, B> = tau_plus.exp();
            let delta: SE3<B, B> = r_plus.compose(r_tau_inv);
            let log_delta: SE3Tangent<B, B, B> = delta.log();
            for row in 0..6 {
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
    fn ljac_small_angle_near_identity(tau in arb_se3_tangent_small()) {
        // For small τ, Jl(τ) ≈ I₆ + ½ ad(τ)
        // where ad(τ) = [[θ]×  [ρ]×]
        //               [ 0    [θ]×]
        let eps = 1e-5;
        let jl = tau.ljac();
        let w_hat = Mat3::skew(tau.ang());
        let v_hat = Mat3::skew(tau.lin());
        let mut ad = Mat6::zeros();
        ad.set_block(0, 0, &w_hat);
        ad.set_block(0, 3, &v_hat);
        ad.set_block(3, 3, &w_hat);
        let expected = Mat6::identity() + ad * 0.5;
        prop_assert!(mat6_approx_eq(&jl, &expected, eps));
    }

    // rjac

    #[test]
    fn rjac_at_zero_is_identity(_dummy in 0..1i32) {
        let jr = SE3Tangent::<A, B, A>::zero().rjac();
        prop_assert!(mat6_approx_eq(&jr, &Mat6::identity(), EPS));
    }

    #[test]
    fn rjac_numerical_right_derivative(tau in arb_se3_tangent()) {
        // Each column of Jr(τ) ≈ Log(Exp(τ)⁻¹ · Exp(τ + h·eᵢ)).data / h
        let eps = 1e-5;
        let jr = tau.rjac();
        let h = 1e-7;
        let r_tau: SE3<A, B> = tau.exp();
        let r_tau_inv: SE3<B, A> = r_tau.inverse();

        for i in 0..6 {
            let mut perturbed = tau.data;
            perturbed[i] += h;
            let tau_plus: SE3Tangent<A, B, A> = SE3Tangent::from_data(perturbed);
            let r_plus: SE3<A, B> = tau_plus.exp();
            let delta: SE3<A, A> = r_tau_inv.compose(r_plus);
            let log_delta: SE3Tangent<A, A, A> = delta.log();
            for row in 0..6 {
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
    fn rjac_small_angle_near_identity(tau in arb_se3_tangent_small()) {
        // For small τ, Jr(τ) = Jl(-τ) ≈ I₆ - ½ ad(τ)
        let eps = 1e-5;
        let jr = tau.rjac();
        let w_hat = Mat3::skew(tau.ang());
        let v_hat = Mat3::skew(tau.lin());
        let mut ad = Mat6::zeros();
        ad.set_block(0, 0, &w_hat);
        ad.set_block(0, 3, &v_hat);
        ad.set_block(3, 3, &w_hat);
        let expected = Mat6::identity() - ad * 0.5;
        prop_assert!(mat6_approx_eq(&jr, &expected, eps));
    }

    // ljacinv

    #[test]
    fn ljacinv_at_zero_is_identity(_dummy in 0..1i32) {
        let jlinv = SE3Tangent::<A, B, A>::zero().ljacinv();
        prop_assert!(mat6_approx_eq(&jlinv, &Mat6::identity(), EPS));
    }

    #[test]
    fn ljacinv_numerical_left_derivative(tau in arb_se3_tangent()) {
        // Jl⁻¹ maps numerical Jl columns back to basis vectors
        let eps = 1e-5;
        let jlinv = tau.ljacinv();
        let h = 1e-7;
        let r_tau: SE3<A, B> = tau.exp();
        let r_tau_inv: SE3<B, A> = r_tau.inverse();

        for i in 0..6 {
            let mut perturbed = tau.data;
            perturbed[i] += h;
            let tau_plus: SE3Tangent<A, B, A> = SE3Tangent::from_data(perturbed);
            let r_plus: SE3<A, B> = tau_plus.exp();
            let delta: SE3<B, B> = r_plus.compose(r_tau_inv);
            let log_delta: SE3Tangent<B, B, B> = delta.log();
            // Jl maps eᵢ to log_delta/h; so Jl⁻¹ maps log_delta/h back to eᵢ
            let recovered = jlinv * log_delta.data;
            for row in 0..6 {
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
    fn ljacinv_small_angle_near_identity(tau in arb_se3_tangent_small()) {
        // For small τ, Jl⁻¹(τ) ≈ I₆ - ½ ad(τ)
        let eps = 1e-5;
        let jlinv = tau.ljacinv();
        let w_hat = Mat3::skew(tau.ang());
        let v_hat = Mat3::skew(tau.lin());
        let mut ad = Mat6::zeros();
        ad.set_block(0, 0, &w_hat);
        ad.set_block(0, 3, &v_hat);
        ad.set_block(3, 3, &w_hat);
        let expected = Mat6::identity() - ad * 0.5;
        prop_assert!(mat6_approx_eq(&jlinv, &expected, eps));
    }

    #[test]
    fn ljacinv_is_inverse_of_ljac(tau in arb_se3_tangent()) {
        // Jl⁻¹(τ) · Jl(τ) = I₆
        let product = tau.ljacinv() * tau.ljac();
        prop_assert!(mat6_approx_eq(&product, &Mat6::identity(), EPS));
    }

    // rjacinv

    #[test]
    fn rjacinv_at_zero_is_identity(_dummy in 0..1i32) {
        let jrinv = SE3Tangent::<A, B, A>::zero().rjacinv();
        prop_assert!(mat6_approx_eq(&jrinv, &Mat6::identity(), EPS));
    }

    #[test]
    fn rjacinv_numerical_right_derivative(tau in arb_se3_tangent()) {
        // Jr⁻¹ maps numerical Jr columns back to basis vectors
        let eps = 1e-5;
        let jrinv = tau.rjacinv();
        let h = 1e-7;
        let r_tau: SE3<A, B> = tau.exp();
        let r_tau_inv: SE3<B, A> = r_tau.inverse();

        for i in 0..6 {
            let mut perturbed = tau.data;
            perturbed[i] += h;
            let tau_plus: SE3Tangent<A, B, A> = SE3Tangent::from_data(perturbed);
            let r_plus: SE3<A, B> = tau_plus.exp();
            let delta: SE3<A, A> = r_tau_inv.compose(r_plus);
            let log_delta: SE3Tangent<A, A, A> = delta.log();
            let recovered = jrinv * log_delta.data;
            for row in 0..6 {
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
    fn rjacinv_small_angle_near_identity(tau in arb_se3_tangent_small()) {
        // For small τ, Jr⁻¹(τ) = Jl⁻¹(-τ) ≈ I₆ + ½ ad(τ)
        let eps = 1e-5;
        let jrinv = tau.rjacinv();
        let w_hat = Mat3::skew(tau.ang());
        let v_hat = Mat3::skew(tau.lin());
        let mut ad = Mat6::zeros();
        ad.set_block(0, 0, &w_hat);
        ad.set_block(0, 3, &v_hat);
        ad.set_block(3, 3, &w_hat);
        let expected = Mat6::identity() + ad * 0.5;
        prop_assert!(mat6_approx_eq(&jrinv, &expected, eps));
    }

    #[test]
    fn rjacinv_is_inverse_of_rjac(tau in arb_se3_tangent()) {
        // Jr⁻¹(τ) · Jr(τ) = I₆
        let product = tau.rjacinv() * tau.rjac();
        prop_assert!(mat6_approx_eq(&product, &Mat6::identity(), EPS));
    }

    // ad

    // compose (BCH)

    #[test]
    fn compose_zero_lhs(tau in arb_se3_tangent()) {
        // 0.bch_compose(τ) ≈ τ
        let zero: SE3Tangent<A, B, A> = SE3Tangent::zero();
        let result = zero.bch_compose(tau);
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(result.data[i], tau.data[i], epsilon = EPS));
        }
    }

    #[test]
    fn compose_zero_rhs(tau in arb_se3_tangent()) {
        // τ.bch_compose(0) ≈ τ
        let zero: SE3Tangent<A, B, A> = SE3Tangent::zero();
        let result = tau.bch_compose(zero);
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(result.data[i], tau.data[i], epsilon = EPS));
        }
    }

    #[test]
    fn compose_approximates_exact(tau1 in arb_se3_tangent(), tau2 in arb_se3_tangent()) {
        // BCH should approximate log(exp(τ₁) · exp(τ₂))
        // Accuracy is O(|τ|³), so scale down for a tighter bound
        let s = 0.05;
        let t1: SE3Tangent<A, A, A> = SE3Tangent::from_data(std::array::from_fn(|i| tau1.data[i] * s));
        let t2: SE3Tangent<A, A, A> = SE3Tangent::from_data(std::array::from_fn(|i| tau2.data[i] * s));
        let bch = t1.bch_compose(t2);
        let exact = t1.exp().compose(t2.exp()).log();
        let eps = 1e-2;
        for i in 0..6 {
            prop_assert!(
                abs_diff_eq!(bch.data[i], exact.data[i], epsilon = eps),
                "BCH mismatch at {}: bch={}, exact={}",
                i, bch.data[i], exact.data[i]
            );
        }
    }

    #[test]
    fn compose_with_negation_near_zero(tau in arb_se3_tangent()) {
        // τ.bch_compose(-τ) ≈ 0 (for small τ, exact at τ = 0)
        let s = 0.1;
        let t: SE3Tangent<A, A, A> = SE3Tangent::from_data(std::array::from_fn(|i| tau.data[i] * s));
        let neg: SE3Tangent<A, A, A> = SE3Tangent::from_data(std::array::from_fn(|i| -t.data[i]));
        let result = t.bch_compose(neg);
        let eps = 1e-2;
        prop_assert!(abs_diff_eq!(result.norm(), 0.0, epsilon = eps));
    }

    // ad

    #[test]
    fn ad_zero_is_zero(_dummy in 0..1i32) {
        let zero: SE3Tangent<A, B, A> = SE3Tangent::zero();
        let ad = zero.ad();
        prop_assert!(mat6_approx_eq(&ad, &Mat6::zeros(), EPS));
    }

    #[test]
    fn ad_structure(tau in arb_se3_tangent()) {
        // ad(τ) = [[ω]×  [v]×]
        //         [ 0    [ω]×]
        let ad = tau.ad();
        let wx = Mat3::skew(tau.ang());
        let vx = Mat3::skew(tau.lin());
        let mut expected = Mat6::zeros();
        expected.set_block(0, 0, &wx);
        expected.set_block(0, 3, &vx);
        expected.set_block(3, 3, &wx);
        prop_assert!(mat6_approx_eq(&ad, &expected, EPS));
    }

    #[test]
    fn ad_self_bracket_is_zero(tau in arb_se3_tangent()) {
        // ad(τ) · τ = [τ, τ] = 0
        let result = tau.ad() * tau.data;
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(result[i], 0.0, epsilon = EPS));
        }
    }

    #[test]
    fn ad_antisymmetric(tau1 in arb_se3_tangent(), tau2 in arb_se3_tangent()) {
        // ad(τ₁) · τ₂ = -ad(τ₂) · τ₁
        let lhs = tau1.ad() * tau2.data;
        let rhs = tau2.ad() * tau1.data;
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(lhs[i], -rhs[i], epsilon = EPS));
        }
    }

    #[test]
    fn ad_jacobi_identity(tau1 in arb_se3_tangent(), tau2 in arb_se3_tangent(), tau3 in arb_se3_tangent()) {
        // ad(τ₁)·(ad(τ₂)·τ₃) + ad(τ₂)·(ad(τ₃)·τ₁) + ad(τ₃)·(ad(τ₁)·τ₂) = 0
        let a = tau1.ad() * (tau2.ad() * tau3.data);
        let b = tau2.ad() * (tau3.ad() * tau1.data);
        let c = tau3.ad() * (tau1.ad() * tau2.data);
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(a[i] + b[i] + c[i], 0.0, epsilon = EPS));
        }
    }

    #[test]
    fn ad_ljac_small_angle(tau in arb_se3_tangent_small()) {
        // For small τ, Jl(τ) ≈ I + ½ ad(τ)
        let eps = 1e-5;
        let jl = tau.ljac();
        let expected = Mat6::identity() + tau.ad() * 0.5;
        prop_assert!(mat6_approx_eq(&jl, &expected, eps));
    }
}
