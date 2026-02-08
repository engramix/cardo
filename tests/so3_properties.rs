use approx::abs_diff_eq;
use cardo::prelude::*;
use proptest::prelude::*;
use std::f64::consts::PI;

#[derive(Debug)]
struct A;
#[derive(Debug)]
struct B;

const EPS: f64 = 1e-10;

fn quat_approx_eq(q1: &Quat<f64>, q2: &Quat<f64>) -> bool {
    let (w1, x1, y1, z1) = q1.wxyz();
    let (w2, x2, y2, z2) = q2.wxyz();
    // Quaternions q and -q represent the same rotation
    let same = abs_diff_eq!(w1, w2, epsilon = EPS)
        && abs_diff_eq!(x1, x2, epsilon = EPS)
        && abs_diff_eq!(y1, y2, epsilon = EPS)
        && abs_diff_eq!(z1, z2, epsilon = EPS);
    let negated = abs_diff_eq!(w1, -w2, epsilon = EPS)
        && abs_diff_eq!(x1, -x2, epsilon = EPS)
        && abs_diff_eq!(y1, -y2, epsilon = EPS)
        && abs_diff_eq!(z1, -z2, epsilon = EPS);
    same || negated
}

fn tangent_approx_eq(v1: &SO3Tangent<A, B, A>, v2: &SO3Tangent<A, B, A>) -> bool {
    abs_diff_eq!(v1.x(), v2.x(), epsilon = EPS)
        && abs_diff_eq!(v1.y(), v2.y(), epsilon = EPS)
        && abs_diff_eq!(v1.z(), v2.z(), epsilon = EPS)
}

// Strategy for generating random tangent vectors (covers full rotation space)
fn arb_tangent() -> impl Strategy<Value = SO3Tangent<A, B, A>> {
    (-PI..PI, -PI..PI, -PI..PI).prop_map(|(x, y, z)| SO3Tangent::new(x, y, z))
}

// Strategy for generating random rotations via exp
fn arb_so3() -> impl Strategy<Value = SO3<A, B>> {
    arb_tangent().prop_map(|v| SO3::exp(&v))
}

// Strategy for generating small tangent vectors (for log/exp roundtrip stability)
fn arb_tangent_small() -> impl Strategy<Value = SO3Tangent<A, B, A>> {
    (-1.0..1.0f64, -1.0..1.0f64, -1.0..1.0f64)
        .prop_map(|(x, y, z)| SO3Tangent::new(x, y, z))
}

proptest! {
    #[test]
    fn exp_log_roundtrip(r in arb_so3()) {
        // exp(log(r)) ≈ r
        let v = r.log();
        let r2 = SO3::<A, B>::exp(&v);
        prop_assert!(quat_approx_eq(&r.quat, &r2.quat));
    }

    #[test]
    fn log_exp_roundtrip(v in arb_tangent_small()) {
        // log(exp(v)) ≈ v (for small v)
        let r = SO3::<A, B>::exp(&v);
        let v2 = r.log();
        prop_assert!(tangent_approx_eq(&v, &v2));
    }

    #[test]
    fn identity_log_is_zero(_dummy in 0..1i32) {
        let r: SO3<A, B> = SO3::identity();
        let v = r.log();
        prop_assert!(abs_diff_eq!(v.x(), 0.0, epsilon = EPS));
        prop_assert!(abs_diff_eq!(v.y(), 0.0, epsilon = EPS));
        prop_assert!(abs_diff_eq!(v.z(), 0.0, epsilon = EPS));
    }

    #[test]
    fn zero_exp_is_identity(_dummy in 0..1i32) {
        let v: SO3Tangent<A, B, A> = SO3Tangent::new(0.0, 0.0, 0.0);
        let r = SO3::<A, B>::exp(&v);
        let id: SO3<A, B> = SO3::identity();
        prop_assert!(quat_approx_eq(&r.quat, &id.quat));
    }

    #[test]
    fn inverse_composition_is_identity(r in arb_so3()) {
        // r * r.inverse() ≈ identity
        let r_inv = r.inverse();
        let composed = r.compose(r_inv);
        let id: SO3<A, A> = SO3::identity();
        prop_assert!(quat_approx_eq(&composed.quat, &id.quat));
    }

    #[test]
    fn rotation_preserves_norm(r in arb_so3(), x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64) {
        let v: Vector3<A> = Vector3::new(x, y, z);
        let v_rotated = r.act(v);
        prop_assert!(abs_diff_eq!(v.norm(), v_rotated.norm(), epsilon = EPS));
    }
}
