use approx::abs_diff_eq;
use cardo::prelude::*;
use proptest::prelude::*;
use std::f64::consts::PI;

struct A;
struct B;
struct C;
struct D;

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

// Strategy for generating random rotations via exp
fn arb_so3() -> impl Strategy<Value = SO3<A, B>> {
    arb_unit_quat().prop_map(SO3::from_quat)
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

// Strategy for generating unit quaternions
fn arb_unit_quat() -> impl Strategy<Value = Quat<f64>> {
    (-1.0..1.0f64, -1.0..1.0f64, -1.0..1.0f64, -1.0..1.0f64)
        .prop_filter_map("non-zero quaternion", |(w, x, y, z)| {
            let q = Quat::new(w, x, y, z);
            if q.norm_squared() > 0.01 {
                Some(q.normalized())
            } else {
                None
            }
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
    fn exp_log_roundtrip(r in arb_so3()) {
        // exp(log(r)) ≈ r
        let v = r.log();
        let r2 = SO3::<A, B>::exp(&v);
        prop_assert!(quat_approx_eq(&r.quat, &r2.quat));
    }

    #[test]
    fn log_exp_roundtrip(v in arb_tangent()) {
        // log(exp(v)) ≈ v
        let r = SO3::<A, B>::exp(&v);
        let v2 = r.log();
        prop_assert!(tangent_approx_eq(&v, &v2));
    }

    #[test]
    fn exp_small_angle_produces_unit_quaternion(v in arb_tangent_small()) {
        let r = SO3::<A, B>::exp(&v);
        prop_assert!(abs_diff_eq!(r.quat.norm_squared(), 1.0, epsilon = EPS));
    }

    #[test]
    fn exp_large_angle_produces_unit_quaternion(v in arb_tangent()) {
        let r = SO3::<A, B>::exp(&v);
        prop_assert!(abs_diff_eq!(r.quat.norm_squared(), 1.0, epsilon = EPS));
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

    #[test]
    fn mul_vec_does_act(r in arb_so3(), x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64) {
        let v: Vector3<A> = Vector3::new(x, y, z);
        let via_mul = r * v;
        let via_act = r.act(v);
        prop_assert!(abs_diff_eq!(via_mul.x(), via_act.x(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(via_mul.y(), via_act.y(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(via_mul.z(), via_act.z(), epsilon = EPS));
    }

    #[test]
    fn mul_group_does_compose(q1 in arb_unit_quat(), q2 in arb_unit_quat()) {
        let r1: SO3<A, B> = SO3::from_quat(q1);
        let r2: SO3<C, A> = SO3::from_quat(q2);
        let via_mul = r1 * r2;
        let via_compose = r1.compose(r2);
        prop_assert!(quat_approx_eq(&via_mul.quat, &via_compose.quat));
    }

    #[test]
    fn from_quat_preserves_rotation(q in arb_unit_quat(), x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64) {
        let r: SO3<A, B> = SO3::from_quat(q);
        let v: Vector3<A> = Vector3::new(x, y, z);
        let v_rotated = r.act(v);
        prop_assert!(abs_diff_eq!(v.norm(), v_rotated.norm(), epsilon = EPS));
    }

    #[test]
    fn from_axis_angle_zero_is_identity(axis in arb_unit_axis()) {
        let r: SO3<A, B> = SO3::from_axis_angle(&axis, 0.0);
        let id: SO3<A, B> = SO3::identity();
        prop_assert!(quat_approx_eq(&r.quat, &id.quat));
    }

    #[test]
    fn from_axis_angle_opposite_angles_cancel(axis in arb_unit_axis(), angle in -PI..PI) {
        let r1: SO3<A, B> = SO3::from_axis_angle(&axis, angle);
        let r2: SO3<B, A> = SO3::from_axis_angle(&Vector3::new(axis.x(), axis.y(), axis.z()), -angle);
        let composed: SO3<A, A> = r2 * r1;
        let id: SO3<A, A> = SO3::identity();
        prop_assert!(quat_approx_eq(&composed.quat, &id.quat));
    }

    #[test]
    fn from_axis_angle_preserves_norm(axis in arb_unit_axis(), angle in -PI..PI, x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64) {
        let r: SO3<A, B> = SO3::from_axis_angle(&axis, angle);
        let v: Vector3<A> = Vector3::new(x, y, z);
        let v_rotated = r.act(v);
        prop_assert!(abs_diff_eq!(v.norm(), v_rotated.norm(), epsilon = EPS));
    }

    #[test]
    fn full_rotation_in_n_steps_is_identity(
        axis in arb_unit_axis(),
        n in 2..100i32,
        x in -10.0..10.0f64,
        y in -10.0..10.0f64,
        z in -10.0..10.0f64
    ) {
        let angle = 2.0 * PI / (n as f64);
        let r: SO3<A, A> = SO3::from_axis_angle(&axis, angle);
        let v: Vector3<A> = Vector3::new(x, y, z);
        let v_rotated = (0..n).fold(v, |acc, _| r.act(acc));
        prop_assert!(abs_diff_eq!(v.x(), v_rotated.x(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v.y(), v_rotated.y(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v.z(), v_rotated.z(), epsilon = EPS));
    }

    #[test]
    fn then_is_left_to_right_composition(
        q1 in arb_unit_quat(),
        q2 in arb_unit_quat(),
        x in -10.0..10.0f64,
        y in -10.0..10.0f64,
        z in -10.0..10.0f64
    ) {
        // r1.then(r2).act(v) = r2.act(r1.act(v))
        let r1: SO3<A, B> = SO3::from_quat(q1);
        let r2: SO3<B, C> = SO3::from_quat(q2);
        let v: Vector3<A> = Vector3::new(x, y, z);
        let v1 = r1.then(r2).act(v);
        let v2 = r2.act(r1.act(v));
        prop_assert!(abs_diff_eq!(v1.x(), v2.x(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1.y(), v2.y(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1.z(), v2.z(), epsilon = EPS));
    }

    #[test]
    fn compose_is_right_to_left_composition(
        q1 in arb_unit_quat(),
        q2 in arb_unit_quat(),
        x in -10.0..10.0f64,
        y in -10.0..10.0f64,
        z in -10.0..10.0f64
    ) {
        // r1.compose(r2).act(v) = r1.act(r2.act(v))
        let r1: SO3<A, B> = SO3::from_quat(q1);
        let r2: SO3<C, A> = SO3::from_quat(q2);
        let v: Vector3<C> = Vector3::new(x, y, z);
        let v1 = r1.compose(r2).act(v);
        let v2 = r1.act(r2.act(v));
        prop_assert!(abs_diff_eq!(v1.x(), v2.x(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1.y(), v2.y(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1.z(), v2.z(), epsilon = EPS));
    }

    #[test]
    fn action_distributes_over_composition(
        q1 in arb_unit_quat(),
        q2 in arb_unit_quat(),
        x in -10.0..10.0f64,
        y in -10.0..10.0f64,
        z in -10.0..10.0f64
    ) {
        // (r1 * r2).act(v) ≈ r1.act(r2.act(v))
        let r1: SO3<B, C> = SO3::from_quat(q1);
        let r2: SO3<A, B> = SO3::from_quat(q2);
        let v: Vector3<A> = Vector3::new(x, y, z);
        let composed = r1 * r2;
        let v1 = composed.act(v);
        let v2 = r1.act(r2.act(v));
        prop_assert!(abs_diff_eq!(v1.x(), v2.x(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1.y(), v2.y(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1.z(), v2.z(), epsilon = EPS));
    }

    #[test]
    fn quaternion_double_cover(
        q in arb_unit_quat(),
        x in -10.0..10.0f64,
        y in -10.0..10.0f64,
        z in -10.0..10.0f64
    ) {
        // q and -q produce the same rotation
        let (w, qx, qy, qz) = q.wxyz();
        let neg_q = Quat::new(-w, -qx, -qy, -qz);
        let r1: SO3<A, B> = SO3::from_quat(q);
        let r2: SO3<A, B> = SO3::from_quat(neg_q);
        let v: Vector3<A> = Vector3::new(x, y, z);
        let v1 = r1.act(v);
        let v2 = r2.act(v);
        prop_assert!(abs_diff_eq!(v1.x(), v2.x(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1.y(), v2.y(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1.z(), v2.z(), epsilon = EPS));
    }

    #[test]
    fn axis_angle_log_consistency(axis in arb_unit_axis(), angle in -PI..PI) {
        // log(from_axis_angle(axis, θ)) ≈ θ * axis
        let r: SO3<A, B> = SO3::from_axis_angle(&axis, angle);
        let v = r.log();
        let expected_x = angle * axis.x();
        let expected_y = angle * axis.y();
        let expected_z = angle * axis.z();
        prop_assert!(abs_diff_eq!(v.x(), expected_x, epsilon = EPS));
        prop_assert!(abs_diff_eq!(v.y(), expected_y, epsilon = EPS));
        prop_assert!(abs_diff_eq!(v.z(), expected_z, epsilon = EPS));
    }

    #[test]
    fn rotation_axis_is_fixed_point(axis in arb_unit_axis(), angle in -PI..PI) {
        // Rotating the axis about itself leaves it unchanged
        let r: SO3<A, B> = SO3::from_axis_angle(&axis, angle);
        let axis_in_b: Vector3<B> = r.act(axis);
        prop_assert!(abs_diff_eq!(axis.x(), axis_in_b.x(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(axis.y(), axis_in_b.y(), epsilon = EPS));
        prop_assert!(abs_diff_eq!(axis.z(), axis_in_b.z(), epsilon = EPS));
    }
}

// Deterministic tests for known rotations
use std::f64::consts::FRAC_PI_2;

#[test]
fn rotate_90_about_z() {
    // 90° about Z: (1,0,0) → (0,1,0)
    let axis: Vector3<A> = Vector3::new(0.0, 0.0, 1.0);
    let r: SO3<A, B> = SO3::from_axis_angle(&axis, FRAC_PI_2);
    let v: Vector3<A> = Vector3::new(1.0, 0.0, 0.0);
    let result = r.act(v);
    assert!(abs_diff_eq!(result.x(), 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.y(), 1.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.z(), 0.0, epsilon = EPS));
}

#[test]
fn rotate_90_about_x() {
    // 90° about X: (0,1,0) → (0,0,1)
    let axis: Vector3<A> = Vector3::new(1.0, 0.0, 0.0);
    let r: SO3<A, B> = SO3::from_axis_angle(&axis, FRAC_PI_2);
    let v: Vector3<A> = Vector3::new(0.0, 1.0, 0.0);
    let result = r.act(v);
    assert!(abs_diff_eq!(result.x(), 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.y(), 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.z(), 1.0, epsilon = EPS));
}

#[test]
fn rotate_90_about_y() {
    // 90° about Y: (0,0,1) → (1,0,0)
    let axis: Vector3<A> = Vector3::new(0.0, 1.0, 0.0);
    let r: SO3<A, B> = SO3::from_axis_angle(&axis, FRAC_PI_2);
    let v: Vector3<A> = Vector3::new(0.0, 0.0, 1.0);
    let result = r.act(v);
    assert!(abs_diff_eq!(result.x(), 1.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.y(), 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.z(), 0.0, epsilon = EPS));
}

#[test]
fn rotate_180_about_z() {
    // 180° about Z: (1,0,0) → (-1,0,0)
    let axis: Vector3<A> = Vector3::new(0.0, 0.0, 1.0);
    let r: SO3<A, B> = SO3::from_axis_angle(&axis, PI);
    let v: Vector3<A> = Vector3::new(1.0, 0.0, 0.0);
    let result = r.act(v);
    assert!(abs_diff_eq!(result.x(), -1.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.y(), 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.z(), 0.0, epsilon = EPS));
}

#[test]
fn rotate_180_about_arbitrary_axis() {
    // 180° about (1,1,0)/√2: (1,0,0) → (0,1,0)
    let s = 1.0 / 2.0_f64.sqrt();
    let axis: Vector3<A> = Vector3::new(s, s, 0.0);
    let r: SO3<A, B> = SO3::from_axis_angle(&axis, PI);
    let v: Vector3<A> = Vector3::new(1.0, 0.0, 0.0);
    let result = r.act(v);
    assert!(abs_diff_eq!(result.x(), 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.y(), 1.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.z(), 0.0, epsilon = EPS));
}

// Tests for chain! macro
#[test]
fn chain_arrow_right_two_rotations() {
    let axis: Vector3<A> = Vector3::new(0.0, 0.0, 1.0);
    let r1: SO3<A, B> = SO3::from_axis_angle(&axis, FRAC_PI_2);
    let r2: SO3<B, C> = SO3::from_axis_angle(&Vector3::new(0.0, 0.0, 1.0), FRAC_PI_2);

    let chained: SO3<A, C> = chain!(r1 -> r2);
    let manual: SO3<A, C> = r1.then(r2);

    assert!(quat_approx_eq(&chained.quat, &manual.quat));
}

#[test]
fn chain_arrow_left_two_rotations() {
    let axis: Vector3<A> = Vector3::new(0.0, 0.0, 1.0);
    let r1: SO3<A, B> = SO3::from_axis_angle(&axis, FRAC_PI_2);
    let r2: SO3<B, C> = SO3::from_axis_angle(&Vector3::new(0.0, 0.0, 1.0), FRAC_PI_2);

    let chained: SO3<A, C> = chain!(r2 <- r1);
    let manual: SO3<A, C> = r2.compose(r1);

    assert!(quat_approx_eq(&chained.quat, &manual.quat));
}

#[test]
fn chain_arrow_right_three_rotations() {
    let r1: SO3<A, B> = SO3::from_axis_angle(&Vector3::new(1.0, 0.0, 0.0), FRAC_PI_2);
    let r2: SO3<B, C> = SO3::from_axis_angle(&Vector3::new(0.0, 1.0, 0.0), FRAC_PI_2);
    let r3: SO3<C, D> = SO3::from_axis_angle(&Vector3::new(0.0, 0.0, 1.0), FRAC_PI_2);

    let chained: SO3<A, D> = chain!(r1 -> r2 -> r3);
    let manual: SO3<A, D> = r1.then(r2).then(r3);

    assert!(quat_approx_eq(&chained.quat, &manual.quat));
}

#[test]
fn chain_arrow_left_three_rotations() {
    let r1: SO3<A, B> = SO3::from_axis_angle(&Vector3::new(1.0, 0.0, 0.0), FRAC_PI_2);
    let r2: SO3<B, C> = SO3::from_axis_angle(&Vector3::new(0.0, 1.0, 0.0), FRAC_PI_2);
    let r3: SO3<C, D> = SO3::from_axis_angle(&Vector3::new(0.0, 0.0, 1.0), FRAC_PI_2);

    let chained: SO3<A, D> = chain!(r3 <- r2 <- r1);
    let manual: SO3<A, D> = r3.compose(r2).compose(r1);

    assert!(quat_approx_eq(&chained.quat, &manual.quat));
}

#[test]
fn chain_both_directions_equivalent() {
    // chain!(r1 -> r2 -> r3) should equal chain!(r3 <- r2 <- r1)
    let r1: SO3<A, B> = SO3::from_axis_angle(&Vector3::new(1.0, 0.0, 0.0), 0.5);
    let r2: SO3<B, C> = SO3::from_axis_angle(&Vector3::new(0.0, 1.0, 0.0), 0.7);
    let r3: SO3<C, D> = SO3::from_axis_angle(&Vector3::new(0.0, 0.0, 1.0), 0.3);

    let left_to_right: SO3<A, D> = chain!(r1 -> r2 -> r3);
    let right_to_left: SO3<A, D> = chain!(r3 <- r2 <- r1);

    assert!(quat_approx_eq(&left_to_right.quat, &right_to_left.quat));
}

// Tests for panic on non-unit inputs
#[test]
#[should_panic]
fn from_quat_panics_on_non_unit() {
    let _: SO3<A, B> = SO3::from_quat(Quat::new(1.0, 1.0, 1.0, 1.0));
}

#[test]
#[should_panic]
fn from_axis_angle_panics_on_non_unit() {
    let axis: Vector3<A> = Vector3::new(1.0, 1.0, 1.0);
    let _: SO3<A, B> = SO3::from_axis_angle(&axis, 1.0);
}

// Tests for Debug and Clone
#[test]
fn so3_debug() {
    let r: SO3<A, B> = SO3::identity();
    let s = format!("{:?}", r);
    assert!(s.contains("SO3"));
}

#[test]
fn so3_clone() {
    let r: SO3<A, B> = SO3::identity();
    let r2 = r.clone();
    assert!(quat_approx_eq(&r.quat, &r2.quat));
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
