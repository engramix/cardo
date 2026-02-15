use approx::abs_diff_eq;
use cardo::prelude::*;
use proptest::prelude::*;
use std::f64::consts::PI;

struct A;
struct B;
struct C;
struct E;

const EPS: f64 = 1e-10;

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

fn arb_se3() -> impl Strategy<Value = SE3<A, B>> {
    (arb_unit_quat(), -10.0..10.0f64, -10.0..10.0f64, -10.0..10.0f64)
        .prop_map(|(q, tx, ty, tz)| SE3::from_vec_quat([tx, ty, tz], q))
}

fn quat_approx_eq(q1: &Quat<f64>, q2: &Quat<f64>) -> bool {
    let (w1, x1, y1, z1) = q1.wxyz();
    let (w2, x2, y2, z2) = q2.wxyz();
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
    // Bound angle away from zero so numerical Jacobian tests don't cross the θ=0
    // boundary (where exp/log Taylor branches change, causing large finite-difference error).
    let angle = prop_oneof![
        -PI..-1e-3f64,
        1e-3..PI,
    ];
    (arb_unit_axis(), angle, -10.0..10.0f64, -10.0..10.0f64, -10.0..10.0f64)
        .prop_map(|(axis, angle, vx, vy, vz)| {
            SE3Tangent::from_lin_ang(
                [vx, vy, vz],
                [axis.x() * angle, axis.y() * angle, axis.z() * angle],
            )
        })
}

fn mat6_approx_eq(a: &Mat6, b: &Mat6) -> bool {
    (0..6).all(|i| (0..6).all(|j| abs_diff_eq!(a.data[i][j], b.data[i][j], epsilon = EPS)))
}

fn se3_approx_eq(a: &SE3<A, B>, b: &SE3<A, B>) -> bool {
    let (w1, x1, y1, z1) = a.quat.wxyz();
    let (w2, x2, y2, z2) = b.quat.wxyz();
    let quat_eq = (abs_diff_eq!(w1, w2, epsilon = EPS)
        && abs_diff_eq!(x1, x2, epsilon = EPS)
        && abs_diff_eq!(y1, y2, epsilon = EPS)
        && abs_diff_eq!(z1, z2, epsilon = EPS))
        || (abs_diff_eq!(w1, -w2, epsilon = EPS)
            && abs_diff_eq!(x1, -x2, epsilon = EPS)
            && abs_diff_eq!(y1, -y2, epsilon = EPS)
            && abs_diff_eq!(z1, -z2, epsilon = EPS));
    let vec_eq = abs_diff_eq!(a.vec[0], b.vec[0], epsilon = EPS)
        && abs_diff_eq!(a.vec[1], b.vec[1], epsilon = EPS)
        && abs_diff_eq!(a.vec[2], b.vec[2], epsilon = EPS);
    quat_eq && vec_eq
}

// Construction

#[test]
fn identity_has_zero_translation() {
    let id: SE3<A, B> = SE3::identity();
    assert_eq!(id.vec, [0.0, 0.0, 0.0]);
}

#[test]
fn identity_has_identity_rotation() {
    let id: SE3<A, B> = SE3::identity();
    let (w, x, y, z) = id.quat.wxyz();
    assert_eq!((w, x, y, z), (1.0, 0.0, 0.0, 0.0));
}

#[test]
fn from_vec_quat_stores_components() {
    let q = Quat::new(1.0, 0.0, 0.0, 0.0);
    let t = [1.0, 2.0, 3.0];
    let se3: SE3<A, B> = SE3::from_vec_quat(t, q);
    assert_eq!(se3.vec, t);
    assert_eq!(se3.quat, q);
}

#[test]
#[should_panic]
fn from_vec_quat_panics_on_non_unit() {
    let _: SE3<A, B> = SE3::from_vec_quat([0.0; 3], Quat::new(1.0, 1.0, 1.0, 1.0));
}

// Traits

#[test]
fn debug_format() {
    let se3: SE3<A, B> = SE3::identity();
    let s = format!("{:?}", se3);
    assert!(s.contains("SE3"));
}

#[test]
fn clone() {
    let se3: SE3<A, B> = SE3::from_vec_quat([1.0, 2.0, 3.0], Quat::new(1.0, 0.0, 0.0, 0.0));
    let se3_clone = se3.clone();
    assert!(se3_approx_eq(&se3, &se3_clone));
}

#[test]
fn copy() {
    let se3: SE3<A, B> = SE3::from_vec_quat([1.0, 2.0, 3.0], Quat::new(1.0, 0.0, 0.0, 0.0));
    let se3_copy = se3;
    assert!(se3_approx_eq(&se3, &se3_copy));
}

#[test]
fn eq() {
    let a: SE3<A, B> = SE3::from_vec_quat([1.0, 2.0, 3.0], Quat::new(1.0, 0.0, 0.0, 0.0));
    let b: SE3<A, B> = SE3::from_vec_quat([1.0, 2.0, 3.0], Quat::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(a, b);
}

// from_quat: pure rotation, zero translation
#[test]
fn from_quat_has_zero_translation() {
    let q = Quat::new(1.0, 0.0, 0.0, 0.0);
    let se3: SE3<A, B> = SE3::from_quat(q);
    assert_eq!(se3.vec, [0.0, 0.0, 0.0]);
    assert_eq!(se3.quat, q);
}

// from_vec: identity rotation, given translation
#[test]
fn from_vec_has_identity_rotation() {
    let se3: SE3<A, B> = SE3::from_vec([1.0, 2.0, 3.0]);
    assert_eq!(se3.vec, [1.0, 2.0, 3.0]);
    let (w, x, y, z) = se3.quat.wxyz();
    assert_eq!((w, x, y, z), (1.0, 0.0, 0.0, 0.0));
}

// rot_x/y/z: same rotation as SO3::rot_x/y/z, zero translation
use std::f64::consts::FRAC_PI_2;

#[test]
fn rot_x_matches_so3() {
    let se3: SE3<A, B> = SE3::rot_x(FRAC_PI_2);
    let so3: SO3<A, B> = SO3::rot_x(FRAC_PI_2);
    assert_eq!(se3.quat, so3.quat);
    assert_eq!(se3.vec, [0.0, 0.0, 0.0]);
}

#[test]
fn rot_y_matches_so3() {
    let se3: SE3<A, B> = SE3::rot_y(FRAC_PI_2);
    let so3: SO3<A, B> = SO3::rot_y(FRAC_PI_2);
    assert_eq!(se3.quat, so3.quat);
    assert_eq!(se3.vec, [0.0, 0.0, 0.0]);
}

#[test]
fn rot_z_matches_so3() {
    let se3: SE3<A, B> = SE3::rot_z(FRAC_PI_2);
    let so3: SO3<A, B> = SO3::rot_z(FRAC_PI_2);
    assert_eq!(se3.quat, so3.quat);
    assert_eq!(se3.vec, [0.0, 0.0, 0.0]);
}

// trans_x/y/z: identity rotation, translation along axis
#[test]
fn trans_x_translates_along_x() {
    let se3: SE3<A, B> = SE3::trans_x(5.0);
    assert_eq!(se3.vec, [5.0, 0.0, 0.0]);
    let (w, x, y, z) = se3.quat.wxyz();
    assert_eq!((w, x, y, z), (1.0, 0.0, 0.0, 0.0));
}

#[test]
fn trans_y_translates_along_y() {
    let se3: SE3<A, B> = SE3::trans_y(5.0);
    assert_eq!(se3.vec, [0.0, 5.0, 0.0]);
    let (w, x, y, z) = se3.quat.wxyz();
    assert_eq!((w, x, y, z), (1.0, 0.0, 0.0, 0.0));
}

#[test]
fn trans_z_translates_along_z() {
    let se3: SE3<A, B> = SE3::trans_z(5.0);
    assert_eq!(se3.vec, [0.0, 0.0, 5.0]);
    let (w, x, y, z) = se3.quat.wxyz();
    assert_eq!((w, x, y, z), (1.0, 0.0, 0.0, 0.0));
}

// Proptests for group operations

proptest! {
    #[test]
    fn inverse_composition_is_identity(se3 in arb_se3()) {
        let result = se3.compose(se3.inverse());
        let id: SE3<B, B> = SE3::identity();
        prop_assert!(abs_diff_eq!(result.vec[0], id.vec[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[1], id.vec[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[2], id.vec[2], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.quat.wxyz().0, id.quat.wxyz().0, epsilon = EPS));
    }

    #[test]
    fn then_is_left_to_right_composition(
        q1 in arb_unit_quat(),
        q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64,
        x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64
    ) {
        let se3_ab: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let se3_bc: SE3<B, C> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let v: Vector3<A> = Vector3::new(x, y, z);
        let v1 = se3_ab.then(se3_bc).act(v);
        let v2 = se3_bc.act(se3_ab.act(v));
        prop_assert!(abs_diff_eq!(v1[0], v2[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1[1], v2[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1[2], v2[2], epsilon = EPS));
    }

    #[test]
    fn compose_is_right_to_left_composition(
        q1 in arb_unit_quat(),
        q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64,
        x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64
    ) {
        let se3_ab: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let se3_ca: SE3<C, A> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let v: Vector3<C> = Vector3::new(x, y, z);
        let v1 = se3_ab.compose(se3_ca).act(v);
        let v2 = se3_ab.act(se3_ca.act(v));
        prop_assert!(abs_diff_eq!(v1[0], v2[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1[1], v2[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1[2], v2[2], epsilon = EPS));
    }

    #[test]
    fn exp_log_roundtrip(se3 in arb_se3()) {
        // exp(log(T)) ≈ T
        let tau = se3.log();
        let se3_2 = tau.exp();
        prop_assert!(se3_approx_eq(&se3, &se3_2));
    }

    #[test]
    fn identity_log_is_zero(_dummy in 0..1i32) {
        let id: SE3<A, B> = SE3::identity();
        let tau = id.log();
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(tau.data[i], 0.0, epsilon = EPS));
        }
    }

    #[test]
    fn then_and_compose_are_consistent(
        q1 in arb_unit_quat(),
        q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64,
        x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64
    ) {
        let se3_ab: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let se3_bc: SE3<B, C> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let v: Vector3<A> = Vector3::new(x, y, z);
        let v1 = se3_ab.then(se3_bc).act(v);
        let v2 = se3_bc.compose(se3_ab).act(v);
        prop_assert!(abs_diff_eq!(v1[0], v2[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1[1], v2[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(v1[2], v2[2], epsilon = EPS));
    }

    // rplus / rminus

    #[test]
    fn rplus_rminus_roundtrip(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        // x.rplus(y.rminus(x)) ≈ y
        let x: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let y: SE3<C, B> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let result = x.rplus(y.rminus(x));
        prop_assert!(quat_approx_eq(&result.quat, &y.quat));
        prop_assert!(abs_diff_eq!(result.vec[0], y.vec[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[1], y.vec[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[2], y.vec[2], epsilon = EPS));
    }

    #[test]
    fn rminus_self_is_zero(se3 in arb_se3()) {
        // x.rminus(x) ≈ 0
        let delta = se3.rminus(se3);
        prop_assert!(abs_diff_eq!(delta.norm(), 0.0, epsilon = EPS));
    }

    #[test]
    fn rplus_zero_is_identity(se3 in arb_se3()) {
        // x.rplus(0) ≈ x
        let zero: SE3Tangent<A, A, A> = SE3Tangent::zero();
        let result = se3.rplus(zero);
        prop_assert!(se3_approx_eq(&se3, &result));
    }

    #[test]
    fn rplus_n_steps_reaches_target(
        q in arb_unit_quat(), n in 2..50i32,
        tx in -10.0..10.0f64, ty in -10.0..10.0f64, tz in -10.0..10.0f64
    ) {
        // Split rminus into N equal steps, rplus N times → reach target
        let target: SE3<A, A> = SE3::from_vec_quat([tx, ty, tz], q);
        let id: SE3<A, A> = SE3::identity();
        let delta = target.rminus(id);
        let step = delta * (1.0 / n as f64);
        let result = (0..n).fold(id, |acc, _| acc.rplus(step));
        prop_assert!(quat_approx_eq(&result.quat, &target.quat));
        prop_assert!(abs_diff_eq!(result.vec[0], target.vec[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[1], target.vec[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[2], target.vec[2], epsilon = EPS));
    }

    // lplus / lminus

    #[test]
    fn lplus_lminus_roundtrip(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        // y.lplus(x.lminus(y)) ≈ x
        let x: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let y: SE3<A, C> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let result = y.lplus(x.lminus(y));
        prop_assert!(quat_approx_eq(&result.quat, &x.quat));
        prop_assert!(abs_diff_eq!(result.vec[0], x.vec[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[1], x.vec[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[2], x.vec[2], epsilon = EPS));
    }

    #[test]
    fn lminus_self_is_zero(se3 in arb_se3()) {
        // x.lminus(x) ≈ 0
        let delta = se3.lminus(se3);
        prop_assert!(abs_diff_eq!(delta.norm(), 0.0, epsilon = EPS));
    }

    #[test]
    fn lplus_zero_is_identity(se3 in arb_se3()) {
        // x.lplus(0) ≈ x
        let zero: SE3Tangent<B, B, B> = SE3Tangent::zero();
        let result = se3.lplus(zero);
        prop_assert!(se3_approx_eq(&se3, &result));
    }

    #[test]
    fn lplus_n_steps_reaches_target(
        q in arb_unit_quat(), n in 2..50i32,
        tx in -10.0..10.0f64, ty in -10.0..10.0f64, tz in -10.0..10.0f64
    ) {
        // Split lminus into N equal steps, lplus N times → reach target
        let target: SE3<A, A> = SE3::from_vec_quat([tx, ty, tz], q);
        let id: SE3<A, A> = SE3::identity();
        let delta = target.lminus(id);
        let step = delta * (1.0 / n as f64);
        let result = (0..n).fold(id, |acc, _| acc.lplus(step));
        prop_assert!(quat_approx_eq(&result.quat, &target.quat));
        prop_assert!(abs_diff_eq!(result.vec[0], target.vec[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[1], target.vec[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(result.vec[2], target.vec[2], epsilon = EPS));
    }

    // slerp

    #[test]
    fn slerp_same_pose(se3 in arb_se3(), t in 0.0..1.0f64) {
        // slerp(p, p, t) = p for all t
        let result = SE3::slerp(se3, se3, t);
        prop_assert!(se3_approx_eq(&se3, &result));
    }

    #[test]
    fn slerp_reverse(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64,
        t in 0.0..1.0f64
    ) {
        // slerp(start, end, t) = slerp(end, start, 1 - t)
        let start: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let end: SE3<A, B> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let forward = SE3::slerp(start, end, t);
        let backward = SE3::slerp(end, start, 1.0 - t);
        prop_assert!(se3_approx_eq(&forward, &backward));
    }

    #[test]
    fn slerp_constant_velocity(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64,
        t in 0.0..1.0f64
    ) {
        // The distance covered after fraction t should be exactly t times the total distance
        let start: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let end: SE3<A, B> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let total = start.between(end).log().norm();
        let partial = start.between(SE3::slerp(start, end, t)).log().norm();
        prop_assert!(abs_diff_eq!(partial, t * total, epsilon = EPS));
    }

    #[test]
    fn mul_group_does_compose(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let a: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let b: SE3<C, A> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let via_mul = a * b;
        let via_compose = a.compose(b);
        prop_assert!(quat_approx_eq(&via_mul.quat, &via_compose.quat));
        prop_assert!(abs_diff_eq!(via_mul.vec[0], via_compose.vec[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(via_mul.vec[1], via_compose.vec[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(via_mul.vec[2], via_compose.vec[2], epsilon = EPS));
    }

    // log_with_jac: result matches log
    #[test]
    fn log_with_jac_result(se3 in arb_se3()) {
        let (result, _) = se3.log_with_jac();
        let expected = se3.log();
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(result.data[i], expected.data[i], epsilon = EPS));
        }
    }

    // log_with_jac: numerical verification via right perturbation
    #[test]
    fn log_with_jac_numerical(se3 in arb_se3()) {
        let (_, jac) = se3.log_with_jac();
        let h = 1e-7;
        let eps = 1e-5;
        let result = se3.log();
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let se3_pert: SE3<A, B> = se3.compose(small);
            let result_pert = se3_pert.log();
            for row in 0..6 {
                let numerical = (result_pert.data[row] - result.data[row]) / h;
                prop_assert!(
                    abs_diff_eq!(jac.data[row][i], numerical, epsilon = eps),
                    "log_jac mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac.data[row][i], numerical
                );
            }
        }
    }

    // inverse_with_jac: result matches inverse
    #[test]
    fn inverse_with_jac_result(se3 in arb_se3()) {
        let (result, _) = se3.inverse_with_jac();
        let expected = se3.inverse();
        prop_assert!(quat_approx_eq(&result.quat, &expected.quat));
        for i in 0..3 {
            prop_assert!(abs_diff_eq!(result.vec[i], expected.vec[i], epsilon = EPS));
        }
    }

    // inverse_with_jac: numerical verification via right perturbation
    // log(f(X)⁻¹ · f(X_pert)) = log(X · X_pert⁻¹) ≈ J · δ
    #[test]
    fn inverse_with_jac_numerical(se3 in arb_se3()) {
        let (_, jac) = se3.inverse_with_jac();
        let h = 1e-7;
        let eps = 1e-5;
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let se3_pert: SE3<A, B> = se3.compose(small);
            let diff = se3.compose(se3_pert.inverse()).log();
            for row in 0..6 {
                let numerical = diff.data[row] / h;
                prop_assert!(
                    abs_diff_eq!(jac.data[row][i], numerical, epsilon = eps),
                    "inverse_jac mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac.data[row][i], numerical
                );
            }
        }
    }

    // compose_with_jac: result matches compose
    #[test]
    fn compose_with_jac_result(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let r1: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let r2: SE3<C, A> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let (result, _, _) = r1.compose_with_jac(r2);
        let expected = r1.compose(r2);
        prop_assert!(quat_approx_eq(&result.quat, &expected.quat));
        for i in 0..3 {
            prop_assert!(abs_diff_eq!(result.vec[i], expected.vec[i], epsilon = EPS));
        }
    }

    // compose_with_jac: J_lhs (numerical verification via right perturbation on r1)
    // log(f(X₁,X₂)⁻¹ · f(X₁_pert,X₂)) ≈ J_lhs · δ
    #[test]
    fn compose_with_jac_lhs(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let r1: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let r2: SE3<C, A> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let (_, jac_lhs, _) = r1.compose_with_jac(r2);
        let h = 1e-7;
        let eps = 1e-5;
        let composed = r1.compose(r2);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let r1_pert: SE3<A, B> = r1.compose(small);
            let composed_pert = r1_pert.compose(r2);
            let diff = composed.inverse().compose(composed_pert).log();
            for row in 0..6 {
                let numerical = diff.data[row] / h;
                prop_assert!(
                    abs_diff_eq!(jac_lhs.data[row][i], numerical, epsilon = eps),
                    "compose J_lhs mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_lhs.data[row][i], numerical
                );
            }
        }
    }

    // compose_with_jac: J_rhs (numerical verification via right perturbation on r2)
    // log(f(X₁,X₂)⁻¹ · f(X₁,X₂_pert)) ≈ J_rhs · δ
    #[test]
    fn compose_with_jac_rhs(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let r1: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let r2: SE3<C, A> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let (_, _, jac_rhs) = r1.compose_with_jac(r2);
        let h = 1e-7;
        let eps = 1e-5;
        let composed = r1.compose(r2);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<C, C> = SE3Tangent::<C, C, C>::from_data(delta).exp();
            let r2_pert: SE3<C, A> = r2.compose(small);
            let composed_pert = r1.compose(r2_pert);
            let diff = composed.inverse().compose(composed_pert).log();
            for row in 0..6 {
                let numerical = diff.data[row] / h;
                prop_assert!(
                    abs_diff_eq!(jac_rhs.data[row][i], numerical, epsilon = eps),
                    "compose J_rhs mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_rhs.data[row][i], numerical
                );
            }
        }
    }

    // act_with_jac on Vector3: result matches act
    #[test]
    fn act_with_jac_vector3_result(
        se3 in arb_se3(),
        x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64
    ) {
        let v: Vector3<A> = Vector3::new(x, y, z);
        let (result, _, _) = se3.act_with_jac(v);
        let expected = se3.act(v);
        for i in 0..3 {
            prop_assert!(abs_diff_eq!(result.data[i], expected.data[i], epsilon = EPS));
        }
    }

    // act_with_jac on Vector3: J_input (numerical verification)
    #[test]
    fn act_with_jac_vector3_input(
        se3 in arb_se3(),
        x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64
    ) {
        let v: Vector3<A> = Vector3::new(x, y, z);
        let (_, _, jac_input) = se3.act_with_jac(v);
        let h = 1e-7;
        let eps = 1e-5;
        let result = se3.act(v);
        for i in 0..3 {
            let mut perturbed = v.data;
            perturbed[i] += h;
            let v_plus: Vector3<A> = Vector3::from_data(perturbed);
            let result_plus = se3.act(v_plus);
            for row in 0..3 {
                let numerical = (result_plus.data[row] - result.data[row]) / h;
                prop_assert!(
                    abs_diff_eq!(jac_input.data[row][i], numerical, epsilon = eps),
                    "act J_input mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_input.data[row][i], numerical
                );
            }
        }
    }

    // act_with_jac on Vector3: J_group (numerical verification via right perturbation)
    #[test]
    fn act_with_jac_vector3_group(
        se3 in arb_se3(),
        x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64
    ) {
        let v: Vector3<A> = Vector3::new(x, y, z);
        let (_, jac_group, _) = se3.act_with_jac(v);
        let h = 1e-7;
        let eps = 1e-5;
        let result = se3.act(v);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let se3_pert: SE3<A, B> = se3.compose(small);
            let result_plus = se3_pert.act(v);
            for row in 0..3 {
                let numerical = (result_plus.data[row] - result.data[row]) / h;
                prop_assert!(
                    abs_diff_eq!(jac_group.data[row][i], numerical, epsilon = eps),
                    "act J_group mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_group.data[row][i], numerical
                );
            }
        }
    }

    // act_with_jac on SE3Tangent: result matches act
    #[test]
    fn act_with_jac_tangent_result(
        se3 in arb_se3(),
        v in arb_se3_tangent()
    ) {
        let (result, _, _) = se3.act_with_jac(v);
        let expected = se3.act(v);
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(result.data[i], expected.data[i], epsilon = EPS));
        }
    }

    // act_with_jac on SE3Tangent: J_input (numerical verification)
    #[test]
    fn act_with_jac_tangent_input(
        se3 in arb_se3(),
        v in arb_se3_tangent()
    ) {
        let (_, _, jac_input) = se3.act_with_jac(v);
        let h = 1e-7;
        let eps = 1e-5;
        let result = se3.act(v);
        for i in 0..6 {
            let mut perturbed = v.data;
            perturbed[i] += h;
            let v_plus: SE3Tangent<A, B, A> = SE3Tangent::from_data(perturbed);
            let result_plus = se3.act(v_plus);
            for row in 0..6 {
                let numerical = (result_plus.data[row] - result.data[row]) / h;
                prop_assert!(
                    abs_diff_eq!(jac_input.data[row][i], numerical, epsilon = eps),
                    "act J_input mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_input.data[row][i], numerical
                );
            }
        }
    }

    // act_with_jac on SE3Tangent: J_group (numerical verification via right perturbation)
    #[test]
    fn act_with_jac_tangent_group(
        se3 in arb_se3(),
        v in arb_se3_tangent()
    ) {
        let (_, jac_group, _) = se3.act_with_jac(v);
        let h = 1e-7;
        let eps = 1e-5;
        let result = se3.act(v);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let se3_pert: SE3<A, B> = se3.compose(small);
            let result_plus = se3_pert.act(v);
            for row in 0..6 {
                let numerical = (result_plus.data[row] - result.data[row]) / h;
                prop_assert!(
                    abs_diff_eq!(jac_group.data[row][i], numerical, epsilon = eps),
                    "act J_group mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_group.data[row][i], numerical
                );
            }
        }
    }

    // rplus_with_jac: result matches rplus
    #[test]
    fn rplus_with_jac_result(se3 in arb_se3(), v in arb_se3_tangent()) {
        let v_a: SE3Tangent<C, A, C> = SE3Tangent::from_data(v.data);
        let (result, _, _) = se3.rplus_with_jac(v_a);
        let expected = se3.rplus(v_a);
        prop_assert!(quat_approx_eq(&result.quat, &expected.quat));
        for i in 0..3 {
            prop_assert!(abs_diff_eq!(result.vec[i], expected.vec[i], epsilon = EPS));
        }
    }

    // rplus_with_jac: J_self (numerical verification via right perturbation on se3)
    #[test]
    fn rplus_with_jac_self(se3 in arb_se3(), v in arb_se3_tangent()) {
        let v_a: SE3Tangent<C, A, C> = SE3Tangent::from_data(v.data);
        let (_, jac_self, _) = se3.rplus_with_jac(v_a);
        let h = 1e-7;
        let eps = 1e-5;
        let result = se3.rplus(v_a);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let se3_pert: SE3<A, B> = se3.compose(small);
            let result_pert = se3_pert.rplus(v_a);
            let diff = result.inverse().compose(result_pert).log();
            for row in 0..6 {
                let numerical = diff.data[row] / h;
                prop_assert!(
                    abs_diff_eq!(jac_self.data[row][i], numerical, epsilon = eps),
                    "rplus J_self mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_self.data[row][i], numerical
                );
            }
        }
    }

    // rplus_with_jac: J_rhs (numerical verification via perturbation on tangent)
    #[test]
    fn rplus_with_jac_rhs(se3 in arb_se3(), v in arb_se3_tangent()) {
        let v_a: SE3Tangent<C, A, C> = SE3Tangent::from_data(v.data);
        let (_, _, jac_rhs) = se3.rplus_with_jac(v_a);
        let h = 1e-7;
        let eps = 1e-5;
        let result = se3.rplus(v_a);
        for i in 0..6 {
            let mut perturbed = v_a.data;
            perturbed[i] += h;
            let v_pert: SE3Tangent<C, A, C> = SE3Tangent::from_data(perturbed);
            let result_pert = se3.rplus(v_pert);
            let diff = result.inverse().compose(result_pert).log();
            for row in 0..6 {
                let numerical = diff.data[row] / h;
                prop_assert!(
                    abs_diff_eq!(jac_rhs.data[row][i], numerical, epsilon = eps),
                    "rplus J_rhs mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_rhs.data[row][i], numerical
                );
            }
        }
    }

    // rminus_with_jac: result matches rminus
    #[test]
    fn rminus_with_jac_result(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let r1: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let r2: SE3<C, B> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let (result, _, _) = r1.rminus_with_jac(r2);
        let expected = r1.rminus(r2);
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(result.data[i], expected.data[i], epsilon = EPS));
        }
    }

    // rminus_with_jac: J_self (numerical verification via right perturbation on r1)
    #[test]
    fn rminus_with_jac_self(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let r1: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let r2: SE3<C, B> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let (_, jac_self, _) = r1.rminus_with_jac(r2);
        let h = 1e-7;
        let eps = 5e-5;
        let result = r1.rminus(r2);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let r1_pert: SE3<A, B> = r1.compose(small);
            let result_pert = r1_pert.rminus(r2);
            for row in 0..6 {
                let numerical = (result_pert.data[row] - result.data[row]) / h;
                prop_assert!(
                    abs_diff_eq!(jac_self.data[row][i], numerical, epsilon = eps),
                    "rminus J_self mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_self.data[row][i], numerical
                );
            }
        }
    }

    // rminus_with_jac: J_rhs (numerical verification via right perturbation on r2)
    #[test]
    fn rminus_with_jac_rhs(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let r1: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let r2: SE3<C, B> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let (_, _, jac_rhs) = r1.rminus_with_jac(r2);
        let h = 1e-7;
        let eps = 5e-5;
        let result = r1.rminus(r2);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<C, C> = SE3Tangent::<C, C, C>::from_data(delta).exp();
            let r2_pert: SE3<C, B> = r2.compose(small);
            let result_pert = r1.rminus(r2_pert);
            for row in 0..6 {
                let numerical = (result_pert.data[row] - result.data[row]) / h;
                prop_assert!(
                    abs_diff_eq!(jac_rhs.data[row][i], numerical, epsilon = eps),
                    "rminus J_rhs mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_rhs.data[row][i], numerical
                );
            }
        }
    }

    // lplus_with_jac: result matches lplus
    #[test]
    fn lplus_with_jac_result(se3 in arb_se3(), v in arb_se3_tangent()) {
        let v_b: SE3Tangent<B, C, B> = SE3Tangent::from_data(v.data);
        let (result, _, _) = se3.lplus_with_jac(v_b);
        let expected = se3.lplus(v_b);
        prop_assert!(quat_approx_eq(&result.quat, &expected.quat));
        for i in 0..3 {
            prop_assert!(abs_diff_eq!(result.vec[i], expected.vec[i], epsilon = EPS));
        }
    }

    // lplus_with_jac: J_self (numerical verification via right perturbation on se3)
    #[test]
    fn lplus_with_jac_self(se3 in arb_se3(), v in arb_se3_tangent()) {
        let v_b: SE3Tangent<B, C, B> = SE3Tangent::from_data(v.data);
        let (_, jac_self, _) = se3.lplus_with_jac(v_b);
        let h = 1e-7;
        let eps = 1e-5;
        let result = se3.lplus(v_b);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let se3_pert: SE3<A, B> = se3.compose(small);
            let result_pert = se3_pert.lplus(v_b);
            let diff = result.inverse().compose(result_pert).log();
            for row in 0..6 {
                let numerical = diff.data[row] / h;
                prop_assert!(
                    abs_diff_eq!(jac_self.data[row][i], numerical, epsilon = eps),
                    "lplus J_self mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_self.data[row][i], numerical
                );
            }
        }
    }

    // lplus_with_jac: J_lhs (numerical verification via perturbation on tangent)
    #[test]
    fn lplus_with_jac_lhs(se3 in arb_se3(), v in arb_se3_tangent()) {
        let v_b: SE3Tangent<B, C, B> = SE3Tangent::from_data(v.data);
        let (_, _, jac_lhs) = se3.lplus_with_jac(v_b);
        let h = 1e-7;
        let eps = 1e-5;
        let result = se3.lplus(v_b);
        for i in 0..6 {
            let mut perturbed = v_b.data;
            perturbed[i] += h;
            let v_pert: SE3Tangent<B, C, B> = SE3Tangent::from_data(perturbed);
            let result_pert = se3.lplus(v_pert);
            let diff = result.inverse().compose(result_pert).log();
            for row in 0..6 {
                let numerical = diff.data[row] / h;
                prop_assert!(
                    abs_diff_eq!(jac_lhs.data[row][i], numerical, epsilon = eps),
                    "lplus J_lhs mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_lhs.data[row][i], numerical
                );
            }
        }
    }

    // lminus_with_jac: result matches lminus
    #[test]
    fn lminus_with_jac_result(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let r1: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let r2: SE3<A, C> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let (result, _, _) = r1.lminus_with_jac(r2);
        let expected = r1.lminus(r2);
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(result.data[i], expected.data[i], epsilon = EPS));
        }
    }

    // lminus_with_jac: J_self (numerical verification via right perturbation on r1)
    #[test]
    fn lminus_with_jac_self(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let r1: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let r2: SE3<A, C> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let (_, jac_self, _) = r1.lminus_with_jac(r2);
        let h = 1e-7;
        let eps = 5e-5;
        let result = r1.lminus(r2);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let r1_pert: SE3<A, B> = r1.compose(small);
            let result_pert = r1_pert.lminus(r2);
            for row in 0..6 {
                let numerical = (result_pert.data[row] - result.data[row]) / h;
                prop_assert!(
                    abs_diff_eq!(jac_self.data[row][i], numerical, epsilon = eps),
                    "lminus J_self mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_self.data[row][i], numerical
                );
            }
        }
    }

    // lminus_with_jac: J_rhs (numerical verification via right perturbation on r2)
    #[test]
    fn lminus_with_jac_rhs(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let r1: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let r2: SE3<A, C> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let (_, _, jac_rhs) = r1.lminus_with_jac(r2);
        let h = 1e-7;
        let eps = 5e-5;
        let result = r1.lminus(r2);
        for i in 0..6 {
            let mut delta = [0.0; 6];
            delta[i] = h;
            let small: SE3<A, A> = SE3Tangent::<A, A, A>::from_data(delta).exp();
            let r2_pert: SE3<A, C> = r2.compose(small);
            let result_pert = r1.lminus(r2_pert);
            for row in 0..6 {
                let numerical = (result_pert.data[row] - result.data[row]) / h;
                prop_assert!(
                    abs_diff_eq!(jac_rhs.data[row][i], numerical, epsilon = eps),
                    "lminus J_rhs mismatch at ({}, {}): analytic={}, numerical={}",
                    row, i, jac_rhs.data[row][i], numerical
                );
            }
        }
    }

    // adjoint

    // Ad_X(v) = log(X_rhs * exp(v) * X_lhs^{-1})
    #[test]
    fn adjoint_is_conjugation(
        q in arb_unit_quat(),
        tx in -10.0..10.0f64, ty in -10.0..10.0f64, tz in -10.0..10.0f64,
        v in arb_se3_tangent()
    ) {
        let x_lhs: SE3<A, B> = SE3::from_vec_quat([tx, ty, tz], q);
        let x_rhs: SE3<B, E> = SE3::from_vec_quat([tx, ty, tz], q);
        let adj_v = x_rhs.adjoint(v);
        let conjugated = x_rhs.compose(v.exp()).compose(x_lhs.inverse()).log();
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(adj_v.data[i], conjugated.data[i], epsilon = EPS));
        }
    }

    // identity adjoint is a no-op
    #[test]
    fn adjoint_identity_is_noop(v in arb_se3_tangent()) {
        let id: SE3<B, E> = SE3::identity();
        let adj_v = id.adjoint(v);
        for i in 0..6 {
            prop_assert!(abs_diff_eq!(adj_v.data[i], v.data[i], epsilon = EPS));
        }
    }

    // adjoint_matrix at identity is I₆
    #[test]
    fn adjoint_matrix_identity_is_identity(_dummy in 0..1i32) {
        let id: SE3<A, B> = SE3::identity();
        let ad = id.adjoint_matrix();
        prop_assert!(mat6_approx_eq(&ad, &Mat6::identity()));
    }

    // to_matrix

    // M * [v; 1] acts like SE3::act(v)
    #[test]
    fn to_matrix_acts_like_se3(
        se3 in arb_se3(),
        x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64
    ) {
        let m = se3.to_matrix();
        let v: Vector3<A> = Vector3::new(x, y, z);
        let via_act = se3.act(v);
        let homogeneous = m * [x, y, z, 1.0];
        prop_assert!(abs_diff_eq!(homogeneous[0], via_act[0], epsilon = EPS));
        prop_assert!(abs_diff_eq!(homogeneous[1], via_act[1], epsilon = EPS));
        prop_assert!(abs_diff_eq!(homogeneous[2], via_act[2], epsilon = EPS));
        prop_assert!(abs_diff_eq!(homogeneous[3], 1.0, epsilon = EPS));
    }

    // to_matrix of identity is I₄
    #[test]
    fn to_matrix_identity_is_identity(_dummy in 0..1i32) {
        let id: SE3<A, B> = SE3::identity();
        let m = id.to_matrix();
        let i4: Mat4 = Mat4::identity();
        for i in 0..4 {
            for j in 0..4 {
                prop_assert!(abs_diff_eq!(m.data[i][j], i4.data[i][j], epsilon = EPS));
            }
        }
    }

    // to_matrix of compose equals matrix product
    #[test]
    fn to_matrix_compose_is_product(
        q1 in arb_unit_quat(), q2 in arb_unit_quat(),
        t1x in -10.0..10.0f64, t1y in -10.0..10.0f64, t1z in -10.0..10.0f64,
        t2x in -10.0..10.0f64, t2y in -10.0..10.0f64, t2z in -10.0..10.0f64
    ) {
        let a: SE3<A, B> = SE3::from_vec_quat([t1x, t1y, t1z], q1);
        let b: SE3<C, A> = SE3::from_vec_quat([t2x, t2y, t2z], q2);
        let via_compose = a.compose(b).to_matrix();
        let via_product = a.to_matrix() * b.to_matrix();
        for i in 0..4 {
            for j in 0..4 {
                prop_assert!(abs_diff_eq!(via_compose.data[i][j], via_product.data[i][j], epsilon = EPS));
            }
        }
    }
}

// Deterministic inverse tests

#[test]
fn inverse_of_identity_is_identity() {
    let id: SE3<A, B> = SE3::identity();
    let inv = id.inverse();
    assert_eq!(inv, SE3::<B, A>::identity());
}

#[test]
fn inverse_of_pure_translation() {
    let se3: SE3<A, B> = SE3::from_vec([1.0, 2.0, 3.0]);
    let inv = se3.inverse();
    assert!(abs_diff_eq!(inv.vec[0], -1.0, epsilon = EPS));
    assert!(abs_diff_eq!(inv.vec[1], -2.0, epsilon = EPS));
    assert!(abs_diff_eq!(inv.vec[2], -3.0, epsilon = EPS));
}

#[test]
fn inverse_of_pure_rotation() {
    let se3: SE3<A, B> = SE3::rot_z(FRAC_PI_2);
    let inv = se3.inverse();
    let expected: SE3<B, A> = SE3::rot_z(-FRAC_PI_2);
    assert!(abs_diff_eq!(inv.quat.wxyz().0, expected.quat.wxyz().0, epsilon = EPS));
    assert!(abs_diff_eq!(inv.quat.wxyz().1, expected.quat.wxyz().1, epsilon = EPS));
    assert!(abs_diff_eq!(inv.quat.wxyz().2, expected.quat.wxyz().2, epsilon = EPS));
    assert!(abs_diff_eq!(inv.quat.wxyz().3, expected.quat.wxyz().3, epsilon = EPS));
    assert_eq!(inv.vec, [0.0, 0.0, 0.0]);
}

// Deterministic compose/then tests

#[test]
fn compose_pure_translations_adds() {
    let t1: SE3<A, B> = SE3::from_vec([1.0, 2.0, 3.0]);
    let t2: SE3<C, A> = SE3::from_vec([4.0, 5.0, 6.0]);
    let result = t1.compose(t2);
    assert!(abs_diff_eq!(result.vec[0], 5.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.vec[1], 7.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.vec[2], 9.0, epsilon = EPS));
}

#[test]
fn compose_pure_rotations_matches_so3() {
    let r1: SE3<A, B> = SE3::rot_z(FRAC_PI_2);
    let r2: SE3<C, A> = SE3::rot_x(FRAC_PI_2);
    let result = r1.compose(r2);
    let so3_r1: SO3<A, B> = SO3::rot_z(FRAC_PI_2);
    let so3_r2: SO3<C, A> = SO3::rot_x(FRAC_PI_2);
    let so3_result = so3_r1.compose(so3_r2);
    assert_eq!(result.quat, so3_result.quat);
    assert_eq!(result.vec, [0.0, 0.0, 0.0]);
}

// Act on Vector3

#[test]
fn act_identity_on_vector_is_noop() {
    let se3: SE3<A, B> = SE3::identity();
    let v: Vector3<A> = Vector3::new(1.0, 2.0, 3.0);
    let result = se3.act(v);
    assert!(abs_diff_eq!(result[0], 1.0, epsilon = EPS));
    assert!(abs_diff_eq!(result[1], 2.0, epsilon = EPS));
    assert!(abs_diff_eq!(result[2], 3.0, epsilon = EPS));
}

#[test]
fn act_pure_translation_on_vector() {
    let se3: SE3<A, B> = SE3::from_vec([10.0, 20.0, 30.0]);
    let v: Vector3<A> = Vector3::new(1.0, 2.0, 3.0);
    let result = se3.act(v);
    assert!(abs_diff_eq!(result[0], 11.0, epsilon = EPS));
    assert!(abs_diff_eq!(result[1], 22.0, epsilon = EPS));
    assert!(abs_diff_eq!(result[2], 33.0, epsilon = EPS));
}

#[test]
fn act_pure_rotation_on_vector() {
    // rot_z(pi/2) maps (1,0,0) -> (0,1,0)
    let se3: SE3<A, B> = SE3::rot_z(FRAC_PI_2);
    let v: Vector3<A> = Vector3::new(1.0, 0.0, 0.0);
    let result = se3.act(v);
    assert!(abs_diff_eq!(result[0], 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result[1], 1.0, epsilon = EPS));
    assert!(abs_diff_eq!(result[2], 0.0, epsilon = EPS));
}

#[test]
fn act_rotation_then_translation_on_vector() {
    // SE3 acts as: t + R*v
    // rot_z(pi/2) maps (1,0,0) -> (0,1,0), then translate by (10,0,0) -> (10,1,0)
    let se3: SE3<A, B> = SE3::from_vec_quat([10.0, 0.0, 0.0], SE3::<A, B>::rot_z(FRAC_PI_2).quat);
    let v: Vector3<A> = Vector3::new(1.0, 0.0, 0.0);
    let result = se3.act(v);
    assert!(abs_diff_eq!(result[0], 10.0, epsilon = EPS));
    assert!(abs_diff_eq!(result[1], 1.0, epsilon = EPS));
    assert!(abs_diff_eq!(result[2], 0.0, epsilon = EPS));
}

// Act on SE3Tangent

#[test]
fn act_identity_on_tangent_is_noop() {
    let se3: SE3<A, B> = SE3::identity();
    let v: SE3Tangent<A, A, A> = SE3Tangent::from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = se3.act(v);
    assert_eq!(result.lin(), [1.0, 2.0, 3.0]);
    assert_eq!(result.ang(), [4.0, 5.0, 6.0]);
}

#[test]
fn act_rotates_both_lin_and_ang() {
    // rot_z(pi/2): x->y, y->-x
    let se3: SE3<A, B> = SE3::rot_z(FRAC_PI_2);
    let v: SE3Tangent<A, A, A> = SE3Tangent::from_lin_ang([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let result = se3.act(v);
    assert!(abs_diff_eq!(result.lin()[0], 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.lin()[1], 1.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.lin()[2], 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.ang()[0], 0.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.ang()[1], 1.0, epsilon = EPS));
    assert!(abs_diff_eq!(result.ang()[2], 0.0, epsilon = EPS));
}

#[test]
fn act_on_tangent_ignores_translation() {
    // Pure translation should not affect tangent re-expression
    let se3: SE3<A, B> = SE3::from_vec([100.0, 200.0, 300.0]);
    let v: SE3Tangent<A, A, A> = SE3Tangent::from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = se3.act(v);
    assert_eq!(result.lin(), [1.0, 2.0, 3.0]);
    assert_eq!(result.ang(), [4.0, 5.0, 6.0]);
}

// Slerp boundary conditions

#[test]
fn slerp_t0_is_start() {
    let start: SE3<A, B> = SE3::from_vec_quat([1.0, 2.0, 3.0], SE3::<A, B>::rot_x(0.5).quat);
    let end: SE3<A, B> = SE3::from_vec_quat([4.0, 5.0, 6.0], SE3::<A, B>::rot_y(1.0).quat);
    let result = SE3::slerp(start, end, 0.0);
    assert!(se3_approx_eq(&result, &start));
}

#[test]
fn slerp_t1_is_end() {
    let start: SE3<A, B> = SE3::from_vec_quat([1.0, 2.0, 3.0], SE3::<A, B>::rot_x(0.5).quat);
    let end: SE3<A, B> = SE3::from_vec_quat([4.0, 5.0, 6.0], SE3::<A, B>::rot_y(1.0).quat);
    let result = SE3::slerp(start, end, 1.0);
    assert!(se3_approx_eq(&result, &end));
}

// act() and * are equivalent for SE3 composition, Vector3, and SE3Tangent
#[test]
fn act_and_mul_are_equivalent() {
    let se3: SE3<A, B> = SE3::from_vec_quat([1.0, 2.0, 3.0], SE3::<A, B>::rot_z(FRAC_PI_2).quat);
    let vec: Vector3<A> = Vector3::new(1.0, 2.0, 3.0);
    let tangent: SE3Tangent<C, E, A> = SE3Tangent::from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let vec_act = se3.act(vec);
    let vec_mul = se3 * vec;
    assert_eq!(vec_act, vec_mul);

    let tan_act = se3.act(tangent);
    let tan_mul = se3 * tangent;
    assert_eq!(tan_act, tan_mul);
}
