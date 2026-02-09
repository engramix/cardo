use approx::abs_diff_eq;
use cardo::primitives::Quat;

const EPS: f64 = 1e-10;

#[test]
fn identity() {
    let q = Quat::<f64>::identity();
    let (w, x, y, z) = q.wxyz();
    assert_eq!((w, x, y, z), (1.0, 0.0, 0.0, 0.0));
}

#[test]
fn conjugate() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0).normalized();
    let c = q.conjugate();
    let (w, x, y, z) = q.wxyz();
    let (cw, cx, cy, cz) = c.wxyz();
    assert_eq!(w, cw);
    assert_eq!(x, -cx);
    assert_eq!(y, -cy);
    assert_eq!(z, -cz);
}

#[test]
fn mul_identity() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0).normalized();
    let id = Quat::identity();
    let r = q * id;
    assert!(abs_diff_eq!(q.w, r.w, epsilon = EPS));
    assert!(abs_diff_eq!(q.x, r.x, epsilon = EPS));
    assert!(abs_diff_eq!(q.y, r.y, epsilon = EPS));
    assert!(abs_diff_eq!(q.z, r.z, epsilon = EPS));
}

#[test]
fn norm() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    let n = q.norm();
    assert!(abs_diff_eq!(n, (30.0_f64).sqrt(), epsilon = EPS));
}

#[test]
fn norm_squared() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    assert!(abs_diff_eq!(q.norm_squared(), 30.0, epsilon = EPS));
}

#[test]
fn normalized_has_unit_norm() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0).normalized();
    assert!(abs_diff_eq!(q.norm(), 1.0, epsilon = EPS));
}

#[test]
fn debug() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    let s = format!("{:?}", q);
    assert!(s.contains("Quat"));
}

#[test]
fn clone() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    let q2 = q.clone();
    assert_eq!(q.w, q2.w);
    assert_eq!(q.x, q2.x);
    assert_eq!(q.y, q2.y);
    assert_eq!(q.z, q2.z);
}

#[test]
fn copy() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    let q2 = q; // copy
    assert_eq!(q.w, q2.w); // q still usable
}

#[test]
fn eq() {
    let a = Quat::new(1.0, 2.0, 3.0, 4.0);
    let b = Quat::new(1.0, 2.0, 3.0, 4.0);
    let c = Quat::new(1.0, 2.0, 3.0, 5.0);
    assert_eq!(a, b);
    assert_ne!(a, c);
}
