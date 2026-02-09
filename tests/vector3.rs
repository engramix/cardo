use cardo::prelude::*;

struct F;

#[test]
fn add() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let b: Vector3<F> = Vector3::new(4.0, 5.0, 6.0);
    let c = a + b;
    assert_eq!(c.xyz(), &[5.0, 7.0, 9.0]);
}

#[test]
fn sub() {
    let a: Vector3<F> = Vector3::new(4.0, 5.0, 6.0);
    let b: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let c = a - b;
    assert_eq!(c.xyz(), &[3.0, 3.0, 3.0]);
}

#[test]
fn neg() {
    let a: Vector3<F> = Vector3::new(1.0, -2.0, 3.0);
    let b = -a;
    assert_eq!(b.xyz(), &[-1.0, 2.0, -3.0]);
}

#[test]
fn mul_scalar_right() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let b = a * 2.0;
    assert_eq!(b.xyz(), &[2.0, 4.0, 6.0]);
}

#[test]
fn mul_scalar_left() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let b = 2.0 * a;
    assert_eq!(b.xyz(), &[2.0, 4.0, 6.0]);
}

#[test]
fn mul_elementwise() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let b: Vector3<F> = Vector3::new(4.0, 5.0, 6.0);
    let c = a * b;
    assert_eq!(c.xyz(), &[4.0, 10.0, 18.0]);
}

#[test]
fn dot() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let b: Vector3<F> = Vector3::new(4.0, 5.0, 6.0);
    assert_eq!(a.dot(&b), 32.0);
}

#[test]
fn cross() {
    let a: Vector3<F> = Vector3::new(1.0, 0.0, 0.0);
    let b: Vector3<F> = Vector3::new(0.0, 1.0, 0.0);
    let c = a.cross(&b);
    assert_eq!(c.xyz(), &[0.0, 0.0, 1.0]);
}

#[test]
fn norm() {
    let a: Vector3<F> = Vector3::new(3.0, 4.0, 0.0);
    assert_eq!(a.norm(), 5.0);
}

#[test]
fn norm_squared() {
    let a: Vector3<F> = Vector3::new(3.0, 4.0, 0.0);
    assert_eq!(a.norm_squared(), 25.0);
}

#[test]
fn normalized() {
    let a: Vector3<F> = Vector3::new(3.0, 4.0, 0.0);
    let n = a.normalized();
    assert_eq!(n.xyz(), &[0.6, 0.8, 0.0]);
}

#[test]
fn zero() {
    let z: Vector3<F> = Vector3::zero();
    assert_eq!(z.xyz(), &[0.0, 0.0, 0.0]);
}

#[test]
fn index() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    assert_eq!(a[0], 1.0);
    assert_eq!(a[1], 2.0);
    assert_eq!(a[2], 3.0);
}

#[test]
fn index_mut() {
    let mut a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    a[1] = 5.0;
    assert_eq!(a.xyz(), &[1.0, 5.0, 3.0]);
}

#[test]
fn coeffs() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    assert_eq!(a.coeffs(), &[1.0, 2.0, 3.0]);
}

#[test]
fn x_y_z() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    assert_eq!(a.x(), 1.0);
    assert_eq!(a.y(), 2.0);
    assert_eq!(a.z(), 3.0);
}

#[test]
fn debug() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let s = format!("{:?}", a);
    assert!(s.contains("Vector3"));
}

#[test]
fn clone() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let b = a.clone();
    assert_eq!(a.xyz(), b.xyz());
}

#[test]
fn copy() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let b = a; // copy
    assert_eq!(a.xyz(), b.xyz()); // a still usable
}

#[test]
fn eq() {
    let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let b: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
    let c: Vector3<F> = Vector3::new(1.0, 2.0, 4.0);
    assert_eq!(a, b);
    assert_ne!(a, c);
}
