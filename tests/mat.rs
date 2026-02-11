use cardo::prelude::*;

#[test]
fn zeros() {
    let m: Mat3 = Mat::zeros();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(m[i][j], 0.0);
        }
    }
}

#[test]
fn identity() {
    let m: Mat3 = Mat::identity();
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert_eq!(m[i][j], 1.0);
            } else {
                assert_eq!(m[i][j], 0.0);
            }
        }
    }
}

#[test]
fn from_data() {
    let m: Mat<2, 3> = Mat::from_data([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);
    assert_eq!(m[0][0], 1.0);
    assert_eq!(m[0][2], 3.0);
    assert_eq!(m[1][1], 5.0);
}

#[test]
fn transpose() {
    let m: Mat<2, 3> = Mat::from_data([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);
    let mt: Mat<3, 2> = m.transpose();
    assert_eq!(mt[0][0], 1.0);
    assert_eq!(mt[0][1], 4.0);
    assert_eq!(mt[1][0], 2.0);
    assert_eq!(mt[2][1], 6.0);
}

#[test]
fn mat_mat_mul() {
    let a: Mat<2, 3> = Mat::from_data([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);
    let b: Mat<3, 2> = Mat::from_data([
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ]);
    let c: Mat<2, 2> = a * b;
    assert_eq!(c[0][0], 58.0);
    assert_eq!(c[0][1], 64.0);
    assert_eq!(c[1][0], 139.0);
    assert_eq!(c[1][1], 154.0);
}

#[test]
fn mat_vec_mul() {
    let m: Mat<2, 3> = Mat::from_data([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);
    let v = [1.0, 2.0, 3.0];
    let result: [f64; 2] = m * v;
    assert_eq!(result[0], 14.0);
    assert_eq!(result[1], 32.0);
}

#[test]
fn identity_is_multiplicative_identity() {
    let m: Mat3 = Mat::from_data([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);
    let id: Mat3 = Mat::identity();
    assert_eq!(m * id, m);
    assert_eq!(id * m, m);
}

#[test]
fn add() {
    let a: Mat<2, 2> = Mat::from_data([[1.0, 2.0], [3.0, 4.0]]);
    let b: Mat<2, 2> = Mat::from_data([[5.0, 6.0], [7.0, 8.0]]);
    let c = a + b;
    assert_eq!(c, Mat::from_data([[6.0, 8.0], [10.0, 12.0]]));
}

#[test]
fn sub() {
    let a: Mat<2, 2> = Mat::from_data([[5.0, 6.0], [7.0, 8.0]]);
    let b: Mat<2, 2> = Mat::from_data([[1.0, 2.0], [3.0, 4.0]]);
    let c = a - b;
    assert_eq!(c, Mat::from_data([[4.0, 4.0], [4.0, 4.0]]));
}

#[test]
fn neg() {
    let m: Mat<2, 2> = Mat::from_data([[1.0, -2.0], [-3.0, 4.0]]);
    assert_eq!(-m, Mat::from_data([[-1.0, 2.0], [3.0, -4.0]]));
}

#[test]
fn scalar_mul_right() {
    let m: Mat<2, 2> = Mat::from_data([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(m * 2.0, Mat::from_data([[2.0, 4.0], [6.0, 8.0]]));
}

#[test]
fn scalar_mul_left() {
    let m: Mat<2, 2> = Mat::from_data([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(2.0 * m, Mat::from_data([[2.0, 4.0], [6.0, 8.0]]));
}

#[test]
fn index_mut() {
    let mut m: Mat3 = Mat::zeros();
    m[1][2] = 42.0;
    assert_eq!(m[1][2], 42.0);
}

#[test]
fn debug() {
    let m: Mat<2, 2> = Mat::from_data([[1.0, 2.0], [3.0, 4.0]]);
    let s = format!("{:?}", m);
    assert!(s.contains("Mat"));
}

#[test]
fn clone_and_copy() {
    let m: Mat3 = Mat::identity();
    let m2 = m;
    let m3 = m.clone();
    assert_eq!(m, m2);
    assert_eq!(m, m3);
}

#[test]
fn eq() {
    let a: Mat<2, 2> = Mat::from_data([[1.0, 2.0], [3.0, 4.0]]);
    let b: Mat<2, 2> = Mat::from_data([[1.0, 2.0], [3.0, 4.0]]);
    let c: Mat<2, 2> = Mat::from_data([[1.0, 2.0], [3.0, 5.0]]);
    assert_eq!(a, b);
    assert_ne!(a, c);
}
