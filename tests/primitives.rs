use cardo::primitives::Quat;

mod vec3 {
    // We test Vec3 through the public Vector3 API since Vec3 is pub(crate)
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
    fn mul_elementwise() {
        let a: Vector3<F> = Vector3::new(1.0, 2.0, 3.0);
        let b: Vector3<F> = Vector3::new(4.0, 5.0, 6.0);
        let c = a * b;
        assert_eq!(c.xyz(), &[4.0, 10.0, 18.0]);
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
}

mod quat {
    use super::*;
    use approx::abs_diff_eq;

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
    fn normalized_has_unit_norm() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0).normalized();
        assert!(abs_diff_eq!(q.norm(), 1.0, epsilon = EPS));
    }
}
