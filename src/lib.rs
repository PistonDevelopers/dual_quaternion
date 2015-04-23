//! A simple and type agnostic dual-quaternion math library designed for reexporting

extern crate vecmath;
extern crate quaternion;

use quaternion::Quaternion;
use vecmath::Vector3;
use vecmath::traits::Float;

/// A dual-quaternion consists of a real component and a dual component,
/// and can be used to represent both rotation and translation
pub type DualQuaternion<T> = (Quaternion<T>, Quaternion<T>);

/// Constructs identity dual-quaternion, representing no rotation or translation.
#[inline(always)]
pub fn id<T: Float>() -> DualQuaternion<T> {
    let one = T::one();
    let zero = T::zero();
    (
        (one, [zero, zero, zero]),
        (zero, [zero, zero, zero])
    )
}

/// Construct a dual-quaternion from separate rotation and translation components
#[inline(always)]
pub fn from_rotation_and_translation<T: Float>(rotation: Quaternion<T>, translation: Vector3<T>) -> DualQuaternion<T> {
    let zero = T::zero();
    let half = T::from_f64(0.5);
    (
        rotation,
        quaternion::scale(
            quaternion::mul((zero, translation), rotation),
            half
        )
    )
}

/// Multiplies two dual-quaternions
#[inline(always)]
pub fn mul<T: Float>(
    a: DualQuaternion<T>,
    b: DualQuaternion<T>
) -> DualQuaternion<T> {
    (
        quaternion::mul(b.0, a.0),
        quaternion::add(
            quaternion::mul(b.1, a.0),
            quaternion::mul(b.0, a.1)
        )
    )
}

/// Extracts rotation component from a dual-quaternion
pub fn get_rotation<T: Float>(q: DualQuaternion<T>) -> Quaternion<T> {
    q.0
}

/// Extracts translation component from a dual-quaternion
pub fn get_translation<T: Float>(q: DualQuaternion<T>) -> Vector3<T> {
    let two = T::from_f64(2.0);
    let t = quaternion::mul(
        quaternion::scale(q.1, two),
        quaternion::conj(q.0)
    );
    t.1
}

/// Tests
#[cfg(test)]
mod test {

    use std::f32::consts::PI;
    use quaternion;
    use vecmath::Vector3;

    const EPSILON: f32 = 0.000001;

    #[test]
    fn test_construction_and_extraction() {
        let r = quaternion::euler_angles(PI, PI, PI);
        let t = [1.0, 2.0, 3.0];

        let dq = super::from_rotation_and_translation(r, t);

        let r_prime = super::get_rotation(dq);
        let t_prime = super::get_translation(dq);

        assert!((t_prime[0] - t[0]).abs() < EPSILON);
        assert!((t_prime[1] - t[1]).abs() < EPSILON);
        assert!((t_prime[2] - t[2]).abs() < EPSILON);

        assert!((r_prime.0 - r.0).abs() < EPSILON);
        assert!((r_prime.1[0] - r.1[0]).abs() < EPSILON);
        assert!((r_prime.1[1] - r.1[1]).abs() < EPSILON);
        assert!((r_prime.1[2] - r.1[2]).abs() < EPSILON);
    }

    #[test]
    fn test_mul_rotations() {

        let r1 = quaternion::euler_angles(PI / 2.0, PI, PI);
        let r2 = quaternion::euler_angles(PI / 2.0, -PI, 0.0);

        let t1 = [0.0, 0.0, 0.0];
        let t2 = [0.0, 0.0, 0.0];

        let dq1 = super::from_rotation_and_translation(r1, t1);
        let dq2 = super::from_rotation_and_translation(r2, t2);
        let dq3 = super::mul(dq1, dq2);

        let r_prime = super::get_rotation(dq3);
        let t_prime = super::get_translation(dq3);

        let r_expected = quaternion::euler_angles(PI, 0.0, PI);
        let t_expected = [0.0, 0.0, 0.0];

        assert!((t_prime[0] - t_expected[0]).abs() < EPSILON);
        assert!((t_prime[1] - t_expected[1]).abs() < EPSILON);
        assert!((t_prime[2] - t_expected[2]).abs() < EPSILON);

        let rotate_test_1 = quaternion::rotate_vector(r_prime, [1.0, 1.0, 1.0]);
        let rotate_test_2 = quaternion::rotate_vector(r_expected, [1.0, 1.0, 1.0]);

        assert!((rotate_test_1[0] - rotate_test_2[0]).abs() < EPSILON);
        assert!((rotate_test_1[1] - rotate_test_2[1]).abs() < EPSILON);
        assert!((rotate_test_1[2] - rotate_test_2[2]).abs() < EPSILON);

    }

    #[test]
    fn test_mul_translations() {

        let t1: Vector3<f32> = [1.0, 2.0, 3.0];
        let t2: Vector3<f32> = [1.0, -2.0, 0.0];

        let dq1 = super::from_rotation_and_translation(quaternion::id(), t1);
        let dq2 = super::from_rotation_and_translation(quaternion::id(), t2);
        let dq3 = super::mul(dq1, dq2);

        let r_prime = super::get_rotation(dq3);
        let t_prime = super::get_translation(dq3);

        let r_expected = quaternion::id();
        let t_expected = [2.0, 0.0, 3.0];

        assert!((t_prime[0] - t_expected[0]).abs() < EPSILON);
        assert!((t_prime[1] - t_expected[1]).abs() < EPSILON);
        assert!((t_prime[2] - t_expected[2]).abs() < EPSILON);

        let rotate_test_1 = quaternion::rotate_vector(r_prime, [1.0, 1.0, 1.0]);
        let rotate_test_2 = quaternion::rotate_vector(r_expected, [1.0, 1.0, 1.0]);

        assert!((rotate_test_1[0] - rotate_test_2[0]).abs() < EPSILON);
        assert!((rotate_test_1[1] - rotate_test_2[1]).abs() < EPSILON);
        assert!((rotate_test_1[2] - rotate_test_2[2]).abs() < EPSILON);

    }

}
