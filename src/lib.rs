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

/// Adds two dual-quaternions
#[inline(always)]
pub fn add<T: Float>(a: DualQuaternion<T>, b: DualQuaternion<T>) -> DualQuaternion<T> {
    (
        quaternion::add(a.0, b.0),
        quaternion::add(a.1, b.1)
    )
}

/// Multiplies two dual-quaternions
#[inline(always)]
pub fn mul<T: Float>(a: DualQuaternion<T>, b: DualQuaternion<T>) -> DualQuaternion<T> {
    (
        quaternion::mul(a.0, b.0),
        quaternion::add(
            quaternion::mul(a.0, b.1),
            quaternion::mul(a.1, b.0)
        )
    )
}

/// Scales a dual-quaternion (element-wise) by a scalar
#[inline(always)]
pub fn scale<T: Float>(q: DualQuaternion<T>, t: T) -> DualQuaternion<T>
{
    (quaternion::scale(q.0, t), quaternion::scale(q.1, t))
}

/// Returns the dual-quaternion conjugate
#[inline(always)]
pub fn conj<T: Float>(q: DualQuaternion<T>) -> DualQuaternion<T> {
    (
        quaternion::conj(q.0),
        quaternion::conj(q.1)
    )
}

/// Dot product of two dual-quaternions
#[inline(always)]
pub fn dot<T: Float>(a: DualQuaternion<T>, b: DualQuaternion<T>) -> T {
    quaternion::dot(a.0, b.0)
}

/// Normalizes a dual-quaternion
pub fn normalize<T: Float>(q: DualQuaternion<T>) -> DualQuaternion<T> {
    let real_len_recip = T::one() / dot(q, q).sqrt();
    (
        quaternion::scale(q.0, real_len_recip),
        quaternion::add(
            quaternion::scale(q.1, real_len_recip),
            quaternion::scale(q.0, -quaternion::dot(q.0, q.1))
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

        // rotate 90 degrees about Y
        let r1 = quaternion::euler_angles(0.0, PI / 2.0, 0.0);

        // rotate 90 degrees about Z
        let r2 = quaternion::euler_angles(0.0, 0.0, PI / 2.0);

        // rotate 90 degrees about X
         let r3 = quaternion::euler_angles(PI / 2.0, 0.0, 0.0);

        let dq1 = super::from_rotation_and_translation(r1, [0.0, 0.0, 0.0]);
        let dq2 = super::from_rotation_and_translation(r2, [0.0, 0.0, 0.0]);
        let dq3 = super::from_rotation_and_translation(r3, [0.0, 0.0, 0.0]);
        let dq4 = super::mul(super::mul(dq1, dq2), dq3);

        let r_prime = super::get_rotation(dq4);
        let t_prime = super::get_translation(dq4);

        let t_expected = [0.0, 0.0, 0.0];
        assert!((t_prime[0] - t_expected[0]).abs() < EPSILON);
        assert!((t_prime[1] - t_expected[1]).abs() < EPSILON);
        assert!((t_prime[2] - t_expected[2]).abs() < EPSILON);

        let rotate_test = quaternion::rotate_vector(r_prime, [1.0, 0.0, 0.0]);
        let expected = [0.0, 1.0, 0.0];

        println!("{:?}", rotate_test);

        assert!((rotate_test[0] - expected[0]).abs() < EPSILON);
        assert!((rotate_test[1] - expected[1]).abs() < EPSILON);
        assert!((rotate_test[2] - expected[2]).abs() < EPSILON);

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

    #[test]
    fn test_mul_conj() {
        let r = quaternion::euler_angles(PI, PI, PI);
        let t = [1.0, 2.0, 3.0];

        let dq = super::from_rotation_and_translation(r, t);
        let dq_conj = super::conj(dq);

        let dq_prime = super::mul(dq, dq_conj);
        let r_prime = super::get_rotation(dq_prime);
        let t_prime = super::get_translation(dq_prime);

        assert!((t_prime[0] - 0.0).abs() < EPSILON);
        assert!((t_prime[1] - 0.0).abs() < EPSILON);
        assert!((t_prime[2] - 0.0).abs() < EPSILON);

        assert!((r_prime.0 - 1.0).abs() < EPSILON);
        assert!((r_prime.1[0] - 0.0).abs() < EPSILON);
        assert!((r_prime.1[1] - 0.0).abs() < EPSILON);
        assert!((r_prime.1[2] - 0.0).abs() < EPSILON);
    }

}
