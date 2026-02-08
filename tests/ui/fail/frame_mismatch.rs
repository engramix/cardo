use cardo::prelude::*;

struct World;
struct Body;
struct Sensor;

fn main() {
    let r1: SO3<Sensor, Body> = SO3::identity();
    let r2: SO3<Body, World> = SO3::identity();

    let _r = r1 * r2;

    let v: Vector3<Sensor> = Vector3::zero();
    let _v2 = r1 * v;
}
