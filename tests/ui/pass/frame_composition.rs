use cardo::prelude::*;

struct World;
struct Body;
struct Sensor;

fn main() {
    let r1: SO3<Sensor, Body> = SO3::identity();
    let r2: SO3<Body, World> = SO3::identity();

    let r: SO3<Sensor, World> = r2 * r1;

    let _ = r;
}
