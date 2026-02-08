use cardo::prelude::*;

// Define coordinate frames as zero-sized types
struct World;
struct Body;
struct Sensor;

fn main() {
    // ==========================================================
    // Frame-Safe Geometry: Compile-time coordinate frame checking
    // ==========================================================

    // Sensor -> Body
    let r1: SO3<Sensor, Body> = SO3::identity();

    // Body -> World
    let r2: SO3<Body, World> = SO3::identity();

    // Compose Sensor -> Body -> World
    let r: SO3<Sensor, World> = r2 * r1;

    // Rotate vector from Sensor to World
    let v: Vector3<Sensor> = Vector3::new(1.0, 0.0, 0.0);
    let w = r * v;

    // Frame mismatch? The compiler catches it:
    //
    //   let v: Vector3<Body> = Vector3::new(1.0, 0.0, 0.0);
    //   let w = r * v;
    //            ^^^ error: SO3<Sensor, World> cannot act on Vector3<Body>
    //
    // This bug would silently produce wrong results in other libraries.
    // With cardo, it's a compile-time error.

    println!("{:?}", w);
}
