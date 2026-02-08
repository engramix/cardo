use cardo::prelude::*;

struct World;
struct Body;

fn main() {
    let r: SO3<Body, World> = SO3::identity();
    let v: Vector3<Body> = Vector3::new(1.0, 0.0, 0.0);

    // Correct: r expects Body frame, v is in Body frame
    let v_world: Vector3<World> = r * v;

    let _ = v_world;
}
