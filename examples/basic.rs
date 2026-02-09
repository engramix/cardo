use cardo::prelude::*;

struct World;
struct Body;
struct Camera;

fn main() {
    let camera_to_body: SO3<Camera, Body> = SO3::identity();
    let body_to_world: SO3<Body, World> = SO3::identity();

    // Chain transformations: Camera -> Body -> World
    let camera_to_world: SO3<Camera, World> = chain!(camera_to_body -> body_to_world);

    let v: Vector3<Camera> = Vector3::new(1.0, 0.0, 0.0);
    let w: Vector3<World> = camera_to_world * v;

    // Frame mismatch? The compiler catches it:
    //
    //   let v: Vector3<Body> = Vector3::new(1.0, 0.0, 0.0);
    //   let w = camera_to_world * v;
    //
    //   error[E0277]: cannot multiply `SO3<Camera, World>` by `Vector3<Body>`
    //     |
    //     |     let w = camera_to_world * v;
    //     |                             ^ the trait `Mul<Vector3<Body>>` is not
    //     |                               implemented for `SO3<Camera, World>`

    println!("{:?}", w);
}
