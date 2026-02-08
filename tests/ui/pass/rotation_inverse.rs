use cardo::prelude::*;

struct World;
struct Body;

fn main() {
    let r: SO3<Body, World> = SO3::identity();
    let r_inv: SO3<World, Body> = r.inverse();

    // Composing with inverse gives identity (frames cancel out)
    let _identity: SO3<Body, Body> = r_inv * r;
    let _identity: SO3<World, World> = r * r_inv;
}
