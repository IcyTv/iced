struct Uniforms {
    transform: mat4x4<f32>,
    sizes: vec4<f32>,
    clip: vec4<f32>,
    opacity: vec4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var content: texture_2d<f32>;
@group(0) @binding(2) var filtering: sampler;

struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) logical: vec2<f32>,
}
@vertex fn vertex_main(@builtin(vertex_index) index: u32) -> Vertex {
    let points = array<vec2<f32>, 6>(vec2(0., 0.), vec2(1., 0.), vec2(0., 1.),
                                   vec2(0., 1.), vec2(1., 0.), vec2(1., 1.));
    let uv = points[index];
    let point = uniforms.transform * vec4(uv * uniforms.sizes.xy, 0., 1.);
    var output: Vertex;
    output.position = vec4(point.x / uniforms.sizes.z * 2. - 1.,
                           1. - point.y / uniforms.sizes.w * 2., 0., 1.);
    output.uv = uv;
    output.logical = point.xy;
    return output;
}
@fragment fn fragment_main(input: Vertex) -> @location(0) vec4<f32> {
    if any(input.logical < uniforms.clip.xy) || any(input.logical >= uniforms.clip.zw) {
        discard;
    }
    // Captured pixels are already premultiplied. Scale RGB and alpha together.
    return textureSample(content, filtering, input.uv) * uniforms.opacity.x;
}
