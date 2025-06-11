// Setup
const canvas = document.getElementById("glCanvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const gl = canvas.getContext("webgl2");
if (!gl) throw new Error("WebGL2 not supported");

// Shaders
const vertexShaderSrc = `#version 300 es
precision mediump float;
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const fragmentShaderSrc = `#version 300 es
precision highp float;
out vec4 outColor;
in vec2 v_uv;

// Uniforms
uniform vec3 u_cameraPos;
uniform vec3 u_cameraTarget;
uniform vec3 u_cameraUp;
uniform float u_fov;
uniform float u_aspect;

// Sphere intersection
float sphereIntersect(vec3 ro, vec3 rd, vec3 center, float radius) {
  vec3 oc = ro - center;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - radius * radius;
  float h = b * b - c;
  if (h < 0.0) return -1.0;
  return -b - sqrt(h);
}

// Main
void main() {
  // Build camera basis
  vec3 forward = normalize(u_cameraTarget - u_cameraPos);
  vec3 right = normalize(cross(forward, u_cameraUp));
  vec3 up = cross(right, forward);

  // Screen UV
  vec2 uv = v_uv * 2.0 - 1.0;
  uv.x *= u_aspect;
  float focalLength = 1.0 / tan(u_fov * 0.5);

  // Ray setup
  vec3 ro = u_cameraPos;
  vec3 rd = normalize(forward * focalLength + right * uv.x + up * uv.y);

  // Scene
  vec3 sphereCenter = vec3(0.0, 0.0, -2.0);
  float t = sphereIntersect(ro, rd, sphereCenter, 0.6);

  if (t > 0.0) {
    vec3 hit = ro + t * rd;
    vec3 normal = normalize(hit - sphereCenter);
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(dot(normal, lightDir), 0.0);
    vec3 albedo = vec3(0.0, 0.0, 1.0);
    outColor = vec4(diffuse * albedo, 1.0); // lit color
  } else {
    outColor = vec4(0.6, 0.8, 1.0, 1.0); // background
  }
}
`;

// Shader helpers
function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSrc);
const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSrc);

const program = gl.createProgram();
gl.attachShader(program, vertexShader);
gl.attachShader(program, fragmentShader);
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS))
  throw new Error(gl.getProgramInfoLog(program));

// Quad
const quad = new Float32Array([
  -1, -1,  1, -1,
  -1,  1,  1,  1
]);

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

const loc = gl.getAttribLocation(program, "a_position");
gl.enableVertexAttribArray(loc);
gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

// Uniform locations
const u_cameraPosLoc = gl.getUniformLocation(program, "u_cameraPos");
const u_cameraTargetLoc = gl.getUniformLocation(program, "u_cameraTarget");
const u_cameraUpLoc = gl.getUniformLocation(program, "u_cameraUp");
const u_fovLoc = gl.getUniformLocation(program, "u_fov");
const u_aspectLoc = gl.getUniformLocation(program, "u_aspect");

// Camera parameters
let cameraPos = [0, 0, 1];
let cameraTarget = [0, 0, -2];
const cameraUp = [0, 1, 0];
const fov = Math.PI / 3; // 60 deg
const aspect = canvas.width / canvas.height;

// Keyboard handling
const keys = [];
window.addEventListener("keydown", e => {
  if (!keys.includes(e.key)) keys.push(e.key);
});
window.addEventListener("keyup", e => {
  const index = keys.indexOf(e.key);
  if (index !== -1) keys.splice(index, 1);
});

function render() {
  // Camera move speed
  const speed = 0.05;

  // Camera movement with WASD
  if (keys.includes("a") || keys.includes("A")) {
    cameraPos[0] -= speed;
  }
  if (keys.includes("d") || keys.includes("D")) {
    cameraPos[0] += speed;
  }
  if (keys.includes("w") || keys.includes("W")) {
    cameraPos[2] -= speed;
  }
  if (keys.includes("s") || keys.includes("S")) {
    cameraPos[2] += speed;
  }
  if (keys.includes("ArrowUp")) {
    cameraTarget[1] += speed;
  }
  if (keys.includes("ArrowDown")) {
    cameraTarget[1] -= speed;
  }
  if (keys.includes("ArrowRight")) {
    cameraTarget[2] += speed;
  }
  if (keys.includes("ArrowLeft")) {
    cameraTarget[2] -= speed;
  }

  // Render
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(program);
  gl.bindVertexArray(vao);

  // Upload uniforms
  gl.uniform3fv(u_cameraPosLoc, cameraPos);
  gl.uniform3fv(u_cameraTargetLoc, cameraTarget);
  gl.uniform3fv(u_cameraUpLoc, cameraUp);
  gl.uniform1f(u_fovLoc, fov);
  gl.uniform1f(u_aspectLoc, aspect);

  // Draw
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  requestAnimationFrame(render);
}
render();