// === Setup ===
const canvas = document.getElementById("glCanvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const gl = canvas.getContext("webgl2");
if (!gl) throw new Error("WebGL2 not supported");

// === Shader sources ===
const vertexShaderSrc = `#version 300 es
precision mediump float;
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const fragmentShaderSrc = `#version 300 es
precision highp float;
out vec4 outColor;
in vec2 v_uv;

#define SPHERE_COUNT 10

uniform vec3 u_cameraPos;
uniform vec3 u_cameraTarget;
uniform vec3 u_cameraUp;
uniform float u_fov;
uniform float u_aspect;
uniform vec3 u_sphereCenters[SPHERE_COUNT];
uniform float u_sphereRadii[SPHERE_COUNT];
uniform vec3 u_colors[SPHERE_COUNT];

float sphereIntersect(vec3 ro, vec3 rd, vec3 center, float radius) {
  vec3 oc = ro - center;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - radius * radius;
  float h = b * b - c;
  if (h < 0.0) return -1.0;
  return -b - sqrt(h);
}

float random(vec2 seed) {
  return fract(sin(dot(seed.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float computeSoftShadow(vec3 hit, vec3 normal, vec3 lightPos) {
  vec3 lightDir = normalize(lightPos - hit);
  float lightDist = length(lightPos - hit);
  vec3 shadowOrigin = hit + normal * 0.001;

  float shadow = 0.0;
  int samples = 16;

  for (int i = 0; i < samples; i++) {
    vec2 pixelSeed = v_uv * 1000.0;

    vec3 randOffset = vec3(
      random(vec2(float(i), pixelSeed.x)),
      random(vec2(float(i) + 10.0, pixelSeed.y * 0.732)),
      random(vec2(float(i) + 20.0, pixelSeed.x * 1.618))
    ) - vec3(0.5);

    randOffset *= 0.05; // light size

    vec3 sampleDir = normalize(lightDir + randOffset);

    bool blocked = false;
    for (int j = 0; j < SPHERE_COUNT; j++) {
      float tShadow = sphereIntersect(shadowOrigin, sampleDir, u_sphereCenters[j], u_sphereRadii[j]);
      if (tShadow > 0.0 && tShadow < lightDist) {
        blocked = true;
        break;
      }
    }

    shadow += blocked ? 0.0 : 1.0;
  }

  shadow /= float(samples);
  return shadow;
}

vec3 raytrace(vec2 uv) {
  vec3 forward = normalize(u_cameraTarget - u_cameraPos);
  vec3 right = normalize(cross(forward, u_cameraUp));
  vec3 up = cross(right, forward);

  uv = uv * 2.0 - 1.0;
  uv.x *= u_aspect;
  float focalLength = 1.0 / tan(u_fov * 0.5);
  vec3 ro = u_cameraPos;
  vec3 rd = normalize(forward * focalLength + right * uv.x + up * uv.y);

  float closestT = 1e20;
  int hitSphere = -1;

  for (int i = 0; i < SPHERE_COUNT; i++) {
    float t = sphereIntersect(ro, rd, u_sphereCenters[i], u_sphereRadii[i]);
    if (t > 0.0 && t < closestT) {
      closestT = t;
      hitSphere = i;
    }
  }

  if (hitSphere != -1) {
    vec3 hit = ro + closestT * rd;
    vec3 normal = normalize(hit - u_sphereCenters[hitSphere]);

    vec3 lightPos = vec3(5.0, 5.0, 5.0);
    vec3 lightDir = normalize(lightPos - hit);

    float diffuse = max(dot(normal, lightDir), 0.0);

    float shadow = computeSoftShadow(hit, normal, lightPos);

    vec3 color = u_colors[hitSphere];
    return color * (shadow * diffuse); // No ambient — pure soft shadow * diffuse
  }

  // Background
  return vec3(0.6, 0.8, 1.0);
}

void main() {
  vec3 color = vec3(0.0);
  float offset = 1.0 / 800.0;
  for (int dx = 0; dx < 1; dx++) {
    for (int dy = 0; dy < 1; dy++) {
      vec2 offsetUV = v_uv + vec2(float(dx), float(dy)) * offset;
      color += raytrace(offsetUV);
    }
  }
  color /= 1.0;
  outColor = vec4(color, 1.0);
}`;

function createShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

// Compile and link
const program = gl.createProgram();
const vs = createShader(gl, gl.VERTEX_SHADER, vertexShaderSrc);
const fs = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSrc);
gl.attachShader(program, vs);
gl.attachShader(program, fs);
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS))
  throw new Error(gl.getProgramInfoLog(program));

// Quad setup
const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
const vao = gl.createVertexArray();
const vbo = gl.createBuffer();
const loc = gl.getAttribLocation(program, "a_position");

gl.bindVertexArray(vao);
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
gl.enableVertexAttribArray(loc);
gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

// Uniforms
const u_cameraPosLoc = gl.getUniformLocation(program, "u_cameraPos");
const u_cameraTargetLoc = gl.getUniformLocation(program, "u_cameraTarget");
const u_cameraUpLoc = gl.getUniformLocation(program, "u_cameraUp");
const u_fovLoc = gl.getUniformLocation(program, "u_fov");
const u_aspectLoc = gl.getUniformLocation(program, "u_aspect");
const u_sphereCentersLoc = gl.getUniformLocation(program, "u_sphereCenters");
const u_sphereRadiiLoc = gl.getUniformLocation(program, "u_sphereRadii");
const u_colorsLoc = gl.getUniformLocation(program, "u_colors");

const sphereCenters = [
  [0.0, 0.0, -2.0], [1.2, 0.5, -3.0], [-1.0, -0.3, -2.5], [-2.0, 0.5, -4.0],
  [2.5, -0.5, -5.0], [0.5, 1.5, -4.5], [-1.5, -1.0, -3.5], [1.5, -1.0, -2.8],
  [-2.5, 1.2, -6.0], [0.0, 2.0, -5.5]
];

const sphereRadii = [0.6, 0.4, 0.5, 0.7, 0.6, 0.5, 0.6, 0.5, 0.8, 0.7];
const colors = [
  [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0],
  [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]
];

let cameraPos = [0, 0, 1];
let yaw = 0;
let pitch = 0;
let keys = [];

// FPS Counter
const fpsCounter = document.getElementById("fpsCounter");
let lastFrameTime = performance.now();
let frameCount = 0;
let lastFpsUpdate = performance.now();

window.addEventListener("keydown", e => {
  if (!keys.includes(e.key)) keys.push(e.key);
});

window.addEventListener("keyup", e => {
  const index = keys.indexOf(e.key);
  if (index !== -1) keys.splice(index, 1);
});

canvas.requestPointerLock = canvas.requestPointerLock || canvas.mozRequestPointerLock;
canvas.onclick = () => canvas.requestPointerLock();

document.addEventListener("pointerlockchange", () => {
  if (document.pointerLockElement === canvas) {
    document.addEventListener("mousemove", onMouseMove, false);
  } else {
    document.removeEventListener("mousemove", onMouseMove, false);
  }
});

function onMouseMove(e) {
  const sensitivity = 0.002;
  yaw += e.movementX * sensitivity;
  pitch -= e.movementY * sensitivity;
  pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, pitch));
}

function render() {
  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);

  const forward = [
    cosPitch * sinYaw,
    sinPitch,
    cosPitch * cosYaw * -1
  ];

  const right = [cosYaw, 0, sinYaw];
  const speed = 0.05;

  if (keys.includes("w") || keys.includes("W")) {
    cameraPos[0] += forward[0] * speed;
    cameraPos[1] += forward[1] * speed;
    cameraPos[2] += forward[2] * speed;
  }
  if (keys.includes("s") || keys.includes("S")) {
    cameraPos[0] -= forward[0] * speed;
    cameraPos[1] -= forward[1] * speed;
    cameraPos[2] -= forward[2] * speed;
  }
  if (keys.includes("a") || keys.includes("A")) {
    cameraPos[0] -= right[0] * speed;
    cameraPos[2] -= right[2] * speed;
  }
  if (keys.includes("d") || keys.includes("D")) {
    cameraPos[0] += right[0] * speed;
    cameraPos[2] += right[2] * speed;
  }

  const cameraTarget = [
    cameraPos[0] + forward[0],
    cameraPos[1] + forward[1],
    cameraPos[2] + forward[2]
  ];

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(program);
  gl.bindVertexArray(vao);

  gl.uniform3fv(u_cameraPosLoc, cameraPos);
  gl.uniform3fv(u_cameraTargetLoc, cameraTarget);
  gl.uniform3fv(u_cameraUpLoc, [0, 1, 0]);
  gl.uniform1f(u_fovLoc, Math.PI / 3);
  gl.uniform1f(u_aspectLoc, canvas.width / canvas.height);

  gl.uniform3fv(u_sphereCentersLoc, sphereCenters.flat());
  gl.uniform1fv(u_sphereRadiiLoc, sphereRadii);
  gl.uniform3fv(u_colorsLoc, colors.flat());

  const now = performance.now();
  frameCount++;
  if (now - lastFpsUpdate >= 500) {
    const fps = (frameCount / (now - lastFpsUpdate)) * 1000;
    fpsCounter.style.color = fps > 50 ? "green" : fps > 30 ? "yellow" : "red";
    fpsCounter.textContent = `FPS: ${fps.toFixed(1)}`;
    lastFpsUpdate = now;
    frameCount = 0;
  }

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  requestAnimationFrame(render);
}
render();