// === Global State & Constants ===
const canvas = document.getElementById("glCanvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const gl = canvas.getContext("webgl2");
if (!gl) throw new Error("WebGL2 not supported");

const fpsCounter = document.getElementById("fpsCounter");

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
let yaw = 0, pitch = 0;
const keys = [];

let lastFpsUpdate = performance.now();
let frameCount = 0;

// === Input Handling ===
window.addEventListener("keydown", e => {
  if (!keys.includes(e.key)) keys.push(e.key);
});
window.addEventListener("keyup",   e => {
  const i = keys.indexOf(e.key);
  if (i !== -1) keys.splice(i, 1);
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
  yaw   += e.movementX * sensitivity;
  pitch -= e.movementY * sensitivity;
  pitch = Math.max(-Math.PI/2 + 0.01, Math.min(Math.PI/2 - 0.01, pitch));
}

// === Utility: Compile a shader ===
function createShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

// === Step 1: Load shader source texts ===
async function loadShaders() {
  const [vsResp, fsResp] = await Promise.all([
    fetch('shader.vert'),
    fetch('shader.frag')
  ]);
  if (!vsResp.ok) throw new Error("Vertex shader failed to load");
  if (!fsResp.ok) throw new Error("Fragment shader failed to load");
  const [vsSrc, fsSrc] = await Promise.all([vsResp.text(), fsResp.text()]);
  return { vsSrc, fsSrc };
}

// === Step 2: Create & link program ===
function createProgramFromSources(gl, vsSrc, fsSrc) {
  const vs = createShader(gl, gl.VERTEX_SHADER,   vsSrc);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

// === Step 3: Set up VAO + capture uniform locations ===
function setupScene(gl, program) {
  // Quad covering NDC
  const quad = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  const posLoc = gl.getAttribLocation(program, "a_position");

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  // Uniforms
  return {
    vao,
    u_cameraPos:    gl.getUniformLocation(program, "u_cameraPos"),
    u_cameraTarget: gl.getUniformLocation(program, "u_cameraTarget"),
    u_cameraUp:     gl.getUniformLocation(program, "u_cameraUp"),
    u_fov:          gl.getUniformLocation(program, "u_fov"),
    u_aspect:       gl.getUniformLocation(program, "u_aspect"),
    u_centers:      gl.getUniformLocation(program, "u_sphereCenters"),
    u_radii:        gl.getUniformLocation(program, "u_sphereRadii"),
    u_colors:       gl.getUniformLocation(program, "u_colors")
  };
}

// === Step 4: Render loop ===
function startRenderLoop(gl, program, scene) {
  const { vao, u_cameraPos, u_cameraTarget, u_cameraUp, u_fov, u_aspect, u_centers, u_radii, u_colors } = scene;

  function render() {
    // Camera orientation
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    const cosY = Math.cos(yaw),   sinY = Math.sin(yaw);
    const forward = [ cosP*sinY, sinP, -cosP*cosY ];
    const right   = [ cosY, 0, sinY ];
    const speed = 0.05;

    // WASD movement
    if (keys.includes("w")||keys.includes("W")) cameraPos = cameraPos.map((v,i)=>v + forward[i]*speed);
    if (keys.includes("s")||keys.includes("S")) cameraPos = cameraPos.map((v,i)=>v - forward[i]*speed);
    if (keys.includes("a")||keys.includes("A")) cameraPos = cameraPos.map((v,i)=>v - right[i]*speed);
    if (keys.includes("d")||keys.includes("D")) cameraPos = cameraPos.map((v,i)=>v + right[i]*speed);

    const cameraTarget = cameraPos.map((v,i)=>v + forward[i]);

    // Draw
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.uniform3fv(u_cameraPos,    cameraPos);
    gl.uniform3fv(u_cameraTarget, cameraTarget);
    gl.uniform3fv(u_cameraUp,      [0,1,0]);
    gl.uniform1f(u_fov,            Math.PI/3);
    gl.uniform1f(u_aspect,         canvas.width/canvas.height);
    gl.uniform3fv(u_centers,       sphereCenters.flat());
    gl.uniform1fv(u_radii,         sphereRadii);
    gl.uniform3fv(u_colors,        colors.flat());

    // FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsUpdate >= 500) {
      const fps = (frameCount/(now-lastFpsUpdate))*1000;
      fpsCounter.style.color = fps>50 ? "green" : fps>30 ? "yellow" : "red";
      fpsCounter.textContent = `FPS: ${fps.toFixed(1)}`;
      lastFpsUpdate = now;
      frameCount = 0;
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    requestAnimationFrame(render);
  }

  render();
}

// === Bootstrap ===
async function main() {
  const { vsSrc, fsSrc } = await loadShaders();
  const program = createProgramFromSources(gl, vsSrc, fsSrc);
  const scene   = setupScene(gl, program);
  startRenderLoop(gl, program, scene);
}

main().catch(err => console.error(err));