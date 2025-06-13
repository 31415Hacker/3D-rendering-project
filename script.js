// === raytracer_temporal.js ===

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
  [0,0,1],[1,0,0],[0,1,0],[0,0,1],[1,0,0],
  [0,1,0],[0,0,1],[1,0,0],[0,1,0],[1,1,1]
];

let cameraPos = [0,0,1];
let yaw = 0, pitch = 0;
const keys = [];

let lastFpsUpdate = performance.now();
let frameCount = 0;
let frameIndex = 0;
const MAX_FRAMES = 10000;       // maximum accumulation frames
const MOVE_THRESHOLD = 0.001;  // movement speed that reduces accumulation

// store previous camera state for speed calculation
let prevCameraPos = [...cameraPos];
let prevYaw = yaw, prevPitch = pitch;

// === Temporal accumulation setup ===
const accumFBO = gl.createFramebuffer();
let texA = createAccumTexture(), texB = createAccumTexture();
let pingTex = texA, pongTex = texB;

function createAccumTexture() {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, canvas.width, canvas.height, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

// === Input Handling ===
window.addEventListener("keydown", e => { if (!keys.includes(e.key)) keys.push(e.key); });
window.addEventListener("keyup", e => { const i = keys.indexOf(e.key); if (i !== -1) keys.splice(i,1); });
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
  const s = 0.002;
  yaw   += e.movementX * s;
  pitch -= e.movementY * s;
  pitch = Math.max(-Math.PI/2+0.01, Math.min(Math.PI/2-0.01, pitch));
}

// === Utility: Compile a shader ===
function createShader(gl,type,src){
  const s = gl.createShader(type);
  gl.shaderSource(s,src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s,gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}

// === Step 1: Load shader sources ===
async function loadShaders(){
  const [vsR,fsR] = await Promise.all([ fetch('shader.vert'), fetch('shader.frag') ]);
  if (!vsR.ok) throw new Error("Vertex shader failed to load");
  if (!fsR.ok) throw new Error("Fragment shader failed to load");
  const [vsSrc,fsSrc] = await Promise.all([ vsR.text(), fsR.text() ]);
  return { vsSrc, fsSrc };
}

// === Step 2: Create & link program ===
function createProgram(gl,vsSrc,fsSrc){
  const vs = createShader(gl,gl.VERTEX_SHADER,vsSrc);
  const fs = createShader(gl,gl.FRAGMENT_SHADER,fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog,vs);
  gl.attachShader(prog,fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog,gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

// === Step 3: Setup VAO + Uniforms ===
function setupScene(gl,prog){
  const quad = new Float32Array([-1,-1,1,-1,-1,1,1,1]);
  const vao = gl.createVertexArray(), vbo = gl.createBuffer();
  const posLoc = gl.getAttribLocation(prog,"a_position");
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER,vbo);
  gl.bufferData(gl.ARRAY_BUFFER,quad,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc,2,gl.FLOAT,false,0,0);
  return {
    vao,
    u_cameraPos:    gl.getUniformLocation(prog,"u_cameraPos"),
    u_cameraTarget: gl.getUniformLocation(prog,"u_cameraTarget"),
    u_cameraUp:     gl.getUniformLocation(prog,"u_cameraUp"),
    u_fov:          gl.getUniformLocation(prog,"u_fov"),
    u_aspect:       gl.getUniformLocation(prog,"u_aspect"),
    u_centers:      gl.getUniformLocation(prog,"u_sphereCenters"),
    u_radii:        gl.getUniformLocation(prog,"u_sphereRadii"),
    u_colors:       gl.getUniformLocation(prog,"u_colors"),
    u_time:         gl.getUniformLocation(prog,"u_time"),
    u_frameIndex:   gl.getUniformLocation(prog,"u_frameIndex"),
    u_maxFrames:    gl.getUniformLocation(prog,"u_maxFrames"),
    u_prevFrameTex: gl.getUniformLocation(prog,"u_prevFrameTex")
  };
}

// === Step 4: Render Loop with dynamic accumulation cap ===
function startRenderLoop(gl,prog,scene){
  const {
    vao,u_cameraPos,u_cameraTarget,u_cameraUp,
    u_fov,u_aspect,u_centers,u_radii,u_colors,
    u_time,u_frameIndex,u_prevFrameTex,u_maxFrames
  } = scene;

  function render(){
    // update camera
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    const cosY = Math.cos(yaw),   sinY = Math.sin(yaw);
    const forward = [cosP*sinY, sinP, -cosP*cosY];
    const right   = [cosY, 0, sinY];
    const speed = 0.05;
    if(keys.includes("w")) cameraPos = cameraPos.map((v,i)=>v+forward[i]*speed);
    if(keys.includes("s")) cameraPos = cameraPos.map((v,i)=>v-forward[i]*speed);
    if(keys.includes("a")) cameraPos = cameraPos.map((v,i)=>v-right[i]*speed);
    if(keys.includes("d")) cameraPos = cameraPos.map((v,i)=>v+right[i]*speed);
    const cameraTarget = cameraPos.map((v,i)=>v+forward[i]);

    // compute movement metric
    const dp = Math.hypot(
      cameraPos[0]-prevCameraPos[0],
      cameraPos[1]-prevCameraPos[1],
      cameraPos[2]-prevCameraPos[2]
    );
    const da = Math.abs(yaw - prevYaw) + Math.abs(pitch - prevPitch);
    const movement = dp + da;
    // dynamic max accumulation frames
    const tNorm = Math.min(movement / MOVE_THRESHOLD, 1.0);
    const dynamicMax = Math.max(1, Math.floor(MAX_FRAMES * (1.0 - tNorm)));

    // update frameIndex with cap
    frameIndex = Math.min(frameIndex + 1, dynamicMax - 1);

    // set GL state
    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.viewport(0, 0, canvas.width, canvas.height);

    gl.uniform3fv(u_cameraPos,    cameraPos);
    gl.uniform3fv(u_cameraTarget, cameraTarget);
    gl.uniform3fv(u_cameraUp,      [0,1,0]);
    gl.uniform1f(u_fov,            Math.PI/4);
    gl.uniform1f(u_aspect,         canvas.width/canvas.height);
    gl.uniform3fv(u_centers,       sphereCenters.flat());
    gl.uniform1fv(u_radii,         sphereRadii);
    gl.uniform3fv(u_colors,        colors.flat());
    gl.uniform1f(u_time,           performance.now()*0.001);
    gl.uniform1i(u_frameIndex,     frameIndex);
    gl.uniform1i(u_maxFrames,      dynamicMax);

    // bind previous frame texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pingTex);
    gl.uniform1i(u_prevFrameTex, 0);

    // render into FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, accumFBO);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, pongTex, 0
    );
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // blit to screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, pongTex);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // ping-pong
    [pingTex, pongTex] = [pongTex, pingTex];

    // update prev camera for next frame
    prevCameraPos = [...cameraPos];
    prevYaw = yaw; prevPitch = pitch;

    // FPS counter
    frameCount++;
    const now = performance.now();
    if (now - lastFpsUpdate >= 500) {
      const fps = (frameCount / (now - lastFpsUpdate)) * 1000;
      fpsCounter.style.color = fps > 50 ? "green" : fps > 30 ? "yellow" : "red";
      fpsCounter.textContent = `FPS: ${fps.toFixed(1)}`;
      lastFpsUpdate = now;
      frameCount = 0;
    }

    requestAnimationFrame(render);
  }
  render();
}

// === Bootstrap ===
async function main(){
  const { vsSrc, fsSrc } = await loadShaders();
  const prog = createProgram(gl, vsSrc, fsSrc);
  const scene=setupScene(gl, prog);
  startRenderLoop(gl, prog, scene);
}
main().catch(console.error);