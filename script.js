// main.js
const canvas = document.getElementById("gpuCanvas");
canvas.width  = window.innerWidth;
canvas.height = window.innerHeight;
const gl = canvas.getContext("webgl2");
if (!gl) throw new Error("WebGL2 not supported");

const fpsCounter = document.getElementById('fpsCounter');

// — Helpers —
async function loadText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load ${url}`);
  return await r.text();
}

async function loadObj(url) {
  const txt = await loadText(url);
  const pos = [], nor = [], vIdx = [], nIdx = [];
  for (let line of txt.split("\n")) {
    line = line.trim();
    if (line.startsWith("v ")) {
      const [,x,y,z] = line.split(/\s+/);
      pos.push(+x,+y,+z);
    } else if (line.startsWith("vn ")) {
      const [,x,y,z] = line.split(/\s+/);
      nor.push(+x,+y,+z);
    } else if (line.startsWith("f ")) {
      const parts = line.split(/\s+/).slice(1);
      for (let i=1; i<parts.length-1; i++) {
        const [v0,n0] = parts[0].split("//").map(v=>+v-1);
        const [v1,n1] = parts[i  ].split("//").map(v=>+v-1);
        const [v2,n2] = parts[i+1].split("//").map(v=>+v-1);
        vIdx.push(v0,v1,v2);
        nIdx.push(n0,n1,n2);
      }
    }
  }
  return {
    positions: new Float32Array(pos),
    normals:   new Float32Array(nor),
    vertIdx:   new Uint32Array(vIdx),
    normIdx:   new Uint32Array(nIdx)
  };
}

// — Build a simple median-split BVH —
function buildBVH(mesh) {
  const nodes = [];
  const triOrder = [];
  const allTris = Array.from({length: mesh.vertIdx.length/3}, (_,i)=>i);

  function recurse(tris) {
    const idx = nodes.length;
    nodes.push(null);
    let mn = [1e9,1e9,1e9], mx=[-1e9,-1e9,-1e9];
    for (let ti of tris) {
      for (let v=0; v<3; v++) {
        const off = mesh.vertIdx[ti*3+v]*3;
        for (let k=0;k<3;k++){
          const c = mesh.positions[off+k];
          mn[k]=Math.min(mn[k],c);
          mx[k]=Math.max(mx[k],c);
        }
      }
    }
    if (tris.length <= 1) {
      const start = triOrder.length;
      tris.forEach(ti=>triOrder.push(ti));
      nodes[idx] = { mn, mx, left:start, right:-tris.length };
    } else {
      const ext = [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]];
      const ax = ext.indexOf(Math.max(...ext));
      tris.sort((a,b)=> {
        let ca=[0,0,0], cb=[0,0,0];
        for(let v=0;v<3;v++){
          const oa = mesh.vertIdx[a*3+v]*3;
          ca[0]+=mesh.positions[oa];
          ca[1]+=mesh.positions[oa+1];
          ca[2]+=mesh.positions[oa+2];
          const ob = mesh.vertIdx[b*3+v]*3;
          cb[0]+=mesh.positions[ob];
          cb[1]+=mesh.positions[ob+1];
          cb[2]+=mesh.positions[ob+2];
        }
        return (ca[ax]/3) - (cb[ax]/3);
      });
      const mid = Math.floor(tris.length/2);
      const leftIdx  = recurse(tris.slice(0,mid));
      const rightIdx = recurse(tris.slice(mid));
      nodes[idx] = { mn, mx, left:leftIdx, right:rightIdx };
    }
    return idx;
  }

  recurse(allTris);
  return { nodes, triOrder };
}

// — Shader setup —
function compile(gl,type,src){
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function link(gl, vs, fs){
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER,   vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p));
  return p;
}

// — Main bootstrap —
async function main(){
  const [vsSrc, fsSrc, mesh] = await Promise.all([
    loadText("shader.vert"),
    loadText("shader.frag"),
    loadObj("model.obj")
  ]);
  const program = link(gl, vsSrc, fsSrc);
  gl.useProgram(program);

  // screen quad
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const qb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, qb);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
     -1,-1,  1,-1,  -1,1,  1,1
  ]), gl.STATIC_DRAW);
  const posLoc = gl.getAttribLocation(program, "a_position");
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  // build BVH
  const { nodes, triOrder } = buildBVH(mesh);
  const triCount  = triOrder.length;
  const nodeCount = nodes.length;

  // flatten triangles into RGBA32F texture unit 0
  const floatsPerTri = 3 * 4;
  const triTexSize   = Math.ceil(Math.sqrt(triCount * floatsPerTri / 4));
  const triData      = new Float32Array(triTexSize * triTexSize * 4);
  for (let i = 0; i < triCount; i++) {
    const ti = triOrder[i];
    for (let v = 0; v < 3; v++) {
      const offV = mesh.vertIdx[ti*3 + v] * 3;
      const pix  = (i*3 + v) * 4;
      triData[pix  ] = mesh.positions[offV];
      triData[pix+1] = mesh.positions[offV+1];
      triData[pix+2] = mesh.positions[offV+2];
      triData[pix+3] = 0;
    }
  }
  gl.activeTexture(gl.TEXTURE0);
  const triTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, triTex);
  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA32F,
    triTexSize, triTexSize, 0,
    gl.RGBA, gl.FLOAT, triData
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  // normals → unit 1
  const normData = new Float32Array(triTexSize * triTexSize * 4);
  for (let i = 0; i < triCount; i++) {
    const ti = triOrder[i];
    for (let v = 0; v < 3; v++) {
      const offN = mesh.normIdx[ti*3 + v] * 3;
      const pix  = (i*3 + v) * 4;
      normData[pix  ] = mesh.normals[offN];
      normData[pix+1] = mesh.normals[offN+1];
      normData[pix+2] = mesh.normals[offN+2];
      normData[pix+3] = 0;
    }
  }
  gl.activeTexture(gl.TEXTURE1);
  const normTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, normTex);
  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA32F,
    triTexSize, triTexSize, 0,
    gl.RGBA, gl.FLOAT, normData
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  // BVH nodes → unit 2
  const nodeFloats  = nodeCount * 8;
  const nodeTexSize = Math.ceil(Math.sqrt(nodeFloats / 4));
  const nodeData    = new Float32Array(nodeTexSize * nodeTexSize * 4);
  for (let i = 0; i < nodeCount; i++) {
    const n    = nodes[i];
    const base = i * 8;
    nodeData[base  ] = n.mn[0];
    nodeData[base+1] = n.mn[1];
    nodeData[base+2] = n.mn[2];
    nodeData[base+3] = n.left;
    nodeData[base+4] = n.mx[0];
    nodeData[base+5] = n.mx[1];
    nodeData[base+6] = n.mx[2];
    nodeData[base+7] = n.right;
  }
  gl.activeTexture(gl.TEXTURE2);
  const nodeTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, nodeTex);
  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA32F,
    nodeTexSize, nodeTexSize, 0,
    gl.RGBA, gl.FLOAT, nodeData
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  // set uniforms
  gl.uniform1i(gl.getUniformLocation(program, "u_triTex"),     0);
  gl.uniform1i(gl.getUniformLocation(program, "u_normTex"),    1);
  gl.uniform1i(gl.getUniformLocation(program, "u_nodeTex"),    2);
  gl.uniform1i(gl.getUniformLocation(program, "u_texSize"),    triTexSize);
  gl.uniform1i(gl.getUniformLocation(program, "u_nodeTexSize"),nodeTexSize);
  gl.uniform1i(gl.getUniformLocation(program, "u_triCount"),   triCount);
  gl.uniform1i(gl.getUniformLocation(program, "u_nodeCount"),  nodeCount);

  const uCamPos = gl.getUniformLocation(program, "u_cameraPos");
  const uCamTgt = gl.getUniformLocation(program, "u_cameraTarget");
  const uCamUp  = gl.getUniformLocation(program, "u_cameraUp");
  const uFov    = gl.getUniformLocation(program, "u_fov");
  const uAsp    = gl.getUniformLocation(program, "u_aspect");
  const uModel  = gl.getUniformLocation(program, "u_model");

  // camera + controls
  let cameraPos = [0,0,3], yaw = 0, pitch = 0, keys = [];
  window.addEventListener("keydown", e => { if (!keys.includes(e.key)) keys.push(e.key); });
  window.addEventListener("keyup",   e => { const i = keys.indexOf(e.key); if (i>=0) keys.splice(i,1); });
  canvas.onclick = () => canvas.requestPointerLock();
  document.addEventListener("pointerlockchange", () => {
    if (document.pointerLockElement === canvas)
      document.addEventListener("mousemove", onMouseMove);
    else
      document.removeEventListener("mousemove", onMouseMove);
  });
  function onMouseMove(e){
    const s = 0.002;
    yaw   += e.movementX * s;
    pitch = Math.max(-Math.PI/2+0.01, Math.min(Math.PI/2-0.01, pitch - e.movementY * s));
  }
  function updateCam(){
    const cp = Math.cos(pitch), sp = Math.sin(pitch),
          cy = Math.cos(yaw),   sy = Math.sin(yaw);
    const fwd  = [cp*sy, sp, -cp*cy],
          right= [ cy ,  0,  sy ];
    const speed = 0.05;
    if (keys.includes("w")) cameraPos = cameraPos.map((v,i)=>v + fwd[i]*speed);
    if (keys.includes("s")) cameraPos = cameraPos.map((v,i)=>v - fwd[i]*speed);
    if (keys.includes("a")) cameraPos = cameraPos.map((v,i)=>v - right[i]*speed);
    if (keys.includes("d")) cameraPos = cameraPos.map((v,i)=>v + right[i]*speed);
    return [ cameraPos, cameraPos.map((v,i)=>v+fwd[i]), [0,1,0] ];
  }

  // FPS smoothing init
  let lastTime  = performance.now();
  let smoothFps = 60;
  const alpha   = 1;

  // render loop
  function render(){
    const now = performance.now();
    const dt  = now - lastTime;
    const currentFps = 1000 / dt;
    smoothFps = alpha * currentFps + (1 - alpha) * smoothFps;
    fpsCounter.textContent = `FPS: ${smoothFps.toFixed(1)}`;
    lastTime = now;

    // rotate model (optional)
    const t = now * 0.001, c = Math.cos(t), s = Math.sin(t);
    gl.uniformMatrix4fv(uModel, false, new Float32Array([
       c,0, s,0,
       0,1, 0,0,
      -s,0, c,0,
       0,0, 0,1,
    ]));

    // update camera uniforms
    const [cp, ct, cu] = updateCam();
    gl.uniform3fv(uCamPos, cp);
    gl.uniform3fv(uCamTgt, ct);
    gl.uniform3fv(uCamUp,  cu);
    gl.uniform1f( uFov,   Math.PI/4);
    gl.uniform1f( uAsp,   canvas.width / canvas.height);

    // draw
    gl.viewport(0,0,canvas.width,canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    requestAnimationFrame(render);
  }

  render();
}

main().catch(console.error);