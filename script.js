// script.js

async function main() {
  // — Setup WebGPU —
  const canvas = document.getElementById("gpuCanvas");
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const adapter = await navigator.gpu.requestAdapter();
  const device  = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const format  = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  // — Helpers to load text & OBJ —
  async function loadText(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Failed to load ${url}`);
    return r.text();
  }
  async function loadObj(url) {
    const txt = await loadText(url);
    const pos  = [], nor  = [], vIdx = [], nIdx = [];

    // Helper to convert an OBJ index (which may be negative) into a 0-based JS index
    function toIndex(raw, count) {
      const i = parseInt(raw, 10);
      return i > 0
        ? i - 1                // positive: 1→0, 2→1, …
        : count + i;           // negative: -1 → last (count-1), -2→(count-2), …
    }

    for (let line of txt.split("\n")) {
      line = line.trim();
      if (line.startsWith("v ")) {
        const [, x, y, z] = line.split(/\s+/);
        pos.push(+x, +y, +z);
      } else if (line.startsWith("vn ")) {
        const [, x, y, z] = line.split(/\s+/);
        nor.push(+x, +y, +z);
      } else if (line.startsWith("f ")) {
        const parts = line.split(/\s+/).slice(1);
        // Pre-split each vertex reference into [v,vt,vn]
        const refs = parts.map(p => p.split("/"));

        const vertCount = pos.length / 3;
        const normCount = nor.length / 3;

        // Triangulate any polygon (fan)
        for (let i = 1; i < refs.length - 1; i++) {
          // indices for the three corners
          const [v0, , n0] = refs[0];
          const [v1, , n1] = refs[i];
          const [v2, , n2] = refs[i + 1];

          const vi0 = toIndex(v0, vertCount);
          const vi1 = toIndex(v1, vertCount);
          const vi2 = toIndex(v2, vertCount);

          const ni0 = toIndex(n0, normCount);
          const ni1 = toIndex(n1, normCount);
          const ni2 = toIndex(n2, normCount);

          vIdx.push(vi0, vi1, vi2);
          nIdx.push(ni0, ni1, ni2);
        }
      }
    }

    return {
      positions: new Float32Array(pos),
      normals:   new Float32Array(nor),
      vertIdx:   new Uint32Array(vIdx),
      normIdx:   new Uint32Array(nIdx),
    };
  }

  // — Normalize mesh into unit cube centered at origin —
  function normalizeMesh(m) {
    const p=m.positions, n=p.length/3;
    const mn=[Infinity,Infinity,Infinity], mx=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<n;i++) for(let k=0;k<3;k++){
      const v=p[3*i+k];
      mn[k]=Math.min(mn[k],v);
      mx[k]=Math.max(mx[k],v);
    }
    const ctr=mn.map((v,i)=>(v+mx[i])*0.5);
    const sc=1/Math.max(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2]);
    for(let i=0;i<n;i++) for(let k=0;k<3;k++){
      p[3*i+k]=(p[3*i+k]-ctr[k])*sc;
    }
    return m;
  }

  // — Build LBVH (Morton + recursive) —
  function buildLBVH(mesh, leafSize=1) {
    const nodes=[], triOrder=[];
    const tris = mesh.vertIdx.length/3;
    const centroids = new Float32Array(tris*3);
    for(let i=0;i<tris;i++){
      let cx=0,cy=0,cz=0;
      for(let v=0;v<3;v++){
        const off=mesh.vertIdx[i*3+v]*3;
        cx+=mesh.positions[off];
        cy+=mesh.positions[off+1];
        cz+=mesh.positions[off+2];
      }
      centroids[i*3]=cx/3; centroids[i*3+1]=cy/3; centroids[i*3+2]=cz/3;
    }
    const sceneMin=[Infinity,Infinity,Infinity], sceneMax=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<tris;i++){
      for(let k=0;k<3;k++){
        const v=centroids[i*3+k];
        sceneMin[k]=Math.min(sceneMin[k],v);
        sceneMax[k]=Math.max(sceneMax[k],v);
      }
    }
    const diag=[sceneMax[0]-sceneMin[0],sceneMax[1]-sceneMin[1],sceneMax[2]-sceneMin[2]];
    function expandBits(v){
      v=(v*0x00010001)&0xFF0000FF;
      v=(v*0x00000101)&0x0F00F00F;
      v=(v*0x00000011)&0xC30C30C3;
      v=(v*0x00000005)&0x49249249;
      return v;
    }
    function morton(x,y,z){
      x=Math.min(0.999999,Math.max(0,x));
      y=Math.min(0.999999,Math.max(0,y));
      z=Math.min(0.999999,Math.max(0,z));
      const xi=x*1024|0, yi=y*1024|0, zi=z*1024|0;
      return (expandBits(xi)<<2)|(expandBits(yi)<<1)|expandBits(zi);
    }
    const mortonArr=new Uint32Array(tris), order=new Uint32Array(tris);
    for(let i=0;i<tris;i++){
      const cx=(centroids[i*3]-sceneMin[0])/diag[0];
      const cy=(centroids[i*3+1]-sceneMin[1])/diag[1];
      const cz=(centroids[i*3+2]-sceneMin[2])/diag[2];
      mortonArr[i]=morton(cx,cy,cz);
      order[i]=i;
    }
    order.sort((a,b)=>mortonArr[a]-mortonArr[b]);
    for(let i=0;i<tris;i++) triOrder.push(order[i]);
    function computeBounds(slice){
      const mn=[Infinity,Infinity,Infinity], mx=[-Infinity,-Infinity,-Infinity];
      for(const ti of slice){
        for(let v=0;v<3;v++){
          const off=mesh.vertIdx[ti*3+v]*3;
          for(let k=0;k<3;k++){
            const val=mesh.positions[off+k];
            mn[k]=Math.min(mn[k],val);
            mx[k]=Math.max(mx[k],val);
          }
        }
      }
      return [mn,mx];
    }
    function recurse(s,e){
      const idx=nodes.length; nodes.push(null);
      const slice=triOrder.slice(s,e), count=e-s;
      const [mn,mx]=computeBounds(slice);
      if(count<=leafSize){
        nodes[idx]={mn,mx,left:s,right:-count};
      } else {
        const mid=(s+e)>>>1, l=recurse(s,mid), r=recurse(mid,e);
        nodes[idx]={mn,mx,left:l,right:r};
      }
      return idx;
    }
    recurse(0,tris);
    return {nodes,triOrder};
  }

  // — Flatten to SSBO arrays (vec4 padded) —
  function flattenForSSBO(mesh,bvh){
    const triCount=bvh.triOrder.length;
    const triPos=new Float32Array(triCount*3*4);
    const triNor=new Float32Array(triCount*3*4);
    for(let i=0;i<triCount;i++){
      const ti=bvh.triOrder[i];
      for(let v=0;v<3;v++){
        const vi=mesh.vertIdx[ti*3+v],
              ni=mesh.normIdx[ti*3+v];
        const dst=(i*3+v)*4;
        triPos[dst  ]=mesh.positions[vi*3  ];
        triPos[dst+1]=mesh.positions[vi*3+1];
        triPos[dst+2]=mesh.positions[vi*3+2];
        triNor[dst  ]=mesh.normals  [ni*3  ];
        triNor[dst+1]=mesh.normals  [ni*3+1];
        triNor[dst+2]=mesh.normals  [ni*3+2];
      }
    }
    const nodeCount=bvh.nodes.length;
    const bvhData=new Float32Array(nodeCount*8);
    for(let i=0;i<nodeCount;i++){
      const n=bvh.nodes[i], b=i*8;
      bvhData[b  ]=n.mn[0];
      bvhData[b+1]=n.mn[1];
      bvhData[b+2]=n.mn[2];
      bvhData[b+3]=n.left;
      bvhData[b+4]=n.mx[0];
      bvhData[b+5]=n.mx[1];
      bvhData[b+6]=n.mx[2];
      bvhData[b+7]=n.right;
    }
    return {triPos,triNor,bvhData};
  }

  // — Load, build & flatten —
  const meshRaw=await loadObj("dragon.obj").then(normalizeMesh);
  const {nodes,triOrder}=buildLBVH(meshRaw, 1);
  const {triPos,triNor,bvhData}=flattenForSSBO(meshRaw,{nodes,triOrder});

  // — Split each SSBO exactly in half —
  function splitHalf(arr){
    const vec4Count=arr.length/4;
    const half=Math.ceil(vec4Count/2);
    return [
      arr.slice(0, half*4),
      arr.slice(half*4)
    ];
  }
  const [pos0,pos1] = splitHalf(triPos);
  const [nor0,nor1] = splitHalf(triNor);
  const [bvh0,bvh1] = splitHalf(bvhData);

  // — Create GPUStorageBuffers —
  function makeBuf(arr){
    const buf=device.createBuffer({
      size: arr.byteLength,
      usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buf,0,arr);
    return buf;
  }
  const triPos0Buf   = makeBuf(pos0);
  const triPos1Buf   = makeBuf(pos1);
  const triNor0Buf   = makeBuf(nor0);
  const triNor1Buf   = makeBuf(nor1);
  const bvhNodes0Buf = makeBuf(bvh0);
  const bvhNodes1Buf = makeBuf(bvh1);

  // — Uniform buffer —
  const uniformBuf = device.createBuffer({
    size: 144,
    usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST
  });

  // — Load WGSL template & inject counts —
  let wgsl = await loadText("shader.wgsl");
  wgsl = wgsl
    .replace("__COUNT_POS0__",   `${pos0.length/4}u`)
    .replace("__COUNT_NOR0__",   `${nor0.length/4}u`)
    .replace("__COUNT_BVH0__",   `${bvh0.length/4}u`);

  const module = device.createShaderModule({ code: wgsl });

  // — Quad buffer for VS →
  const quadVerts = new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]);
  const quadBuf   = device.createBuffer({
    size: quadVerts.byteLength,
    usage: GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(quadBuf,0,quadVerts);

  // — Bind group layout & entries →
  const bgl = device.createBindGroupLayout({ entries:[
    {binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
    {binding:1,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
    {binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
    {binding:3,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
    {binding:4,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
    {binding:5,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
    {binding:6,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},
  ]});
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex: { module, entryPoint:"vs_main",
      buffers:[{arrayStride:8,attributes:[{shaderLocation:0,offset:0,format:"float32x2"}]}]
    },
    fragment:{ module, entryPoint:"fs_main", targets:[{format}] },
    primitive:{ topology:"triangle-list" }
  });
  const bg = device.createBindGroup({ layout:bgl, entries:[
    {binding:0,resource:{buffer:triPos0Buf}},
    {binding:1,resource:{buffer:triPos1Buf}},
    {binding:2,resource:{buffer:triNor0Buf}},
    {binding:3,resource:{buffer:triNor1Buf}},
    {binding:4,resource:{buffer:bvhNodes0Buf}},
    {binding:5,resource:{buffer:bvhNodes1Buf}},
    {binding:6,resource:{buffer:uniformBuf}},
  ]});

  // — Camera & controls —
  let cameraPos=[0,0,3], yaw=0, pitch=0, keys=[];
  window.addEventListener("keydown",e=>{if(!keys.includes(e.key))keys.push(e.key);});
  window.addEventListener("keyup",  e=>{keys=keys.filter(k=>k!==e.key);});
  canvas.onclick=()=>canvas.requestPointerLock();
  document.addEventListener("pointerlockchange",()=>{
    if(document.pointerLockElement===canvas)
      document.addEventListener("mousemove",onMouseMove);
    else
      document.removeEventListener("mousemove",onMouseMove);
  });
  function onMouseMove(e){
    const s=0.002;
    yaw   += e.movementX*s;
    pitch = Math.max(-Math.PI/2+0.01,Math.min(Math.PI/2-0.01,pitch-e.movementY*s));
  }
  function updateCamera(){
    const cp=Math.cos(pitch), sp=Math.sin(pitch),
          cy=Math.cos(yaw),   sy=Math.sin(yaw);
    const fwd  =[cp*sy,sp,-cp*cy], right=[cy,0,sy];
    const speed=0.05;
    if(keys.includes("w")) cameraPos=cameraPos.map((v,i)=>v+fwd[i]*speed);
    if(keys.includes("s")) cameraPos=cameraPos.map((v,i)=>v-fwd[i]*speed);
    if(keys.includes("a")) cameraPos=cameraPos.map((v,i)=>v-right[i]*speed);
    if(keys.includes("d")) cameraPos=cameraPos.map((v,i)=>v+right[i]*speed);
    return {pos:cameraPos,tgt:cameraPos.map((v,i)=>v+fwd[i]),up:[0,1,0]};
  }

  // — Render loop —
  let lastTime=performance.now(), frameCount=0;
  function frame(nowMS){
    const now=nowMS*0.001;
    frameCount++;
    if(performance.now()-lastTime>=1000){
      const fps=frameCount/((performance.now()-lastTime)/1000);
      document.getElementById("fpsCounter").textContent=`FPS: ${fps.toFixed(1)}`;
      lastTime=performance.now(); frameCount=0;
    }
    // update uniforms
    const {pos,tgt,up}=updateCamera();
    const fov=Math.PI/4, aspect=canvas.width/canvas.height;
    const c=Math.cos(now), s=Math.sin(now);
    const model=new Float32Array([c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1]);
    const udata=new Float32Array([
      pos[0],pos[1],pos[2],1,  tgt[0],tgt[1],tgt[2],1,
      up[0], up[1], up[2],0,  5,-5,-5,1,
      fov,aspect,0,0,  ...model
    ]);
    device.queue.writeBuffer(uniformBuf,0,udata);
    // draw
    const cmd=device.createCommandEncoder();
    const pass=cmd.beginRenderPass({ colorAttachments:[{
      view:context.getCurrentTexture().createView(),
      loadOp:"clear", clearValue:{r:0.6,g:0.8,b:1,a:1}, storeOp:"store"
    }]});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0,bg);
    pass.setVertexBuffer(0,quadBuf);
    pass.draw(6);
    pass.end();
    device.queue.submit([cmd.finish()]);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch(console.error);