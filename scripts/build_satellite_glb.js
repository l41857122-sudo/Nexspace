const { Document, NodeIO } = require('@gltf-transform/core');
const fs = require('fs');

function buildSatelliteGLB() {
  const doc = new Document();
  const buffer = doc.createBuffer('satellite_buffer');
  const scene = doc.createScene('Scene');

  // Materials
  const matBus = doc.createMaterial('Bus-Metallic')
    .setBaseColorFactor([0.26, 0.29, 0.34, 1.0])
    .setMetallicFactor(0.82)
    .setRoughnessFactor(0.32);

  const matSolar = doc.createMaterial('Solar-Photovoltaic')
    .setBaseColorFactor([0.03, 0.06, 0.18, 1.0])
    .setMetallicFactor(0.35)
    .setRoughnessFactor(0.22);

  const matOptics = doc.createMaterial('Optical-Sensor')
    .setBaseColorFactor([0.09, 0.10, 0.13, 1.0])
    .setMetallicFactor(0.75)
    .setRoughnessFactor(0.35);

  const matAntenna = doc.createMaterial('Antenna-Dish')
    .setBaseColorFactor([0.80, 0.82, 0.86, 1.0])
    .setMetallicFactor(0.85)
    .setRoughnessFactor(0.25);

  const matGold = doc.createMaterial('Thermal-Gold')
    .setBaseColorFactor([0.88, 0.66, 0.18, 1.0])
    .setMetallicFactor(0.92)
    .setRoughnessFactor(0.24);

  const matTitanium = doc.createMaterial('Titanium-Bracket')
    .setBaseColorFactor([0.48, 0.51, 0.56, 1.0])
    .setMetallicFactor(0.88)
    .setRoughnessFactor(0.28);

  const matLens = doc.createMaterial('Lens-Glass')
    .setBaseColorFactor([0.04, 0.07, 0.12, 1.0])
    .setMetallicFactor(0.15)
    .setRoughnessFactor(0.08);

  // Geometry Helper Functions
  function createGeom() {
    return { positions: [], normals: [], uvs: [], indices: [] };
  }

  function addBox(target, w, h, d, ox = 0, oy = 0, oz = 0, rx = 0, ry = 0, rz = 0) {
    const hw = w / 2, hh = h / 2, hd = d / 2;
    const baseIdx = target.positions.length / 3;

    function rot(x, y, z) {
      let y1 = y * Math.cos(rx) - z * Math.sin(rx);
      let z1 = y * Math.sin(rx) + z * Math.cos(rx);
      let x2 = x * Math.cos(ry) + z1 * Math.sin(ry);
      let z2 = -x * Math.sin(ry) + z1 * Math.cos(ry);
      let x3 = x2 * Math.cos(rz) - y1 * Math.sin(rz);
      let y3 = x2 * Math.sin(rz) + y1 * Math.cos(rz);
      return [x3 + ox, y3 + oy, z2 + oz];
    }

    function rotNorm(nx, ny, nz) {
      let y1 = ny * Math.cos(rx) - nz * Math.sin(rx);
      let z1 = ny * Math.sin(rx) + nz * Math.cos(rx);
      let x2 = nx * Math.cos(ry) + z1 * Math.sin(ry);
      let z2 = -nx * Math.sin(ry) + z1 * Math.cos(ry);
      let x3 = x2 * Math.cos(rz) - y1 * Math.sin(rz);
      let y3 = x2 * Math.sin(rz) + y1 * Math.cos(rz);
      return [x3, y3, z2];
    }

    const faces = [
      // +Z
      [-hw,-hh, hd, 0,0,1, 0,0],[ hw,-hh, hd, 0,0,1, 1,0],[ hw, hh, hd, 0,0,1, 1,1],[-hw, hh, hd, 0,0,1, 0,1],
      // -Z
      [ hw,-hh,-hd, 0,0,-1, 0,0],[-hw,-hh,-hd, 0,0,-1, 1,0],[-hw, hh,-hd, 0,0,-1, 1,1],[ hw, hh,-hd, 0,0,-1, 0,1],
      // +X
      [ hw,-hh, hd, 1,0,0, 0,0],[ hw,-hh,-hd, 1,0,0, 1,0],[ hw, hh,-hd, 1,0,0, 1,1],[ hw, hh, hd, 1,0,0, 0,1],
      // -X
      [-hw,-hh,-hd,-1,0,0, 0,0],[-hw,-hh, hd,-1,0,0, 1,0],[-hw, hh, hd,-1,0,0, 1,1],[-hw, hh,-hd,-1,0,0, 0,1],
      // +Y
      [-hw, hh, hd, 0,1,0, 0,0],[ hw, hh, hd, 0,1,0, 1,0],[ hw, hh,-hd, 0,1,0, 1,1],[-hw, hh,-hd, 0,1,0, 0,1],
      // -Y
      [-hw,-hh,-hd, 0,-1,0, 0,0],[ hw,-hh,-hd, 0,-1,0, 1,0],[ hw,-hh, hd, 0,-1,0, 1,1],[-hw,-hh, hd, 0,-1,0, 0,1],
    ];

    for (const f of faces) {
      const p = rot(f[0], f[1], f[2]);
      const n = rotNorm(f[3], f[4], f[5]);
      target.positions.push(p[0], p[1], p[2]);
      target.normals.push(n[0], n[1], n[2]);
      target.uvs.push(f[6], f[7]);
    }

    for (let i = 0; i < 6; i++) {
      const idx = baseIdx + i * 4;
      target.indices.push(idx, idx + 1, idx + 2, idx, idx + 2, idx + 3);
    }
  }

  function addCylinder(target, rTop, rBottom, height, segs = 16, ox = 0, oy = 0, oz = 0, rx = 0, ry = 0, rz = 0) {
    const hh = height / 2;
    const baseIdx = target.positions.length / 3;

    function rot(x, y, z) {
      let y1 = y * Math.cos(rx) - z * Math.sin(rx);
      let z1 = y * Math.sin(rx) + z * Math.cos(rx);
      let x2 = x * Math.cos(ry) + z1 * Math.sin(ry);
      let z2 = -x * Math.sin(ry) + z1 * Math.cos(ry);
      let x3 = x2 * Math.cos(rz) - y1 * Math.sin(rz);
      let y3 = x2 * Math.sin(rz) + y1 * Math.cos(rz);
      return [x3 + ox, y3 + oy, z2 + oz];
    }

    function rotNorm(nx, ny, nz) {
      let y1 = ny * Math.cos(rx) - nz * Math.sin(rx);
      let z1 = ny * Math.sin(rx) + nz * Math.cos(rx);
      let x2 = nx * Math.cos(ry) + z1 * Math.sin(ry);
      let z2 = -nx * Math.sin(ry) + z1 * Math.cos(ry);
      let x3 = x2 * Math.cos(rz) - y1 * Math.sin(rz);
      let y3 = x2 * Math.sin(rz) + y1 * Math.cos(rz);
      return [x3, y3, z2];
    }

    // Side
    for (let s = 0; s <= segs; s++) {
      const theta = (s / segs) * Math.PI * 2;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);
      const u = s / segs;

      const pTop = rot(rTop * cosT, hh, rTop * sinT);
      const nSide = rotNorm(cosT, (rBottom - rTop) / height, sinT);
      const pBot = rot(rBottom * cosT, -hh, rBottom * sinT);

      target.positions.push(pTop[0], pTop[1], pTop[2]);
      target.normals.push(nSide[0], nSide[1], nSide[2]);
      target.uvs.push(u, 1);

      target.positions.push(pBot[0], pBot[1], pBot[2]);
      target.normals.push(nSide[0], nSide[1], nSide[2]);
      target.uvs.push(u, 0);
    }

    for (let s = 0; s < segs; s++) {
      const i0 = baseIdx + s * 2;
      const i1 = i0 + 1;
      const i2 = i0 + 2;
      const i3 = i0 + 3;
      target.indices.push(i0, i1, i2, i1, i3, i2);
    }

    // Top cap
    if (rTop > 0.001) {
      const topCenterIdx = target.positions.length / 3;
      const topCenter = rot(0, hh, 0);
      const topNorm = rotNorm(0, 1, 0);
      target.positions.push(topCenter[0], topCenter[1], topCenter[2]);
      target.normals.push(topNorm[0], topNorm[1], topNorm[2]);
      target.uvs.push(0.5, 0.5);

      for (let s = 0; s <= segs; s++) {
        const theta = (s / segs) * Math.PI * 2;
        const pt = rot(rTop * Math.cos(theta), hh, rTop * Math.sin(theta));
        target.positions.push(pt[0], pt[1], pt[2]);
        target.normals.push(topNorm[0], topNorm[1], topNorm[2]);
        target.uvs.push(0.5 + 0.5 * Math.cos(theta), 0.5 + 0.5 * Math.sin(theta));
      }

      for (let s = 0; s < segs; s++) {
        target.indices.push(topCenterIdx, topCenterIdx + 1 + s, topCenterIdx + 2 + s);
      }
    }

    // Bottom cap
    if (rBottom > 0.001) {
      const botCenterIdx = target.positions.length / 3;
      const botCenter = rot(0, -hh, 0);
      const botNorm = rotNorm(0, -1, 0);
      target.positions.push(botCenter[0], botCenter[1], botCenter[2]);
      target.normals.push(botNorm[0], botNorm[1], botNorm[2]);
      target.uvs.push(0.5, 0.5);

      for (let s = 0; s <= segs; s++) {
        const theta = (s / segs) * Math.PI * 2;
        const pt = rot(rBottom * Math.cos(theta), -hh, rBottom * Math.sin(theta));
        target.positions.push(pt[0], pt[1], pt[2]);
        target.normals.push(botNorm[0], botNorm[1], botNorm[2]);
        target.uvs.push(0.5 + 0.5 * Math.cos(theta), 0.5 + 0.5 * Math.sin(theta));
      }

      for (let s = 0; s < segs; s++) {
        target.indices.push(botCenterIdx, botCenterIdx + 2 + s, botCenterIdx + 1 + s);
      }
    }
  }

  // Create geometry buckets for each material
  const busGeom = createGeom();
  const solarGeom = createGeom();
  const opticsGeom = createGeom();
  const antennaGeom = createGeom();
  const goldGeom = createGeom();
  const titaniumGeom = createGeom();
  const lensGeom = createGeom();

  // =========================================================================
  // 1. CENTRAL SPACECRAFT BUS (Authentic Hexagonal/Octagonal Service & Payload Bay)
  // =========================================================================
  // Main octagonal bus structure (height 0.44, radius 0.125)
  addCylinder(busGeom, 0.125, 0.125, 0.44, 8, 0, 0.02, 0);

  // Structural upper & lower equipment collar rings
  addCylinder(titaniumGeom, 0.138, 0.138, 0.025, 8, 0, 0.23, 0);
  addCylinder(titaniumGeom, 0.138, 0.138, 0.025, 8, 0, -0.19, 0);
  addCylinder(titaniumGeom, 0.132, 0.132, 0.015, 8, 0, 0.04, 0);

  // Lower Service & Propulsion module wrapped in Kapton Gold MLI Foil
  addCylinder(goldGeom, 0.126, 0.126, 0.18, 8, 0, -0.09, 0);

  // Upper payload deck plate
  addBox(busGeom, 0.22, 0.015, 0.22, 0, 0.245, 0);

  // =========================================================================
  // 2. DUAL SOLAR WINGS (Two Wings, Multi-Panel, Realistic Cant Angle)
  // =========================================================================
  // Wingspan along X axis. Realistic sun-tracking tilt angle: rotX = 0.28 (~16 degrees)
  const solarTiltX = 0.28;

  // Left & Right Solar Array Drive Mechanisms (SADM)
  addCylinder(titaniumGeom, 0.026, 0.026, 0.08, 12, -0.16, 0.04, 0, 0, 0, Math.PI / 2);
  addCylinder(titaniumGeom, 0.026, 0.026, 0.08, 12,  0.16, 0.04, 0, 0, 0, Math.PI / 2);

  // Structural yoke truss extending from SADM to inner panel
  addCylinder(titaniumGeom, 0.010, 0.010, 0.14, 8, -0.24, 0.04, 0, 0, 0, Math.PI / 2);
  addCylinder(titaniumGeom, 0.010, 0.010, 0.14, 8,  0.24, 0.04, 0, 0, 0, Math.PI / 2);

  // Panel dimensions: 2 segments per wing
  const segW = 0.36;
  const segH = 0.22;
  const segD = 0.008;

  // --- LEFT WING (Negative X) ---
  // Inner panel 1
  addBox(solarGeom, segW, segH, segD, -0.49, 0.04, 0, solarTiltX, 0, 0);
  addBox(titaniumGeom, segW + 0.015, 0.006, segD * 1.5, -0.49, 0.04 + segH/2, 0, solarTiltX, 0, 0);
  addBox(titaniumGeom, segW + 0.015, 0.006, segD * 1.5, -0.49, 0.04 - segH/2, 0, solarTiltX, 0, 0);

  // Inter-panel hinge bracket
  addBox(titaniumGeom, 0.025, 0.04, 0.018, -0.68, 0.04, 0, solarTiltX, 0, 0);

  // Outer panel 2
  addBox(solarGeom, segW, segH, segD, -0.87, 0.04, 0, solarTiltX, 0, 0);
  addBox(titaniumGeom, segW + 0.015, 0.006, segD * 1.5, -0.87, 0.04 + segH/2, 0, solarTiltX, 0, 0);
  addBox(titaniumGeom, segW + 0.015, 0.006, segD * 1.5, -0.87, 0.04 - segH/2, 0, solarTiltX, 0, 0);

  // --- RIGHT WING (Positive X) ---
  // Inner panel 1
  addBox(solarGeom, segW, segH, segD, 0.49, 0.04, 0, solarTiltX, 0, 0);
  addBox(titaniumGeom, segW + 0.015, 0.006, segD * 1.5, 0.49, 0.04 + segH/2, 0, solarTiltX, 0, 0);
  addBox(titaniumGeom, segW + 0.015, 0.006, segD * 1.5, 0.49, 0.04 - segH/2, 0, solarTiltX, 0, 0);

  // Inter-panel hinge bracket
  addBox(titaniumGeom, 0.025, 0.04, 0.018, 0.68, 0.04, 0, solarTiltX, 0, 0);

  // Outer panel 2
  addBox(solarGeom, segW, segH, segD, 0.87, 0.04, 0, solarTiltX, 0, 0);
  addBox(titaniumGeom, segW + 0.015, 0.006, segD * 1.5, 0.87, 0.04 + segH/2, 0, solarTiltX, 0, 0);
  addBox(titaniumGeom, segW + 0.015, 0.006, segD * 1.5, 0.87, 0.04 - segH/2, 0, solarTiltX, 0, 0);

  // =========================================================================
  // 3. EARTH-FACING OPTICAL TELESCOPE & REMOTE SENSING PAYLOAD
  // =========================================================================
  // Forward Earth-pointing payload aimed toward +Z with slight down-pitch (-0.12 rad)
  const opticPitchX = Math.PI / 2 - 0.12;

  // Main telescope barrel (cylindrical housing)
  addCylinder(opticsGeom, 0.065, 0.075, 0.20, 20, 0, 0.06, 0.16, opticPitchX, 0, 0);

  // Multi-stage external stray-light baffle cone & hood
  addCylinder(opticsGeom, 0.082, 0.068, 0.07, 20, 0, 0.08, 0.27, opticPitchX, 0, 0);

  // Front aperture bezel ring (titanium retention flange)
  addCylinder(titaniumGeom, 0.085, 0.085, 0.014, 20, 0, 0.085, 0.305, opticPitchX, 0, 0);

  // Recessed dark optical glass front lens element
  addCylinder(lensGeom, 0.068, 0.068, 0.008, 20, 0, 0.083, 0.295, opticPitchX, 0, 0);

  // Secondary mirror spider hub inside the aperture
  addCylinder(titaniumGeom, 0.016, 0.016, 0.02, 12, 0, 0.084, 0.298, opticPitchX, 0, 0);
  addBox(titaniumGeom, 0.065, 0.003, 0.01, 0, 0.084, 0.298, opticPitchX, 0, 0);
  addBox(titaniumGeom, 0.003, 0.065, 0.01, 0, 0.084, 0.298, opticPitchX, 0, 0);

  // Focal plane electronics radiator on the anti-sun side
  addBox(busGeom, 0.12, 0.08, 0.018, 0, -0.06, 0.14, 0, 0, 0);

  // =========================================================================
  // 4. HIGH-GAIN PARABOLIC COMMUNICATIONS DISH ANTENNA
  // =========================================================================
  // Parabolic dish mounted on upper-rear deck on articulated gimbal boom
  const dishPitchX = -0.55;
  const dishYawY = 0.35;

  // Gimbal boom struts
  addCylinder(titaniumGeom, 0.010, 0.010, 0.15, 8, 0.08, 0.31, -0.06, dishPitchX, dishYawY, 0);

  // Parabolic reflector dish (bowl shape with inner & outer radii)
  addCylinder(antennaGeom, 0.135, 0.035, 0.045, 24, 0.11, 0.38, -0.08, dishPitchX, dishYawY, 0);
  addCylinder(titaniumGeom, 0.138, 0.138, 0.008, 24, 0.11, 0.39, -0.08, dishPitchX, dishYawY, 0);

  // Antenna central feed horn & sub-reflector tripod
  addCylinder(goldGeom, 0.012, 0.022, 0.055, 12, 0.11, 0.42, -0.09, dishPitchX, dishYawY, 0);
  addCylinder(titaniumGeom, 0.004, 0.004, 0.06, 6, 0.11, 0.44, -0.09, dishPitchX, dishYawY, 0);

  // =========================================================================
  // 5. STAR TRACKERS, SENSORS & RCS ATTITUDE THRUSTERS
  // =========================================================================
  // Dual optical star tracker sensor cones on anti-sun side
  addCylinder(opticsGeom, 0.022, 0.012, 0.045, 12, -0.06, 0.18, -0.11, -0.65, -0.45, 0);
  addCylinder(opticsGeom, 0.022, 0.012, 0.045, 12,  0.06, 0.18, -0.11, -0.65,  0.45, 0);

  // 4 Quad-Thruster RCS Attitude Control blocks on lower bus corners
  const rcsCorners = [
    [ 0.095, -0.16,  0.095],
    [-0.095, -0.16,  0.095],
    [ 0.095, -0.16, -0.095],
    [-0.095, -0.16, -0.095],
  ];

  for (const [cx, cy, cz] of rcsCorners) {
    addBox(busGeom, 0.025, 0.025, 0.025, cx, cy, cz);
    const nx = cx > 0 ? 1 : -1;
    const nz = cz > 0 ? 1 : -1;
    addCylinder(goldGeom, 0.012, 0.005, 0.025, 8, cx + nx * 0.015, cy - 0.018, cz + nz * 0.015, 0.4 * nz, 0, -0.4 * nx);
  }

  // Magnetometer / Science instrument boom extending from lower deck
  addCylinder(titaniumGeom, 0.006, 0.006, 0.22, 8, 0, -0.28, -0.06, 0.25, 0, 0);
  addBox(goldGeom, 0.035, 0.035, 0.035, 0, -0.38, -0.09);

  // =========================================================================
  // 6. BUILD GLTF DOCUMENT & PRIMITIVES
  // =========================================================================
  const mesh = doc.createMesh('NexSpace_Satellite_Mesh');

  function attachPrimitive(geom, material) {
    if (geom.positions.length === 0) return;

    const posAccessor = doc.createAccessor()
      .setType('VEC3')
      .setArray(new Float32Array(geom.positions))
      .setBuffer(buffer);

    const normAccessor = doc.createAccessor()
      .setType('VEC3')
      .setArray(new Float32Array(geom.normals))
      .setBuffer(buffer);

    const uvAccessor = doc.createAccessor()
      .setType('VEC2')
      .setArray(new Float32Array(geom.uvs))
      .setBuffer(buffer);

    const indAccessor = doc.createAccessor()
      .setType('SCALAR')
      .setArray(new Uint16Array(geom.indices))
      .setBuffer(buffer);

    const prim = doc.createPrimitive()
      .setAttribute('POSITION', posAccessor)
      .setAttribute('NORMAL', normAccessor)
      .setAttribute('TEXCOORD_0', uvAccessor)
      .setIndices(indAccessor)
      .setMaterial(material);

    mesh.addPrimitive(prim);
  }

  attachPrimitive(busGeom, matBus);
  attachPrimitive(solarGeom, matSolar);
  attachPrimitive(opticsGeom, matOptics);
  attachPrimitive(antennaGeom, matAntenna);
  attachPrimitive(goldGeom, matGold);
  attachPrimitive(titaniumGeom, matTitanium);
  attachPrimitive(lensGeom, matLens);

  const node = doc.createNode('Satellite_Node').setMesh(mesh);
  scene.addChild(node);

  return doc;
}

async function run() {
  const doc = buildSatelliteGLB();
  const io = new NodeIO();
  const glb = await io.writeBinary(doc);
  fs.writeFileSync('public/models/nexspace-satellite.glb', Buffer.from(glb));
  console.log('Successfully built authentic Earth Observation Satellite GLB!');
  console.log('File size:', glb.byteLength, 'bytes');
}

run().catch(console.error);
