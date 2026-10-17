"use client";

import React, { useEffect, useRef } from "react";
import {
  Renderer,
  Camera,
  Transform,
  Mesh,
  Sphere,
  Program,
  Texture,
  GLTFLoader,
  Geometry,
} from "ogl";

interface EarthSceneProps {
  scrollProgress: number;
  earthScale?: number;
  earthAlpha?: number;
  satelliteAlpha?: number;
  camZ?: number;
  camX?: number;
  camY?: number;
  isReducedMotion?: boolean;
  mouseX?: number;
  mouseY?: number;
  globeMode?: "particle" | "realistic";
  onDragStateChange?: (isDragging: boolean) => void;
}

// =========================================================================
// 1. Earth Surface Shaders (Realistic fallback)
// =========================================================================
const earthVertexShader = `#version 300 es
in vec3 position;
in vec3 normal;
in vec2 uv;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

out vec3 vNormal;
out vec3 vPosition;
out vec2 vUv;

void main() {
  vUv = uv;
  vNormal = normalize(normalMatrix * normal);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vPosition = mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const earthFragmentShader = `#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vPosition;
in vec2 vUv;

uniform sampler2D uDayMap;
uniform sampler2D uNightMap;
uniform float uHasTextures;
uniform float uTime;
uniform float uAlpha;
uniform vec3 uSunDirection;

out vec4 fragColor;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(-vPosition);
  vec3 L = normalize(uSunDirection);
  vec3 H = normalize(L + V);

  float NdotL = dot(N, L);
  float dayTerm = smoothstep(-0.12, 0.22, NdotL);
  float nightTerm = 1.0 - dayTerm;

  vec4 dayTex = texture(uDayMap, vUv);
  vec4 nightTex = texture(uNightMap, vUv);

  if (uHasTextures < 0.5) {
    dayTex = vec4(0.03, 0.14, 0.30, 1.0);
    nightTex = vec4(0.0);
  }

  float isOcean = smoothstep(0.42, 0.20, (dayTex.r + dayTex.g) * 0.5);
  float NdotH = max(0.0, dot(N, H));
  float oceanSpecular = pow(NdotH, 96.0) * isOcean * dayTerm * 0.92;
  vec3 sunGlint = vec3(1.0, 0.96, 0.90) * oceanSpecular;

  vec3 litDay = (dayTex.rgb * max(0.025, dayTerm)) + sunGlint;
  vec3 goldenCityLights = nightTex.rgb * nightTerm * 2.4;

  float fresnel = 1.0 - max(0.0, dot(N, V));
  float atmoLimb = pow(fresnel, 3.8) * max(0.0, NdotL + 0.18) * 1.35;
  vec3 atmoRayleighColor = vec3(0.06, 0.44, 0.82) * atmoLimb;
  vec3 outerGlow = vec3(0.02, 0.22, 0.48) * pow(fresnel, 5.2) * 0.25;

  vec3 finalColor = litDay + goldenCityLights + atmoRayleighColor + outerGlow;
  fragColor = vec4(finalColor, uAlpha);
}
`;

const cloudVertexShader = `#version 300 es
in vec3 position;
in vec3 normal;
in vec2 uv;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

out vec3 vNormal;
out vec3 vPosition;
out vec2 vUv;

void main() {
  vUv = uv;
  vNormal = normalize(normalMatrix * normal);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vPosition = mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const cloudFragmentShader = `#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vPosition;
in vec2 vUv;

uniform sampler2D uCloudMap;
uniform vec3 uSunDirection;
uniform float uHasClouds;
uniform float uAlpha;

out vec4 fragColor;

void main() {
  if (uHasClouds < 0.5) discard;
  vec3 N = normalize(vNormal);
  vec3 V = normalize(-vPosition);
  vec3 L = normalize(uSunDirection);

  vec4 cloudSample = texture(uCloudMap, vUv);
  float cloudDensity = cloudSample.r;
  if (cloudDensity < 0.05) discard;

  float NdotL = dot(N, L);
  float dayTerm = smoothstep(-0.08, 0.25, NdotL);
  vec3 cloudSunlit = vec3(0.96, 0.98, 1.0) * (0.12 + 0.88 * dayTerm);

  float fresnel = 1.0 - max(0.0, dot(N, V));
  vec3 cloudRim = vec3(0.15, 0.75, 1.0) * pow(fresnel, 3.8) * max(0.0, NdotL + 0.2);

  float alpha = cloudDensity * 0.85 * uAlpha * (0.08 + 0.92 * dayTerm);
  fragColor = vec4(cloudSunlit + cloudRim, alpha);
}
`;

// =========================================================================
// 2. Scrolltide Particle Globe Shaders (High-Luminosity Additive Glowing Points)
// =========================================================================
const particleVertexShader = `#version 300 es
in vec3 position;
in vec3 color;
in float size;
in float phase;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float uTime;
uniform float uAlpha;
uniform float uPointSize;
uniform float uPixelRatio;

out vec3 vColor;
out float vAlpha;

void main() {
  vColor = color;
  
  // Organic twinkling phase oscillation
  float twinkle = sin(uTime * 2.5 + phase) * 0.15 + 0.90;
  
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  
  // Distance attenuation
  float distZ = -mvPosition.z;
  float pSize = size * uPointSize * uPixelRatio * 3.8;
  gl_PointSize = clamp((pSize * 4.6 / max(1.0, distZ)) * twinkle, 1.8, 16.0);
  
  // Back-hemisphere depth culling: only front-facing hemisphere shines brightly
  // Camera is at z = 4.6, sphere center at z = 0, so mvPosition.z is -4.6 at sphere center
  float depthFade = smoothstep(-4.68, -4.20, mvPosition.z);
  vAlpha = depthFade * uAlpha;
  
  gl_Position = projectionMatrix * mvPosition;
}
`;

const particleFragmentShader = `#version 300 es
precision highp float;

in vec3 vColor;
in float vAlpha;

uniform float uAlpha;

out vec4 fragColor;

void main() {
  vec2 coord = gl_PointCoord - vec2(0.5);
  float dist = length(coord);
  if (dist > 0.5) discard;
  
  // Smooth Gaussian glowing particle with crisp core
  float halo = exp(-dist * dist * 16.0);
  float core = exp(-dist * dist * 48.0);
  
  vec3 hotWhite = vec3(1.0, 1.0, 1.0);
  vec3 col = mix(vColor * 1.5, hotWhite, core * 0.80);
  float intensity = (halo * 0.75 + core * 0.55) * vAlpha * uAlpha;
  
  fragColor = vec4(col, clamp(intensity, 0.0, 1.0));
}
`;

// Deep Obsidian Dark Core Sphere (drawn inside particles as a dark contrast base)
const coreVertexShader = `#version 300 es
in vec3 position;
in vec3 normal;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

out vec3 vNormal;
out vec3 vPosition;

void main() {
  vNormal = normalize(normalMatrix * normal);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vPosition = mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const coreFragmentShader = `#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vPosition;

uniform float uAlpha;

out vec4 fragColor;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(-vPosition);
  
  float fresnel = 1.0 - max(0.0, dot(N, V));
  // Rich midnight cosmic dark blue base
  vec3 deepSpace = vec3(0.008, 0.018, 0.038);
  vec3 rimBlue = vec3(0.0, 0.28, 0.55) * pow(fresnel, 4.0);
  
  fragColor = vec4((deepSpace + rimBlue) * uAlpha, 1.0);
}
`;

// Subtle Atmospheric Cyan Rim Halo
const atmoRimVertexShader = `#version 300 es
in vec3 position;
in vec3 normal;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

out vec3 vNormal;
out vec3 vPosition;

void main() {
  vNormal = normalize(normalMatrix * normal);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vPosition = mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const atmoRimFragmentShader = `#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vPosition;

uniform float uAlpha;

out vec4 fragColor;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(-vPosition);
  
  float fresnel = pow(1.0 - max(0.0, dot(N, V)), 4.2);
  vec3 rim = vec3(0.0, 0.82, 1.0) * fresnel * 0.70;
  
  fragColor = vec4(rim, fresnel * 0.70 * uAlpha);
}
`;

// Orbital Rings Shaders
const ringVertexShader = `#version 300 es
in vec3 position;
in float ringU;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out float vRingU;

void main() {
  vRingU = ringU;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
}
`;

const ringFragmentShader = `#version 300 es
precision highp float;

in float vRingU;

uniform float uTime;
uniform float uAlpha;
uniform vec3 uColor;

out vec4 fragColor;

void main() {
  float dash = sin(vRingU * 48.0 - uTime * 2.8);
  float pulse = smoothstep(0.1, 0.9, dash) * 0.55 + 0.40;
  vec3 col = uColor * pulse * 1.5;
  fragColor = vec4(col, uAlpha * pulse * 0.85);
}
`;

function createRingGeometry(gl: any, radius: number, segments = 160) {
  const positions = new Float32Array(segments * 3);
  const ringUs = new Float32Array(segments);
  for (let i = 0; i < segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = 0;
    positions[i * 3 + 2] = Math.sin(angle) * radius;
    ringUs[i] = i / segments;
  }
  return new Geometry(gl, {
    position: { size: 3, data: positions },
    ringU: { size: 1, data: ringUs },
  });
}

// Satellite Shaders
const satellitePBRVertexShader = `#version 300 es
in vec3 position;
in vec3 normal;
in vec2 uv;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

out vec3 vNormal;
out vec3 vPosition;
out vec2 vUv;

void main() {
  vUv = uv;
  vNormal = normalize(normalMatrix * normal);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vPosition = mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const satellitePBRFragmentShader = `#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vPosition;
in vec2 vUv;

uniform vec3 uSunDirection;
uniform float uAlpha;
uniform vec3 uBaseColor;
uniform float uMetallic;
uniform float uRoughness;
uniform float uMaterialType;

out vec4 fragColor;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(-vPosition);
  vec3 L = normalize(uSunDirection);
  vec3 H = normalize(L + V);

  float NdotL = max(0.0, dot(N, L));
  float NdotV = max(0.0, dot(N, V));
  float NdotH = max(0.0, dot(N, H));

  vec3 albedo = uBaseColor;

  if (uMaterialType > 0.5 && uMaterialType < 1.5) {
    vec2 cellCoord = fract(vUv * vec2(12.0, 8.0));
    float gridLine = step(0.91, cellCoord.x) + step(0.91, cellCoord.y);
    float busbar = step(0.48, cellCoord.y) * step(cellCoord.y, 0.52);
    vec3 cellColor = vec3(0.025, 0.055, 0.16);
    vec3 busbarColor = vec3(0.72, 0.78, 0.86);
    albedo = mix(cellColor, busbarColor, max(gridLine * 0.4, busbar * 0.7));
  } else if (uMaterialType > 2.5 && uMaterialType < 3.5) {
    float crinkle = sin(vPosition.x * 120.0) * cos(vPosition.y * 120.0) * 0.06;
    albedo = vec3(0.88 + crinkle, 0.66 + crinkle, 0.18);
  }

  vec3 sunColor = vec3(1.0, 0.98, 0.94);
  vec3 directSun = sunColor * NdotL * 1.35;
  vec3 ambient = vec3(0.035, 0.045, 0.065);

  float shininess = mix(16.0, 110.0, 1.0 - uRoughness);
  float specPower = pow(NdotH, shininess);
  vec3 specColor = mix(vec3(0.95), albedo, uMetallic);
  vec3 specular = specColor * specPower * (NdotL > 0.0 ? 1.5 : 0.0);

  float rim = pow(1.0 - NdotV, 3.8);
  vec3 edgeRim = sunColor * rim * 0.25 * max(0.05, NdotL + 0.1);

  vec3 finalColor = (albedo * (ambient + directSun)) + specular + edgeRim;
  fragColor = vec4(finalColor, uAlpha);
}
`;

function getMaterialParams(name: string) {
  const n = (name || "").toLowerCase();
  let baseColor: [number, number, number] = [0.26, 0.29, 0.34];
  let metallic = 0.82;
  let roughness = 0.32;
  let materialType = 0.0;

  if (n.includes("solar") || n.includes("photovoltaic")) {
    materialType = 1.0;
    baseColor = [0.03, 0.06, 0.18];
    metallic = 0.35;
    roughness = 0.22;
  } else if (n.includes("thermal") || n.includes("gold") || n.includes("foil")) {
    materialType = 3.0;
    baseColor = [0.88, 0.66, 0.18];
    metallic = 0.92;
    roughness = 0.24;
  } else {
    materialType = 0.0;
    baseColor = [0.26, 0.29, 0.34];
    metallic = 0.82;
    roughness = 0.32;
  }

  return { baseColor, metallic, roughness, materialType };
}

// =========================================================================
// 3. Component Implementation
// =========================================================================
export default function EarthScene({
  scrollProgress,
  earthScale = 1.0,
  earthAlpha = 1.0,
  satelliteAlpha = 1.0,
  camZ,
  camX,
  camY,
  isReducedMotion = false,
  mouseX = 0,
  mouseY = 0,
  globeMode = "particle",
  onDragStateChange,
}: EarthSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const animFrameRef = useRef<number | null>(null);

  const earthMeshRef = useRef<Mesh | null>(null);
  const cloudMeshRef = useRef<Mesh | null>(null);
  const particleMeshRef = useRef<Mesh | null>(null);
  const innerCoreMeshRef = useRef<Mesh | null>(null);
  const atmoRimMeshRef = useRef<Mesh | null>(null);
  const ringMesh1Ref = useRef<Mesh | null>(null);
  const ringMesh2Ref = useRef<Mesh | null>(null);
  const ringMesh3Ref = useRef<Mesh | null>(null);

  const earthTiltGroupRef = useRef<Transform | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  const satelliteGroupRef = useRef<Transform | null>(null);

  const earthProgramRef = useRef<Program | null>(null);
  const cloudProgramRef = useRef<Program | null>(null);
  const particleProgramRef = useRef<Program | null>(null);
  const coreProgramRef = useRef<Program | null>(null);
  const atmoRimProgramRef = useRef<Program | null>(null);
  const ringProgram1Ref = useRef<Program | null>(null);
  const ringProgram2Ref = useRef<Program | null>(null);
  const ringProgram3Ref = useRef<Program | null>(null);
  const satelliteProgramsRef = useRef<Program[]>([]);
  const rendererRef = useRef<Renderer | null>(null);

  // Drag-to-spin momentum state (Initial rotation = 5.25 for exact Americas & Europe/Africa mockup layout)
  const dragRef = useRef({
    isDragging: false,
    pointerId: -1,
    lastX: 0,
    lastY: 0,
    lastTime: 0,
    rotY: 5.25,
    rotX: 0.08,
    velY: 0,
    velX: 0,
  });

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const d = dragRef.current;
    d.isDragging = true;
    d.pointerId = e.pointerId;
    d.lastX = e.clientX;
    d.lastY = e.clientY;
    d.lastTime = performance.now();
    d.velY = 0;
    d.velX = 0;
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {}
    if (onDragStateChange) onDragStateChange(true);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = dragRef.current;
    if (!d.isDragging || e.pointerId !== d.pointerId) return;
    const now = performance.now();
    const dt = Math.max(1, now - d.lastTime);
    const dx = e.clientX - d.lastX;
    const dy = e.clientY - d.lastY;

    d.rotY += dx * 0.006;
    d.rotX = Math.max(-0.65, Math.min(0.65, d.rotX + dy * 0.004));

    d.velY = (dx / dt) * 0.048;
    d.velX = (dy / dt) * 0.032;

    d.lastX = e.clientX;
    d.lastY = e.clientY;
    d.lastTime = now;
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = dragRef.current;
    if (d.isDragging && e.pointerId === d.pointerId) {
      d.isDragging = false;
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {}
      if (onDragStateChange) onDragStateChange(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    const container = containerRef.current;
    if (!container) return;

    (window as any).__nexspace_drag = dragRef.current;

    const renderer = new Renderer({
      alpha: false,
      antialias: true,
      dpr: Math.min(window.devicePixelRatio || 1, 2),
      powerPreference: "high-performance",
    });
    rendererRef.current = renderer;
    const gl = renderer.gl;
    container.appendChild(gl.canvas);

    gl.clearColor(0.008, 0.020, 0.045, 1.0);

    const camera = new Camera(gl, { fov: 38 });
    camera.position.set(0, 0, 4.6);
    cameraRef.current = camera;

    const scene = new Transform();

    const earthTiltGroup = new Transform();
    earthTiltGroup.setParent(scene);
    earthTiltGroupRef.current = earthTiltGroup;

    const isMobile = window.innerWidth < 768;
    // Position globe centered on the right matching mockup
    earthTiltGroup.position.set(isMobile ? 0.15 : 0.52, isMobile ? -0.05 : 0.05, 0.0);
    earthTiltGroup.rotation.z = -0.409;
    earthTiltGroup.rotation.x = 0.12;

    const sunDir = [1.3, 0.45, 0.85];

    // 1. Realistic Globe Setup
    const earthGeometry = new Sphere(gl, { radius: 1.25, widthSegments: 80, heightSegments: 80 });
    const cloudGeometry = new Sphere(gl, { radius: 1.256, widthSegments: 80, heightSegments: 80 });

    const dayTexture = new Texture(gl);
    const nightTexture = new Texture(gl);
    const cloudsTexture = new Texture(gl);

    const imgDay = new Image();
    imgDay.src = "/textures/earth_diffuse.jpg";
    imgDay.onload = () => {
      if (!isMounted) return;
      dayTexture.image = imgDay;
      if (earthProgramRef.current) earthProgramRef.current.uniforms.uHasTextures.value = 1.0;
    };

    const imgNight = new Image();
    imgNight.src = "/textures/earth_night.jpg";
    imgNight.onload = () => {
      if (!isMounted) return;
      nightTexture.image = imgNight;
    };

    const imgClouds = new Image();
    imgClouds.src = "/textures/earth_clouds.jpg";
    imgClouds.onload = () => {
      if (!isMounted) return;
      cloudsTexture.image = imgClouds;
      if (cloudProgramRef.current) cloudProgramRef.current.uniforms.uHasClouds.value = 1.0;
    };

    const earthProgram = new Program(gl, {
      vertex: earthVertexShader,
      fragment: earthFragmentShader,
      uniforms: {
        uDayMap: { value: dayTexture },
        uNightMap: { value: nightTexture },
        uHasTextures: { value: 0.0 },
        uTime: { value: 0 },
        uAlpha: { value: 0 },
        uSunDirection: { value: sunDir },
      },
      transparent: true,
      depthTest: true,
    });
    earthProgramRef.current = earthProgram;

    const earthMesh = new Mesh(gl, { geometry: earthGeometry, program: earthProgram });
    earthMesh.setParent(earthTiltGroup);
    earthMeshRef.current = earthMesh;
    earthMesh.rotation.y = 4.10;

    const cloudProgram = new Program(gl, {
      vertex: cloudVertexShader,
      fragment: cloudFragmentShader,
      uniforms: {
        uCloudMap: { value: cloudsTexture },
        uSunDirection: { value: sunDir },
        uHasClouds: { value: 0.0 },
        uAlpha: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
    });
    cloudProgramRef.current = cloudProgram;

    const cloudMesh = new Mesh(gl, { geometry: cloudGeometry, program: cloudProgram });
    cloudMesh.setParent(earthTiltGroup);
    cloudMeshRef.current = cloudMesh;
    cloudMesh.rotation.y = 4.10;

    // 2. Scrolltide Particle Globe Assembly
    // Inner Dark Core Sphere (opaque dark blue backdrop preventing back particles from showing)
    const coreGeometry = new Sphere(gl, { radius: 1.236, widthSegments: 64, heightSegments: 64 });
    const coreProgram = new Program(gl, {
      vertex: coreVertexShader,
      fragment: coreFragmentShader,
      uniforms: {
        uAlpha: { value: 1.0 },
      },
      transparent: false,
      depthWrite: true,
      depthTest: true,
    });
    coreProgramRef.current = coreProgram;

    const coreMesh = new Mesh(gl, { geometry: coreGeometry, program: coreProgram });
    coreMesh.setParent(earthTiltGroup);
    innerCoreMeshRef.current = coreMesh;

    // Luminous Atmospheric Cyan Rim Halo
    const atmoRimGeometry = new Sphere(gl, { radius: 1.275, widthSegments: 64, heightSegments: 64 });
    const atmoRimProgram = new Program(gl, {
      vertex: atmoRimVertexShader,
      fragment: atmoRimFragmentShader,
      uniforms: { uAlpha: { value: 1.0 } },
      transparent: true,
      depthWrite: false,
      depthTest: true,
    });
    atmoRimProgram.setBlendFunc(gl.SRC_ALPHA, gl.ONE);
    atmoRimProgramRef.current = atmoRimProgram;

    const atmoRimMesh = new Mesh(gl, { geometry: atmoRimGeometry, program: atmoRimProgram });
    atmoRimMesh.setParent(earthTiltGroup);
    atmoRimMeshRef.current = atmoRimMesh;

    // 3 Concentric Orbital Telemetry Rings
    const ringGeo1 = createRingGeometry(gl, 1.55, 160);
    const ringProg1 = new Program(gl, {
      vertex: ringVertexShader,
      fragment: ringFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uAlpha: { value: 0.85 },
        uColor: { value: [0.0, 0.90, 1.0] },
      },
      transparent: true,
      depthTest: true,
      depthWrite: false,
    });
    ringProg1.setBlendFunc(gl.SRC_ALPHA, gl.ONE);
    ringProgram1Ref.current = ringProg1;
    const ringMesh1 = new Mesh(gl, { mode: gl.LINE_LOOP, geometry: ringGeo1, program: ringProg1 });
    ringMesh1.rotation.x = 0.52;
    ringMesh1.rotation.z = -0.25;
    ringMesh1.setParent(earthTiltGroup);
    ringMesh1Ref.current = ringMesh1;

    const ringGeo2 = createRingGeometry(gl, 1.78, 160);
    const ringProg2 = new Program(gl, {
      vertex: ringVertexShader,
      fragment: ringFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uAlpha: { value: 0.70 },
        uColor: { value: [0.0, 0.95, 0.90] },
      },
      transparent: true,
      depthTest: true,
      depthWrite: false,
    });
    ringProg2.setBlendFunc(gl.SRC_ALPHA, gl.ONE);
    ringProgram2Ref.current = ringProg2;
    const ringMesh2 = new Mesh(gl, { mode: gl.LINE_LOOP, geometry: ringGeo2, program: ringProg2 });
    ringMesh2.rotation.x = -0.42;
    ringMesh2.rotation.y = 0.35;
    ringMesh2.setParent(earthTiltGroup);
    ringMesh2Ref.current = ringMesh2;

    const ringGeo3 = createRingGeometry(gl, 2.05, 160);
    const ringProg3 = new Program(gl, {
      vertex: ringVertexShader,
      fragment: ringFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uAlpha: { value: 0.55 },
        uColor: { value: [0.1, 0.75, 1.0] },
      },
      transparent: true,
      depthTest: true,
      depthWrite: false,
    });
    ringProg3.setBlendFunc(gl.SRC_ALPHA, gl.ONE);
    ringProgram3Ref.current = ringProg3;
    const ringMesh3 = new Mesh(gl, { mode: gl.LINE_LOOP, geometry: ringGeo3, program: ringProg3 });
    ringMesh3.rotation.x = 0.15;
    ringMesh3.rotation.z = 0.65;
    ringMesh3.setParent(earthTiltGroup);
    ringMesh3Ref.current = ringMesh3;

    // Load High-Density Binary Continental Point Cloud (32,163 points)
    fetch("/data/particle_globe.bin")
      .then((res) => {
        if (!res.ok) throw new Error("Particle globe data fetch failed");
        return res.arrayBuffer();
      })
      .then((buf) => {
        if (!isMounted) return;
        const view = new DataView(buf);
        const count = view.getUint32(0, true);
        let offset = 16;
        const positions = new Float32Array(buf, offset, count * 3);
        offset += count * 3 * 4;
        const colors = new Float32Array(buf, offset, count * 3);
        offset += count * 3 * 4;
        const sizes = new Float32Array(buf, offset, count);
        offset += count * 4;
        const phases = new Float32Array(buf, offset, count);

        const pGeo = new Geometry(gl, {
          position: { size: 3, data: positions },
          color: { size: 3, data: colors },
          size: { size: 1, data: sizes },
          phase: { size: 1, data: phases },
        });

        const pProg = new Program(gl, {
          vertex: particleVertexShader,
          fragment: particleFragmentShader,
          uniforms: {
            uTime: { value: 0 },
            uAlpha: { value: 1.0 },
            uPointSize: { value: 1.0 },
            uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
          },
          transparent: true,
          depthTest: true,
          depthWrite: false,
        });
        // CRITICAL ADDITIVE BLENDING FOR LUMINOUS GLOWING CONTINENTS
        pProg.setBlendFunc(gl.SRC_ALPHA, gl.ONE);
        particleProgramRef.current = pProg;

        const pMesh = new Mesh(gl, {
          mode: gl.POINTS,
          geometry: pGeo,
          program: pProg,
        });
        pMesh.setParent(earthTiltGroup);
        particleMeshRef.current = pMesh;
        pMesh.rotation.y = 5.25;
        pMesh.rotation.x = 0.08;
      })
      .catch((err) => {
        console.error("[NexSpace 3D] Failed to load particle globe binary:", err);
      });

    // 3. Canonical 3D Satellite Assembly
    const satGroup = new Transform();
    satGroup.setParent(scene);
    satelliteGroupRef.current = satGroup;

    // Initial position along the outer orbital arc
    satGroup.position.set(1.35, 0.60, 0.85);
    satGroup.rotation.set(0.24, -0.42, -0.16);
    satGroup.scale.set(0.20, 0.20, 0.20);

    GLTFLoader.load(gl, "/models/nexspace-satellite.glb")
      .then((gltf) => {
        if (!isMounted) return;
        satelliteProgramsRef.current = [];

        if (gltf.meshes) {
          gltf.meshes.forEach((mesh) => {
            mesh.primitives.forEach((primitive: { program?: Program & { gltfMaterial?: { name?: string } } }) => {
              const gltfMat = primitive.program?.gltfMaterial || {};
              const matProps = getMaterialParams(gltfMat.name || "");

              const prog = new Program(gl, {
                vertex: satellitePBRVertexShader,
                fragment: satellitePBRFragmentShader,
                uniforms: {
                  uBaseColor: { value: matProps.baseColor },
                  uRoughness: { value: matProps.roughness },
                  uMetallic: { value: matProps.metallic },
                  uMaterialType: { value: matProps.materialType },
                  uSunDirection: { value: [1.2, 0.6, 0.8] },
                  uAlpha: { value: 1.0 },
                },
                transparent: true,
                depthTest: true,
              });

              primitive.program = prog;
              satelliteProgramsRef.current.push(prog);
            });
          });
        }

        if (gltf.scene) {
          gltf.scene.forEach((root: Transform) => root.setParent(satGroup));
        } else if (gltf.scenes && gltf.scenes[0]) {
          gltf.scenes[0].forEach((root: Transform) => root.setParent(satGroup));
        }
      })
      .catch((err) => {
        console.error("[NexSpace 3D] Failed to load canonical satellite GLB:", err);
      });

    function getResponsiveLayout(aspect: number) {
      if (aspect >= 1.45) {
        return {
          earthBaseX: 0.88,
          earthBaseY: 0.00,
          satBaseX: 1.55,
          satBaseY: 0.62,
          satBaseZ: 0.85,
          satScale: 0.20,
        };
      }
      if (aspect >= 1.15) {
        return {
          earthBaseX: 0.62,
          earthBaseY: 0.02,
          satBaseX: 1.25,
          satBaseY: 0.55,
          satBaseZ: 0.85,
          satScale: 0.19,
        };
      }
      if (aspect >= 0.65) {
        return {
          earthBaseX: 0.16,
          earthBaseY: 0.02,
          satBaseX: 0.55,
          satBaseY: 0.65,
          satBaseZ: 0.85,
          satScale: 0.16,
        };
      }
      return {
        earthBaseX: 0.08,
        earthBaseY: -0.02,
        satBaseX: 0.35,
        satBaseY: 0.65,
        satBaseZ: 0.85,
        satScale: 0.14,
      };
    }

    let currentAspect = 1.6;

    function resize() {
      const width = window.innerWidth;
      const height = window.innerHeight;
      if (height === 0) return;
      currentAspect = width / height;
      renderer.setSize(width, height);
      camera.perspective({ aspect: currentAspect });
    }

    window.addEventListener("resize", resize);
    resize();

    let lastTime = performance.now();
    const mountTime = performance.now();

    function render(currentTime: number) {
      const dt = (currentTime - lastTime) / 1000;
      lastTime = currentTime;

      const elapsed = (currentTime - mountTime) / 1000;
      const introFade = Math.min(1.0, elapsed / 0.7);

      const p = Math.min(1, Math.max(0, propsRef.current.scrollProgress));
      const liveAspect =
        typeof window !== "undefined" && window.innerHeight > 0
          ? window.innerWidth / window.innerHeight
          : currentAspect;
      const { earthBaseX, earthBaseY, satBaseX, satBaseY, satBaseZ, satScale } =
        getResponsiveLayout(liveAspect);

      // 1. Camera Synchronisation
      if (cameraRef.current) {
        cameraRef.current.position.set(
          propsRef.current.camX !== undefined ? propsRef.current.camX : 0,
          propsRef.current.camY !== undefined ? propsRef.current.camY : 0,
          propsRef.current.camZ !== undefined ? propsRef.current.camZ : 4.6
        );
      }

      // 2. Earth Globe Scale and Positioning
      if (earthTiltGroupRef.current) {
        const s = propsRef.current.earthScale ?? 1.0;
        earthTiltGroupRef.current.scale.set(s, s, s);
        earthTiltGroupRef.current.position.x = earthBaseX * (1 - p * 0.65);
        earthTiltGroupRef.current.position.y = earthBaseY + p * 0.05;
      }

      // 3. Scrolltide Momentum Drag Physics Loop
      const d = dragRef.current;
      if (!d.isDragging) {
        d.rotY += d.velY;
        d.rotX = Math.max(-0.65, Math.min(0.65, d.rotX + d.velX));

        d.velY *= 0.945;
        d.velX *= 0.945;

        if (
          Math.abs(d.velY) < 0.0001 &&
          Math.abs(d.velX) < 0.0001 &&
          !propsRef.current.isReducedMotion
        ) {
          const baseSpeed = 0.016;
          const scrollBonus = 0.035 * p;
          d.rotY += dt * (baseSpeed + scrollBonus);
        }
      }

      if (particleMeshRef.current) {
        particleMeshRef.current.rotation.y = d.rotY;
        particleMeshRef.current.rotation.x = d.rotX;
      }
      if (innerCoreMeshRef.current) {
        innerCoreMeshRef.current.rotation.y = d.rotY;
        innerCoreMeshRef.current.rotation.x = d.rotX;
      }
      if (earthMeshRef.current) {
        earthMeshRef.current.rotation.y = d.rotY;
        earthMeshRef.current.rotation.x = d.rotX;
      }
      if (cloudMeshRef.current) {
        cloudMeshRef.current.rotation.y = d.rotY * 1.02;
        cloudMeshRef.current.rotation.x = d.rotX;
      }

      if (ringMesh1Ref.current) ringMesh1Ref.current.rotation.y += dt * 0.06;
      if (ringMesh2Ref.current) ringMesh2Ref.current.rotation.y -= dt * 0.045;
      if (ringMesh3Ref.current) ringMesh3Ref.current.rotation.y += dt * 0.035;

      const isParticleMode = propsRef.current.globeMode === "particle";
      if (particleMeshRef.current) particleMeshRef.current.visible = isParticleMode;
      if (innerCoreMeshRef.current) innerCoreMeshRef.current.visible = isParticleMode;
      if (atmoRimMeshRef.current) atmoRimMeshRef.current.visible = isParticleMode;
      if (ringMesh1Ref.current) ringMesh1Ref.current.visible = isParticleMode;
      if (ringMesh2Ref.current) ringMesh2Ref.current.visible = isParticleMode;
      if (ringMesh3Ref.current) ringMesh3Ref.current.visible = isParticleMode;
      if (earthMeshRef.current) earthMeshRef.current.visible = !isParticleMode;
      if (cloudMeshRef.current) cloudMeshRef.current.visible = !isParticleMode;

      // 4. Spacecraft Kinematics & Cinematic Pass
      if (satelliteGroupRef.current) {
        let orbX = satBaseX;
        let orbY = satBaseY;
        let orbZ = satBaseZ;
        let rotX = 0.22;
        let rotY = -0.36;
        let rotZ = -0.14;
        let passScaleMult = 1.0;

        if (p <= 0.10) {
          const t = currentTime * 0.001;
          const driftX = propsRef.current.isReducedMotion ? 0 : 0.005 * Math.sin(t * 0.35);
          const driftY = propsRef.current.isReducedMotion ? 0 : 0.006 * Math.cos(t * 0.40);
          orbX = satBaseX + driftX;
          orbY = satBaseY + driftY;
          orbZ = satBaseZ;
          rotX = 0.22;
          rotY = -0.36;
          rotZ = -0.14;
          passScaleMult = 1.0;
        } else if (p <= 0.35) {
          const t = (p - 0.10) / 0.25;
          const sinT = Math.sin(t * Math.PI * 0.5);
          orbX = satBaseX - 0.85 * Math.sin(t * Math.PI * 0.65);
          orbY = satBaseY - 0.25 * Math.sin(t * Math.PI * 0.60);
          orbZ = satBaseZ + 1.45 * sinT;
          rotX = 0.22 + 0.18 * t;
          rotY = -0.36 + 0.48 * t;
          rotZ = -0.14 - 0.16 * t;
          passScaleMult = 1.0 + 0.35 * Math.sin(t * Math.PI);
        } else if (p <= 0.58) {
          const t = (p - 0.35) / 0.23;
          orbX = (satBaseX - 0.85) + 0.70 * t;
          orbY = (satBaseY - 0.25) + 0.38 * t;
          orbZ = (satBaseZ + 1.45) - 1.40 * t;
          rotX = 0.40 + 0.08 * t;
          rotY = 0.12 + 0.34 * t;
          rotZ = -0.30 + 0.12 * t;
          passScaleMult = 1.35 - 0.35 * t;
        } else if (p <= 0.82) {
          const t = (p - 0.58) / 0.24;
          orbX = (satBaseX - 0.15) + 0.90 * Math.pow(t, 1.2);
          orbY = (satBaseY + 0.13) + 0.55 * t;
          orbZ = (satBaseZ + 0.05) - 0.75 * t;
          rotX = 0.48 + 0.10 * t;
          rotY = 0.46 + 0.18 * t;
          rotZ = -0.18;
          passScaleMult = 1.0 - 0.15 * t;
        } else {
          const t = (p - 0.82) / 0.18;
          orbX = (satBaseX + 0.75) + 1.20 * t;
          orbY = (satBaseY + 0.68) + 0.80 * t;
          orbZ = (satBaseZ - 0.70) - 1.20 * t;
          rotX = 0.58;
          rotY = 0.64;
          rotZ = -0.18;
          passScaleMult = Math.max(0.6, 0.85 - 0.35 * t);
        }

        // Satellite Character Pointer Parallax & Banking Reaction (Requirement 3)
        const mx = propsRef.current.mouseX ?? 0;
        const my = propsRef.current.mouseY ?? 0;
        const satMouseX = propsRef.current.isReducedMotion ? 0 : mx * 0.08;
        const satMouseY = propsRef.current.isReducedMotion ? 0 : -my * 0.08;
        const satMouseZ = propsRef.current.isReducedMotion ? 0 : (mx + my) * 0.06;
        const satMouseRotZ = propsRef.current.isReducedMotion ? 0 : mx * 0.12;

        satelliteGroupRef.current.position.set(orbX + satMouseX, orbY + satMouseY, orbZ + satMouseZ);
        satelliteGroupRef.current.rotation.set(rotX, rotY, rotZ + satMouseRotZ);

        const currentScale = satScale * passScaleMult;
        satelliteGroupRef.current.scale.set(currentScale, currentScale, currentScale);

        const satAlpha = propsRef.current.satelliteAlpha ?? 1.0;
        const exitFade = p < 0.78 ? 1.0 : Math.max(0, 1.0 - (p - 0.78) / 0.14);
        const effectiveAlpha = satAlpha * exitFade;
        satelliteProgramsRef.current.forEach((prog) => {
          if (prog && prog.uniforms?.uAlpha) {
            prog.uniforms.uAlpha.value = effectiveAlpha;
          }
        });
      }

      const timeSec = currentTime * 0.001;
      const finalEarthAlpha = (propsRef.current.earthAlpha ?? 1.0) * introFade;

      if (particleProgramRef.current) {
        particleProgramRef.current.uniforms.uTime.value = timeSec;
        particleProgramRef.current.uniforms.uAlpha.value = finalEarthAlpha;
      }
      if (coreProgramRef.current) {
        coreProgramRef.current.uniforms.uAlpha.value = finalEarthAlpha;
      }
      if (atmoRimProgramRef.current) {
        atmoRimProgramRef.current.uniforms.uAlpha.value = finalEarthAlpha;
      }
      if (ringProgram1Ref.current) {
        ringProgram1Ref.current.uniforms.uTime.value = timeSec;
        ringProgram1Ref.current.uniforms.uAlpha.value = finalEarthAlpha;
      }
      if (ringProgram2Ref.current) {
        ringProgram2Ref.current.uniforms.uTime.value = timeSec;
        ringProgram2Ref.current.uniforms.uAlpha.value = finalEarthAlpha;
      }
      if (ringProgram3Ref.current) {
        ringProgram3Ref.current.uniforms.uTime.value = timeSec;
        ringProgram3Ref.current.uniforms.uAlpha.value = finalEarthAlpha;
      }
      if (earthProgramRef.current) {
        earthProgramRef.current.uniforms.uTime.value = timeSec;
        earthProgramRef.current.uniforms.uAlpha.value = finalEarthAlpha;
      }
      if (cloudProgramRef.current) {
        cloudProgramRef.current.uniforms.uAlpha.value = finalEarthAlpha;
      }

      renderer.render({ scene, camera });
      animFrameRef.current = requestAnimationFrame(render);
    }

    animFrameRef.current = requestAnimationFrame(render);

    return () => {
      isMounted = false;
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener("resize", resize);
      if (gl.canvas && gl.canvas.parentNode === container) {
        container.removeChild(gl.canvas);
      }
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, [isReducedMotion]);

  const propsRef = useRef({
    scrollProgress,
    earthScale,
    earthAlpha,
    satelliteAlpha,
    camZ,
    camX,
    camY,
    mouseX,
    mouseY,
    isReducedMotion,
    globeMode,
  });

  useEffect(() => {
    propsRef.current = {
      scrollProgress,
      earthScale,
      earthAlpha,
      satelliteAlpha,
      camZ,
      camX,
      camY,
      mouseX,
      mouseY,
      isReducedMotion,
      globeMode,
    };
  }, [
    scrollProgress,
    earthScale,
    earthAlpha,
    satelliteAlpha,
    camZ,
    camX,
    camY,
    mouseX,
    mouseY,
    isReducedMotion,
    globeMode,
  ]);

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      className="absolute inset-0 w-full h-full z-15 cursor-grab active:cursor-grabbing touch-none select-none"
      style={{ opacity: earthAlpha }}
      aria-label="Interactive 3D Particle Globe"
    />
  );
}
