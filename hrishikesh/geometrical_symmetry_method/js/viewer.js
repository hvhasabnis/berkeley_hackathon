const WORLD_UP = new THREE.Vector3(0, 0, 1);

const VIEW_MODES = {
  input: { title: "Input Incomplete Cloud", label: "Input incomplete cloud" },
  reconstructed: { title: "Reconstructed Particles Viewer", label: "Reconstructed particles" },
  completed: { title: "Completed Point Cloud Viewer", label: "Completed cloud" }
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101010);

const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.up.copy(WORLD_UP);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.rotateSpeed = 0.38;
controls.enableZoom = false;
controls.enablePan = false;
controls.screenSpacePanning = false;
controls.minPolarAngle = THREE.MathUtils.degToRad(8);
controls.maxPolarAngle = THREE.MathUtils.degToRad(86);

let gridHelper = null;
const clouds = [];
const statusEl = document.getElementById("status");
const legendEl = document.getElementById("legend");
const titleEl = document.getElementById("viewerTitle");
const subjectSelectEl = document.getElementById("subjectSelect");
const caseSelectEl = document.getElementById("caseSelect");
const viewSelectEl = document.getElementById("viewSelect");
const customUploadBoxEl = document.getElementById("customUploadBox");
const customPlyInputEl = document.getElementById("customPlyInput");
const customModeSelectEl = document.getElementById("customModeSelect");
const clearCustomBtnEl = document.getElementById("clearCustomBtn");

const CUSTOM_SUBJECT_ID = "custom_monument";
let customState = null;

const CUSTOM_RECONSTRUCTION_MODES = {
  conservative: {
    label: "Conservative",
    voxelScale: 0.010,
    supportRadiusMul: 2.8,
    supportMinNeighbors: 3,
    duplicateMul: 1.35,
    redSupportRadiusMul: 2.9,
    redSupportMinNeighbors: 3,
    useAdaptiveCenterline: false,
    maxRedFraction: 0.45
  },
  balanced: {
    label: "Balanced",
    voxelScale: 0.008,
    supportRadiusMul: 2.5,
    supportMinNeighbors: 3,
    duplicateMul: 1.05,
    redSupportRadiusMul: 2.55,
    redSupportMinNeighbors: 2,
    useAdaptiveCenterline: true,
    maxRedFraction: 0.85
  },
  aggressive: {
    label: "Aggressive",
    voxelScale: 0.0065,
    supportRadiusMul: 2.25,
    supportMinNeighbors: 2,
    duplicateMul: 0.78,
    redSupportRadiusMul: 2.25,
    redSupportMinNeighbors: 2,
    useAdaptiveCenterline: true,
    maxRedFraction: 1.25
  }
};

function getCustomMode() {
  const mode = customModeSelectEl ? customModeSelectEl.value : "balanced";
  return CUSTOM_RECONSTRUCTION_MODES[mode] ? mode : "balanced";
}

let allSubjects = null;
let activeSubjectId = null;
let activeCaseId = null;
let activeViewMode = null;

const horizontalOrbitSensitivity = 0.00115;
const verticalSurfaceSensitivity = 0.0022;
const keyboardMoveSpeed = 0.050;
const movementDamping = 0.88;
const orbitDamping = 0.86;
const maxMoveSpeed = 0.30;
const maxOrbitSpeed = 0.065;
const minCameraZ = 0.20;
const moveVelocity = new THREE.Vector3();
const pressedKeys = {};
let orbitVelocity = 0;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeWheelDelta(event) {
  let dx = event.deltaX;
  let dy = event.deltaY;
  if (event.deltaMode === 1) {
    dx *= 16;
    dy *= 16;
  } else if (event.deltaMode === 2) {
    dx *= window.innerHeight;
    dy *= window.innerHeight;
  }
  return { dx, dy };
}

function getSurfaceBasis() {
  const forward3D = new THREE.Vector3();
  camera.getWorldDirection(forward3D).normalize();

  const groundForward = forward3D.clone();
  groundForward.z = 0;
  if (groundForward.lengthSq() < 0.000001) groundForward.set(0, 1, 0);
  groundForward.normalize();

  const right = new THREE.Vector3();
  right.crossVectors(groundForward, WORLD_UP).normalize();
  return { groundForward, right, up: WORLD_UP };
}

function addMoveVelocity(direction, amount) {
  moveVelocity.addScaledVector(direction, amount);
  if (moveVelocity.length() > maxMoveSpeed) moveVelocity.setLength(maxMoveSpeed);
}

function applyKeyboardMovement() {
  const basis = getSurfaceBasis();
  if (pressedKeys.w || pressedKeys.arrowup) addMoveVelocity(basis.groundForward, keyboardMoveSpeed);
  if (pressedKeys.s || pressedKeys.arrowdown) addMoveVelocity(basis.groundForward, -keyboardMoveSpeed);
  if (pressedKeys.a || pressedKeys.arrowleft) orbitVelocity += keyboardMoveSpeed * 0.023;
  if (pressedKeys.d || pressedKeys.arrowright) orbitVelocity -= keyboardMoveSpeed * 0.023;
  if (pressedKeys.q) addMoveVelocity(basis.up, -keyboardMoveSpeed);
  if (pressedKeys.e) addMoveVelocity(basis.up, keyboardMoveSpeed);
}

function applyAtlasOrbit() {
  if (Math.abs(orbitVelocity) < 0.000001) {
    orbitVelocity = 0;
    return;
  }
  const angle = clamp(orbitVelocity, -maxOrbitSpeed, maxOrbitSpeed);
  const center = controls.target.clone();
  const offset = camera.position.clone().sub(center);
  offset.applyAxisAngle(WORLD_UP, angle);
  camera.position.copy(center).add(offset);
  camera.up.copy(WORLD_UP);
  camera.lookAt(center);
  orbitVelocity *= orbitDamping;
}

function applySurfaceMovement() {
  applyKeyboardMovement();
  if (moveVelocity.lengthSq() < 0.000001) {
    moveVelocity.set(0, 0, 0);
    return;
  }
  const nextCameraPosition = camera.position.clone().add(moveVelocity);
  const nextTargetPosition = controls.target.clone().add(moveVelocity);
  if (nextCameraPosition.z > minCameraZ) {
    camera.position.copy(nextCameraPosition);
    controls.target.copy(nextTargetPosition);
  } else {
    moveVelocity.z = Math.max(0, moveVelocity.z);
  }
  moveVelocity.multiplyScalar(movementDamping);
}

renderer.domElement.addEventListener("wheel", function (event) {
  event.preventDefault();
  const { dx, dy } = normalizeWheelDelta(event);
  const basis = getSurfaceBasis();
  orbitVelocity += clamp(-dx * horizontalOrbitSensitivity, -maxOrbitSpeed, maxOrbitSpeed);
  addMoveVelocity(basis.groundForward, clamp(-dy * verticalSurfaceSensitivity, -maxMoveSpeed, maxMoveSpeed));
}, { passive: false });

window.addEventListener("keydown", function (event) {
  pressedKeys[event.key.toLowerCase()] = true;
  if (event.key.toLowerCase() === "h" && gridHelper) gridHelper.visible = !gridHelper.visible;
});
window.addEventListener("keyup", function (event) {
  pressedKeys[event.key.toLowerCase()] = false;
});

function getParams() {
  return new URLSearchParams(window.location.search);
}

function getSubjectCases(subjectCfg) {
  return subjectCfg.cases || {};
}

function getInitialSubject(subjects) {
  const fromUrl = getParams().get("subject");
  const saved = localStorage.getItem("selectedSubject");
  if (fromUrl && subjects[fromUrl]) return fromUrl;
  if (saved && subjects[saved]) return saved;
  return Object.keys(subjects)[0];
}

function getInitialCase(subjects, subjectId) {
  const cases = getSubjectCases(subjects[subjectId]);
  const fromUrl = getParams().get("case");
  const savedKey = `selectedCase:${subjectId}`;
  const saved = localStorage.getItem(savedKey);
  if (fromUrl && cases[fromUrl]) return fromUrl;
  if (saved && cases[saved]) return saved;
  return Object.keys(cases)[0];
}

function getInitialView() {
  const fromUrl = getParams().get("view");
  const bodyMode = document.body.dataset.mode;
  const saved = localStorage.getItem("selectedView");
  if (fromUrl && VIEW_MODES[fromUrl]) return fromUrl;
  if (bodyMode && VIEW_MODES[bodyMode]) return bodyMode;
  if (saved && VIEW_MODES[saved]) return saved;
  return "reconstructed";
}

function updateURL(subjectId, caseId, viewMode) {
  localStorage.setItem("selectedSubject", subjectId);
  localStorage.setItem(`selectedCase:${subjectId}`, caseId);
  localStorage.setItem("selectedView", viewMode);

  const url = new URL(window.location.href);
  url.searchParams.set("subject", subjectId);
  url.searchParams.set("case", caseId);
  url.searchParams.set("view", viewMode);
  window.history.replaceState({}, "", url.toString());
}

function populateCaseSelect(subjectId, selectedCase) {
  const subjectCfg = allSubjects[subjectId];
  const cases = getSubjectCases(subjectCfg);
  caseSelectEl.innerHTML = "";
  for (const [caseId, caseCfg] of Object.entries(cases)) {
    const option = document.createElement("option");
    option.value = caseId;
    option.textContent = caseCfg.name || caseId;
    if (caseId === selectedCase) option.selected = true;
    caseSelectEl.appendChild(option);
  }
}

function configureSelectors(subjects, selectedSubject, selectedCase, selectedView) {
  subjectSelectEl.innerHTML = "";
  for (const [id, cfg] of Object.entries(subjects)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = cfg.name;
    if (id === selectedSubject) option.selected = true;
    subjectSelectEl.appendChild(option);
  }

  populateCaseSelect(selectedSubject, selectedCase);
  viewSelectEl.value = selectedView;

  subjectSelectEl.addEventListener("change", () => {
    const nextSubject = subjectSelectEl.value;
    setCustomUploadVisibility(nextSubject);
    const nextCase = getInitialCase(allSubjects, nextSubject);
    populateCaseSelect(nextSubject, nextCase);
    loadSelected(nextSubject, nextCase, viewSelectEl.value);
  });

  caseSelectEl.addEventListener("change", () => {
    loadSelected(subjectSelectEl.value, caseSelectEl.value, viewSelectEl.value);
  });

  viewSelectEl.addEventListener("change", () => {
    loadSelected(subjectSelectEl.value, caseSelectEl.value, viewSelectEl.value);
  });

  if (customModeSelectEl) {
    customModeSelectEl.addEventListener("change", async () => {
      if (customState && customState.rawPoints && activeSubjectId === CUSTOM_SUBJECT_ID) {
        try {
          statusEl.textContent = `Reprocessing custom PLY in ${CUSTOM_RECONSTRUCTION_MODES[getCustomMode()].label} mode...`;
          await new Promise(resolve => setTimeout(resolve, 20));
          customState = processCustomPointCloud(customState.rawPoints, customState.filename, getCustomMode());
          await loadSelected(CUSTOM_SUBJECT_ID, "uploaded_ply", viewSelectEl.value || "reconstructed");
        } catch (error) {
          statusEl.style.background = "rgba(150,0,0,0.72)";
          statusEl.textContent = error.message;
          console.error(error);
        }
      }
    });
  }

  if (customPlyInputEl) {
    customPlyInputEl.addEventListener("change", async () => {
      const file = customPlyInputEl.files && customPlyInputEl.files[0];
      if (!file) return;
      try {
        await handleCustomPlyFile(file);
      } catch (error) {
        statusEl.style.background = "rgba(150,0,0,0.72)";
        statusEl.textContent = error.message;
        console.error(error);
      }
    });
  }
  if (clearCustomBtnEl) {
    clearCustomBtnEl.addEventListener("click", () => {
      customState = null;
      if (customPlyInputEl) customPlyInputEl.value = "";
      loadSelected(CUSTOM_SUBJECT_ID, "uploaded_ply", viewSelectEl.value || "reconstructed");
    });
  }
  setCustomUploadVisibility(selectedSubject);
}

function mergeConfig(subjectCfg, caseCfg) {
  const merged = { ...subjectCfg, ...caseCfg };
  merged.camera = { ...(subjectCfg.camera || {}), ...(caseCfg.camera || {}) };
  merged.subjectName = subjectCfg.name;
  merged.caseName = caseCfg.name || "Test case";
  merged.subjectDescription = subjectCfg.description || "";
  merged.caseDescription = caseCfg.description || "";
  return merged;
}

function configureCamera(cfg) {
  const c = cfg.camera || {};
  const pos = c.position || [6, -8, 5];
  const target = c.target || [0, 0, 1.5];
  camera.position.set(pos[0], pos[1], pos[2]);
  controls.target.set(target[0], target[1], target[2]);
  controls.update();

  moveVelocity.set(0, 0, 0);
  orbitVelocity = 0;

  if (gridHelper) scene.remove(gridHelper);
  gridHelper = new THREE.GridHelper(c.gridSize || 12, c.gridSize || 12);
  gridHelper.rotation.x = Math.PI / 2;
  gridHelper.position.z = -0.04;
  if (gridHelper.material) {
    gridHelper.material.transparent = true;
    gridHelper.material.opacity = 0.12;
  }
  scene.add(gridHelper);
}

function clearClouds() {
  for (const cloud of clouds) {
    scene.remove(cloud);
    if (cloud.geometry) cloud.geometry.dispose();
    if (cloud.material) cloud.material.dispose();
  }
  clouds.length = 0;
}

function parseCSV(text) {
  const rows = [];
  const lines = text.trim().split(/\r?\n/);
  for (const line of lines) {
    const parts = line.split(",");
    if (parts.length < 3) continue;
    const x = Number(parts[0]);
    const y = Number(parts[1]);
    const z = Number(parts[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    rows.push({ point: [x, y, z], label: (parts[3] || "").trim().toLowerCase() });
  }
  return rows;
}

function addPointCloud(points, color, size, name, opacity = 1.0) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(points.flat()), 3));
  geometry.computeBoundingSphere();
  const material = new THREE.PointsMaterial({
    color,
    size,
    sizeAttenuation: true,
    transparent: opacity < 1.0,
    opacity
  });
  const cloud = new THREE.Points(geometry, material);
  cloud.name = name;
  scene.add(cloud);
  clouds.push(cloud);
  return cloud;
}

function pointsFromRows(rows) {
  return rows.map(r => r.point);
}

function groupByLabel(rows) {
  const groups = new Map();
  for (const row of rows) {
    const label = row.label || "point";
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(row.point);
  }
  return groups;
}

function labelColor(label) {
  const colors = {
    facade: 0xd6bd82,
    seating: 0xc2a46d,
    arch_edge: 0x9c8b6a,
    column: 0xd6bd82,
    entablature: 0xd8c39a,
    base: 0xa98b5d,
    floor: 0x7a6544,
    rock: 0x8a7650,
    doorway_edge: 0x8e8064,
    niche_edge: 0x9f8d65,
    pediment: 0xd0b678,
    cornice: 0xd8c39a,
    tholos: 0xc9ad73,
    stair: 0x9d845a,
    rubble: 0x884322,
    broken_edge: 0xe0c58d,
    reconstructed_only: 0xff2020,
    completed: 0xd6bd82,
    point: 0xd6bd82
  };
  return colors[label] || 0xd6bd82;
}

function loadCSV(path) {
  return fetch(path, { cache: "no-store" }).then(response => {
    if (!response.ok) throw new Error(`${path} not found. Run python/run_subject.py for this subject and test data first.`);
    return response.text();
  });
}

function setCommonLegendFooter() {
  return `
    <div class="hint">Two-finger left/right = curved orbit</div>
    <div class="hint">Two-finger up/down = move over surface</div>
    <div class="hint">Drag = rotate, H = toggle grid</div>
  `;
}


function addCustomSubject(subjects) {
  subjects[CUSTOM_SUBJECT_ID] = {
    name: "Custom monument (upload PLY)",
    description: "User uploaded PLY processed locally in the browser as a custom monument.",
    camera: {
      position: [5.5, -7.0, 4.8],
      target: [0, 0, 1.5],
      gridSize: 12,
      pointSizeInput: 0.023,
      pointSizeRed: 0.030,
      pointSizeCompleted: 0.023
    },
    cases: {
      uploaded_ply: {
        name: "Uploaded PLY file",
        description: "Upload your own incomplete PLY point cloud. Reconstruction runs locally in the browser."
      }
    }
  };
}

function setCustomUploadVisibility(subjectId) {
  if (!customUploadBoxEl) return;
  customUploadBoxEl.classList.toggle("hidden", subjectId !== CUSTOM_SUBJECT_ID);
  if (caseSelectEl) caseSelectEl.disabled = subjectId === CUSTOM_SUBJECT_ID;
}

function computeBounds(points) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let i = 0; i < 3; i++) {
      if (p[i] < min[i]) min[i] = p[i];
      if (p[i] > max[i]) max[i] = p[i];
    }
  }
  const center = [
    (min[0] + max[0]) * 0.5,
    (min[1] + max[1]) * 0.5,
    (min[2] + max[2]) * 0.5
  ];
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const diag = Math.sqrt(size[0] * size[0] + size[1] * size[1] + size[2] * size[2]);
  return { min, max, center, size, diag };
}

function fitCameraToPoints(points) {
  if (!points || points.length === 0) return;
  const b = computeBounds(points);
  const radius = Math.max(b.diag * 0.55, 1.0);
  const center = b.center;
  controls.target.set(center[0], center[1], center[2]);
  camera.position.set(
    center[0] + radius * 0.85,
    center[1] - radius * 1.45,
    center[2] + radius * 0.75
  );
  camera.up.copy(WORLD_UP);
  controls.update();

  if (gridHelper) scene.remove(gridHelper);
  const gridSize = Math.max(4, Math.ceil(Math.max(b.size[0], b.size[1]) * 1.7));
  gridHelper = new THREE.GridHelper(gridSize, Math.max(4, Math.min(24, gridSize)));
  gridHelper.rotation.x = Math.PI / 2;
  gridHelper.position.z = b.min[2] - Math.max(b.diag * 0.006, 0.03);
  if (gridHelper.material) {
    gridHelper.material.transparent = true;
    gridHelper.material.opacity = 0.12;
  }
  scene.add(gridHelper);

  moveVelocity.set(0, 0, 0);
  orbitVelocity = 0;
}

function voxelKey(p, size) {
  return `${Math.floor(p[0] / size)},${Math.floor(p[1] / size)},${Math.floor(p[2] / size)}`;
}

function voxelGridFilterJs(points, size) {
  if (!points || points.length === 0) return [];
  const cells = new Map();
  for (const p of points) {
    const key = voxelKey(p, size);
    let cell = cells.get(key);
    if (!cell) {
      cell = { sx: 0, sy: 0, sz: 0, n: 0 };
      cells.set(key, cell);
    }
    cell.sx += p[0];
    cell.sy += p[1];
    cell.sz += p[2];
    cell.n += 1;
  }
  const out = [];
  for (const c of cells.values()) out.push([c.sx / c.n, c.sy / c.n, c.sz / c.n]);
  return out;
}

function samplePoints(points, maxCount) {
  if (points.length <= maxCount) return points.slice();
  const stride = points.length / maxCount;
  const out = [];
  for (let i = 0; i < maxCount; i++) out.push(points[Math.floor(i * stride)]);
  return out;
}

function median(values) {
  if (!values.length) return 0;
  const v = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) * 0.5;
}

function buildSpatialHash(points, cellSize) {
  const hash = new Map();
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const key = voxelKey(p, cellSize);
    let bucket = hash.get(key);
    if (!bucket) {
      bucket = [];
      hash.set(key, bucket);
    }
    bucket.push(i);
  }
  return { hash, cellSize, points };
}

function nearestDistanceSquared(p, spatial, maxRing = 2) {
  const { hash, cellSize, points } = spatial;
  const ix = Math.floor(p[0] / cellSize);
  const iy = Math.floor(p[1] / cellSize);
  const iz = Math.floor(p[2] / cellSize);
  let best = Infinity;
  for (let dx = -maxRing; dx <= maxRing; dx++) {
    for (let dy = -maxRing; dy <= maxRing; dy++) {
      for (let dz = -maxRing; dz <= maxRing; dz++) {
        const bucket = hash.get(`${ix + dx},${iy + dy},${iz + dz}`);
        if (!bucket) continue;
        for (const idx of bucket) {
          const q = points[idx];
          const ex = p[0] - q[0];
          const ey = p[1] - q[1];
          const ez = p[2] - q[2];
          const d2 = ex * ex + ey * ey + ez * ez;
          if (d2 > 0 && d2 < best) best = d2;
        }
      }
    }
  }
  return best;
}

function estimateSpacing(points, diag) {
  if (points.length < 2) return Math.max(diag * 0.015, 0.01);
  const sample = samplePoints(points, 900);
  const cell = Math.max(diag * 0.03, 0.01);
  const spatial = buildSpatialHash(sample, cell);
  const distances = [];
  for (const p of sample) {
    const d2 = nearestDistanceSquared(p, spatial, 3);
    if (Number.isFinite(d2) && d2 > 0) distances.push(Math.sqrt(d2));
  }
  const m = median(distances);
  return m > 0 ? m : Math.max(diag * 0.015, 0.01);
}

function supportFilterJs(points, radius, minNeighbors) {
  if (points.length === 0) return [];
  const spatial = buildSpatialHash(points, radius);
  const keep = [];
  const r2 = radius * radius;
  for (const p of points) {
    const { hash, cellSize } = spatial;
    const ix = Math.floor(p[0] / cellSize);
    const iy = Math.floor(p[1] / cellSize);
    const iz = Math.floor(p[2] / cellSize);
    let count = 0;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const bucket = hash.get(`${ix + dx},${iy + dy},${iz + dz}`);
          if (!bucket) continue;
          for (const idx of bucket) {
            const q = points[idx];
            const ex = p[0] - q[0];
            const ey = p[1] - q[1];
            const ez = p[2] - q[2];
            if (ex * ex + ey * ey + ez * ez <= r2) count++;
            if (count >= minNeighbors) break;
          }
          if (count >= minNeighbors) break;
        }
        if (count >= minNeighbors) break;
      }
      if (count >= minNeighbors) break;
    }
    if (count >= minNeighbors) keep.push(p);
  }
  return keep;
}

function xyPcaAngles(points) {
  const b = computeBounds(points);
  let cxx = 0, cxy = 0, cyy = 0;
  for (const p of points) {
    const x = p[0] - b.center[0];
    const y = p[1] - b.center[1];
    cxx += x * x;
    cxy += x * y;
    cyy += y * y;
  }
  const angle = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
  return [angle, angle + Math.PI / 2];
}

function mirrorPointAcrossPlane(p, normal, offset) {
  const dist = p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2] - offset;
  return [
    p[0] - 2 * dist * normal[0],
    p[1] - 2 * dist * normal[1],
    p[2] - 2 * dist * normal[2]
  ];
}

function scoreSymmetryPlane(sample, normal, offset, cellSize) {
  const spatial = buildSpatialHash(sample, cellSize);
  let total = 0;
  let count = 0;
  const cap = cellSize * cellSize * 16;
  for (const p of sample) {
    const m = mirrorPointAcrossPlane(p, normal, offset);
    const d2 = nearestDistanceSquared(m, spatial, 3);
    total += Number.isFinite(d2) ? Math.min(d2, cap) : cap;
    count += 1;
  }
  return count ? Math.sqrt(total / count) : Infinity;
}

function detectSymmetryPlane(points, spacing) {
  const b = computeBounds(points);
  const sample = samplePoints(points, 900);
  const angles = [];
  for (let i = 0; i < 18; i++) angles.push((i * Math.PI) / 18);
  for (const a of xyPcaAngles(points)) {
    angles.push(a);
    angles.push(a + Math.PI / 18);
    angles.push(a - Math.PI / 18);
  }
  const uniqueAngles = [];
  for (const a of angles) {
    let aa = a % Math.PI;
    if (aa < 0) aa += Math.PI;
    if (!uniqueAngles.some(x => Math.abs(x - aa) < 0.015)) uniqueAngles.push(aa);
  }

  let best = { score: Infinity, normal: [1, 0, 0], offset: b.center[0], angle: 0 };
  const cell = Math.max(spacing * 3.5, b.diag * 0.025, 0.01);
  for (const angle of uniqueAngles) {
    const normal = [Math.cos(angle), Math.sin(angle), 0];
    const dots = sample.map(p => p[0] * normal[0] + p[1] * normal[1]);
    const centerOffset = median(dots);
    const spread = Math.max(...dots) - Math.min(...dots);
    const offsets = [centerOffset, centerOffset - spread * 0.035, centerOffset + spread * 0.035];
    for (const offset of offsets) {
      const score = scoreSymmetryPlane(sample, normal, offset, cell);
      if (score < best.score) best = { score, normal, offset, angle };
    }
  }
  return best;
}

function estimateCenterline(points, binCount = 18) {
  const b = computeBounds(points);
  const zMin = b.min[2];
  const zMax = b.max[2];
  const zSpan = Math.max(zMax - zMin, 1e-9);
  const bins = [];
  for (let i = 0; i < binCount; i++) bins.push({ xs: [], ys: [], z: zMin + (i + 0.5) * zSpan / binCount });

  for (const p of points) {
    let idx = Math.floor(((p[2] - zMin) / zSpan) * binCount);
    idx = clamp(idx, 0, binCount - 1);
    bins[idx].xs.push(p[0]);
    bins[idx].ys.push(p[1]);
  }

  let lastX = b.center[0];
  let lastY = b.center[1];
  for (const bin of bins) {
    if (bin.xs.length >= 8) {
      bin.x = median(bin.xs);
      bin.y = median(bin.ys);
      lastX = bin.x;
      lastY = bin.y;
    } else {
      bin.x = lastX;
      bin.y = lastY;
    }
  }

  // Light smoothing keeps a leaning monument's centerline stable without fitting to damage holes.
  for (let pass = 0; pass < 2; pass++) {
    const prev = bins.map(b => ({ x: b.x, y: b.y }));
    for (let i = 1; i < bins.length - 1; i++) {
      bins[i].x = (prev[i - 1].x + 2 * prev[i].x + prev[i + 1].x) / 4;
      bins[i].y = (prev[i - 1].y + 2 * prev[i].y + prev[i + 1].y) / 4;
    }
  }

  function centerAt(z) {
    const t = clamp((z - zMin) / zSpan, 0, 1) * (binCount - 1);
    const i0 = Math.floor(t);
    const i1 = Math.min(binCount - 1, i0 + 1);
    const a = t - i0;
    return [
      bins[i0].x * (1 - a) + bins[i1].x * a,
      bins[i0].y * (1 - a) + bins[i1].y * a
    ];
  }

  return { zMin, zMax, bins, centerAt };
}

function normalizeByCenterline(points, centerline) {
  return points.map(p => {
    const c = centerline.centerAt(p[2]);
    return [p[0] - c[0], p[1] - c[1], p[2]];
  });
}

function denormalizeByCenterline(points, centerline) {
  return points.map(p => {
    const c = centerline.centerAt(p[2]);
    return [p[0] + c[0], p[1] + c[1], p[2]];
  });
}

function processCustomPointCloud(rawPoints, filename, modeName = getCustomMode()) {
  if (!rawPoints || rawPoints.length < 20) throw new Error("PLY must contain at least 20 valid xyz vertices.");
  const mode = CUSTOM_RECONSTRUCTION_MODES[modeName] || CUSTOM_RECONSTRUCTION_MODES.balanced;
  const rawBounds = computeBounds(rawPoints);
  const rawDiag = Math.max(rawBounds.diag, 1e-6);
  const voxel = clamp(rawDiag * mode.voxelScale, rawDiag * 0.0025, rawDiag * 0.025);

  let observed = voxelGridFilterJs(rawPoints, voxel);
  const spacing = estimateSpacing(observed, rawDiag);
  const supported = supportFilterJs(observed, Math.max(spacing * mode.supportRadiusMul, voxel * 2.0), mode.supportMinNeighbors);
  if (supported.length > observed.length * 0.34) observed = supported;

  // Balanced/aggressive mode compensates for leaning towers or slanted monuments by reflecting
  // around a per-height centerline instead of a single rigid vertical axis. This is generic and
  // does not use object-specific labels or hard-coded monument dimensions.
  const centerline = mode.useAdaptiveCenterline ? estimateCenterline(observed, 20) : null;
  const symmetryPoints = centerline ? normalizeByCenterline(observed, centerline) : observed;

  const plane = detectSymmetryPlane(symmetryPoints, spacing);
  const reflectedNormalized = symmetryPoints.map(p => mirrorPointAcrossPlane(p, plane.normal, plane.offset));
  const reflected = centerline ? denormalizeByCenterline(reflectedNormalized, centerline) : reflectedNormalized;

  const spatialObserved = buildSpatialHash(observed, Math.max(spacing * 1.55, voxel * 1.9));
  const duplicateThreshold = Math.max(spacing * mode.duplicateMul, voxel * 1.25);
  const duplicateThreshold2 = duplicateThreshold * duplicateThreshold;
  const b = computeBounds(observed);
  const pad = Math.max(b.diag * (modeName === "aggressive" ? 0.10 : 0.075), spacing * 4);

  let reconstructed = [];
  for (const p of reflected) {
    if (p[0] < b.min[0] - pad || p[0] > b.max[0] + pad) continue;
    if (p[1] < b.min[1] - pad || p[1] > b.max[1] + pad) continue;
    if (p[2] < b.min[2] - pad || p[2] > b.max[2] + pad) continue;
    const d2 = nearestDistanceSquared(p, spatialObserved, 3);
    if (!Number.isFinite(d2) || d2 > duplicateThreshold2) reconstructed.push(p);
  }

  reconstructed = voxelGridFilterJs(reconstructed, voxel);

  const redSupported = supportFilterJs(
    reconstructed,
    Math.max(spacing * mode.redSupportRadiusMul, voxel * 2.0),
    mode.redSupportMinNeighbors
  );
  if (redSupported.length > reconstructed.length * 0.22 || redSupported.length > 120) reconstructed = redSupported;

  // Guardrail against over-filling: cap custom reconstruction to a mode-dependent fraction of
  // the observed cloud, prioritizing coherent candidates farthest from existing observed points.
  const maxRed = Math.floor(observed.length * mode.maxRedFraction);
  if (reconstructed.length > maxRed && maxRed > 0) {
    const scored = reconstructed.map(p => {
      const d2 = nearestDistanceSquared(p, spatialObserved, 3);
      return { p, d2: Number.isFinite(d2) ? d2 : Infinity };
    });
    scored.sort((a, b) => b.d2 - a.d2);
    reconstructed = scored.slice(0, maxRed).map(x => x.p);
  }

  reconstructed = voxelGridFilterJs(reconstructed, voxel);

  const completed = voxelGridFilterJs(observed.concat(reconstructed), voxel);
  const angleDeg = (plane.angle * 180 / Math.PI).toFixed(1);

  return {
    filename,
    rawPoints,
    observed,
    reconstructed,
    completed,
    voxel,
    spacing,
    plane,
    modeName,
    modeLabel: mode.label,
    stats: {
      raw: rawPoints.length,
      observed: observed.length,
      reconstructed: reconstructed.length,
      completed: completed.length,
      planeAngleDeg: angleDeg,
      planeScore: plane.score
    }
  };
}

function parsePlyHeader(buffer) {
  const maxHeader = Math.min(buffer.byteLength, 1024 * 1024);
  const headerText = new TextDecoder("utf-8").decode(buffer.slice(0, maxHeader));
  const endIdx = headerText.indexOf("end_header");
  if (endIdx < 0) throw new Error("Invalid PLY: missing end_header.");
  let headerEnd = endIdx + "end_header".length;
  if (headerText[headerEnd] === "\r" && headerText[headerEnd + 1] === "\n") headerEnd += 2;
  else if (headerText[headerEnd] === "\n") headerEnd += 1;
  else if (headerText[headerEnd] === "\r") headerEnd += 1;

  const lines = headerText.slice(0, headerEnd).split(/\r?\n/).map(x => x.trim()).filter(Boolean);
  if (!lines[0] || lines[0].toLowerCase() !== "ply") throw new Error("Invalid PLY: first line must be ply.");
  let format = "ascii";
  let vertexCount = 0;
  let inVertex = false;
  const properties = [];

  for (const line of lines) {
    const parts = line.split(/\s+/);
    if (parts[0] === "format") format = parts[1];
    else if (parts[0] === "element") {
      inVertex = parts[1] === "vertex";
      if (inVertex) vertexCount = Number(parts[2]);
    } else if (inVertex && parts[0] === "property") {
      if (parts[1] === "list") {
        properties.push({ name: parts[4], type: "list", countType: parts[2], itemType: parts[3] });
      } else {
        properties.push({ name: parts[2], type: parts[1] });
      }
    }
  }
  if (!vertexCount || vertexCount < 1) throw new Error("PLY contains no vertex element.");
  const xIndex = properties.findIndex(p => p.name === "x");
  const yIndex = properties.findIndex(p => p.name === "y");
  const zIndex = properties.findIndex(p => p.name === "z");
  if (xIndex < 0 || yIndex < 0 || zIndex < 0) throw new Error("PLY vertex properties must include x, y, and z.");
  return { format, vertexCount, properties, headerEnd, xIndex, yIndex, zIndex };
}

const PLY_TYPE_INFO = {
  char: [1, (dv, o, le) => dv.getInt8(o)],
  int8: [1, (dv, o, le) => dv.getInt8(o)],
  uchar: [1, (dv, o, le) => dv.getUint8(o)],
  uint8: [1, (dv, o, le) => dv.getUint8(o)],
  short: [2, (dv, o, le) => dv.getInt16(o, le)],
  int16: [2, (dv, o, le) => dv.getInt16(o, le)],
  ushort: [2, (dv, o, le) => dv.getUint16(o, le)],
  uint16: [2, (dv, o, le) => dv.getUint16(o, le)],
  int: [4, (dv, o, le) => dv.getInt32(o, le)],
  int32: [4, (dv, o, le) => dv.getInt32(o, le)],
  uint: [4, (dv, o, le) => dv.getUint32(o, le)],
  uint32: [4, (dv, o, le) => dv.getUint32(o, le)],
  float: [4, (dv, o, le) => dv.getFloat32(o, le)],
  float32: [4, (dv, o, le) => dv.getFloat32(o, le)],
  double: [8, (dv, o, le) => dv.getFloat64(o, le)],
  float64: [8, (dv, o, le) => dv.getFloat64(o, le)]
};

function parsePlyArrayBuffer(buffer) {
  const h = parsePlyHeader(buffer);
  const points = [];
  if (h.format === "ascii") {
    const text = new TextDecoder("utf-8").decode(buffer.slice(h.headerEnd));
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < Math.min(h.vertexCount, lines.length); i++) {
      const parts = lines[i].trim().split(/\s+/);
      if (parts.length < h.properties.length) continue;
      const x = Number(parts[h.xIndex]);
      const y = Number(parts[h.yIndex]);
      const z = Number(parts[h.zIndex]);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) points.push([x, y, z]);
    }
  } else if (h.format === "binary_little_endian" || h.format === "binary_big_endian") {
    const little = h.format === "binary_little_endian";
    const dv = new DataView(buffer, h.headerEnd);
    let offset = 0;
    for (let i = 0; i < h.vertexCount; i++) {
      const values = [];
      for (const prop of h.properties) {
        if (prop.type === "list") throw new Error("Binary PLY with list vertex properties is not supported for custom upload.");
        const info = PLY_TYPE_INFO[prop.type];
        if (!info) throw new Error(`Unsupported PLY property type: ${prop.type}`);
        const [bytes, read] = info;
        values.push(read(dv, offset, little));
        offset += bytes;
      }
      const x = values[h.xIndex], y = values[h.yIndex], z = values[h.zIndex];
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) points.push([x, y, z]);
    }
  } else {
    throw new Error(`Unsupported PLY format: ${h.format}`);
  }
  if (points.length === 0) throw new Error("No valid xyz vertices found in PLY.");
  return points;
}

async function handleCustomPlyFile(file) {
  statusEl.style.background = "rgba(0,0,0,0.68)";
  statusEl.textContent = `Reading ${file.name}...`;
  const buffer = await file.arrayBuffer();
  const points = parsePlyArrayBuffer(buffer);
  statusEl.textContent = `Processing ${points.length} PLY vertices. This may take a few seconds...`;
  await new Promise(resolve => setTimeout(resolve, 25));
  customState = processCustomPointCloud(points, file.name, getCustomMode());
  localStorage.setItem("selectedSubject", CUSTOM_SUBJECT_ID);
  await loadSelected(CUSTOM_SUBJECT_ID, "uploaded_ply", viewSelectEl.value || "reconstructed");
}

async function loadCustomViewer(viewMode) {
  clearClouds();
  titleEl.textContent = VIEW_MODES[viewMode].title;
  if (!customState) {
    configureCamera(allSubjects[CUSTOM_SUBJECT_ID]);
    legendEl.innerHTML = `
      <div><b>Custom monument</b></div>
      <div>Upload an incomplete <code>.ply</code> point cloud to reconstruct it locally.</div>
      <div>Supported: ASCII PLY and binary little/big endian PLY with vertex x, y, z properties.</div>
      ${setCommonLegendFooter()}
    `;
    statusEl.textContent = "Select Custom monument and upload a PLY file.";
    return;
  }

  const sizeInput = 0.023;
  const sizeRed = 0.032;
  fitCameraToPoints(customState.completed.length ? customState.completed : customState.observed);

  if (viewMode === "input") {
    addPointCloud(customState.observed, 0xd6bd82, sizeInput, "custom observed PLY", 0.94);
  } else if (viewMode === "reconstructed") {
    addPointCloud(customState.observed, 0xd6bd82, sizeInput, "custom observed PLY", 0.91);
    addPointCloud(customState.reconstructed, 0xff2020, sizeRed, "custom reconstructed particles", 1.0);
  } else {
    addPointCloud(customState.completed, 0xd6bd82, sizeInput, "custom completed cloud", 0.95);
  }

  legendEl.innerHTML = `
    <div><b>Custom monument</b></div>
    <div><b>${customState.filename}</b></div>
    <div>View: ${VIEW_MODES[viewMode].label}</div>
    <div>Mode: <b>${customState.modeLabel}</b></div>
    <div>Observed = <span style="color:#d6bd82">tan</span>; reconstructed = <span style="color:red">red</span></div>
    <div>Raw vertices: ${customState.stats.raw}</div>
    <div>Clean observed: ${customState.stats.observed}</div>
    <div>Reconstructed: ${customState.stats.reconstructed}</div>
    <div>Completed: ${customState.stats.completed}</div>
    <div>Detected symmetry angle: ${customState.stats.planeAngleDeg}°</div>
    <div class="hint">Custom upload is processed in-browser and does not overwrite saved monument test cases.</div>
    ${setCommonLegendFooter()}
  `;
  statusEl.textContent = `Custom PLY loaded in ${customState.modeLabel} mode: ${customState.stats.observed} observed, ${customState.stats.reconstructed} reconstructed, ${customState.stats.completed} completed points.`;
}

async function loadInputViewer(cfg) {
  const rows = parseCSV(await loadCSV(cfg.input));
  const groups = groupByLabel(rows);
  const size = cfg.camera?.pointSizeInput || 0.025;
  for (const [label, pts] of groups.entries()) {
    const opacity = label === "rubble" ? 0.75 : 0.95;
    const pointSize = label === "rubble" ? size * 1.25 : size;
    addPointCloud(pts, labelColor(label), pointSize, label, opacity);
  }
  legendEl.innerHTML = `
    <div><b>${cfg.subjectName}</b></div>
    <div><b>${cfg.caseName}</b></div>
    <div>View: input incomplete point cloud</div>
    <div>${cfg.caseDescription}</div>
    <div>Source: <code>${cfg.input}</code></div>
    ${setCommonLegendFooter()}
  `;
  statusEl.textContent = `Loaded ${rows.length} input points for ${cfg.subjectName} / ${cfg.caseName}.`;
}

async function loadReconstructedViewer(cfg) {
  const base = cfg.viewerOutput;
  const [pincText, redText] = await Promise.all([
    loadCSV(`${base}/pinc_display_clean.csv`),
    loadCSV(`${base}/reconstructed_only.csv`)
  ]);
  const pinc = pointsFromRows(parseCSV(pincText));
  const red = pointsFromRows(parseCSV(redText));
  addPointCloud(pinc, 0xd6bd82, cfg.camera?.pointSizeInput || 0.023, "clean observed Pinc", 0.93);
  addPointCloud(red, 0xff2020, cfg.camera?.pointSizeRed || 0.030, "reconstructed particles", 1.0);
  legendEl.innerHTML = `
    <div><b>${cfg.subjectName}</b></div>
    <div><b>${cfg.caseName}</b></div>
    <div>Cleaned observed cloud = <span style="color:#d6bd82">tan</span></div>
    <div>Reconstructed/mirrored particles = <span style="color:red">red</span></div>
    <div>${cfg.caseDescription}</div>
    <div>Sources: <code>${base}/pinc_display_clean.csv</code>, <code>${base}/reconstructed_only.csv</code></div>
    ${setCommonLegendFooter()}
  `;
  statusEl.textContent = `Loaded ${pinc.length} clean observed points and ${red.length} reconstructed points for ${cfg.subjectName} / ${cfg.caseName}.`;
}

async function loadCompletedViewer(cfg) {
  const base = cfg.viewerOutput;
  const text = await loadCSV(`${base}/completed.csv`);
  const completed = pointsFromRows(parseCSV(text));
  addPointCloud(completed, 0xd6bd82, cfg.camera?.pointSizeCompleted || 0.023, "completed cloud", 0.95);
  legendEl.innerHTML = `
    <div><b>${cfg.subjectName}</b></div>
    <div><b>${cfg.caseName}</b></div>
    <div>View: completed point cloud</div>
    <div>${cfg.caseDescription}</div>
    <div>Source: <code>${base}/completed.csv</code></div>
    ${setCommonLegendFooter()}
  `;
  statusEl.textContent = `Loaded ${completed.length} completed points for ${cfg.subjectName} / ${cfg.caseName}.`;
}

async function loadSelected(subjectId, caseId, viewMode) {
  if (!allSubjects) return;
  if (!allSubjects[subjectId]) subjectId = Object.keys(allSubjects)[0];
  const cases = getSubjectCases(allSubjects[subjectId]);
  if (!cases[caseId]) caseId = Object.keys(cases)[0];
  if (!VIEW_MODES[viewMode]) viewMode = "reconstructed";

  activeSubjectId = subjectId;
  activeCaseId = caseId;
  activeViewMode = viewMode;
  const subjectCfg = allSubjects[subjectId];
  const cfg = mergeConfig(subjectCfg, cases[caseId]);

  setCustomUploadVisibility(subjectId);
  subjectSelectEl.value = subjectId;
  populateCaseSelect(subjectId, caseId);
  caseSelectEl.value = caseId;
  viewSelectEl.value = viewMode;
  titleEl.textContent = VIEW_MODES[viewMode].title;
  updateURL(subjectId, caseId, viewMode);

  statusEl.style.background = "rgba(0, 0, 0, 0.68)";
  statusEl.textContent = `Loading ${cfg.subjectName} / ${cfg.caseName} / ${VIEW_MODES[viewMode].label}...`;
  legendEl.innerHTML = "";
  clearClouds();

  if (subjectId === CUSTOM_SUBJECT_ID) {
    updateURL(subjectId, caseId, viewMode);
    await loadCustomViewer(viewMode);
    return;
  }

  configureCamera(cfg);

  try {
    if (viewMode === "input") await loadInputViewer(cfg);
    else if (viewMode === "reconstructed") await loadReconstructedViewer(cfg);
    else if (viewMode === "completed") await loadCompletedViewer(cfg);
  } catch (error) {
    statusEl.style.background = "rgba(150,0,0,0.72)";
    statusEl.textContent = error.message;
    console.error(error);
  }
}

async function init() {
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  try {
    allSubjects = await fetch("subjects/subjects.json", { cache: "no-store" }).then(r => r.json());
    addCustomSubject(allSubjects);
    const subjectId = getInitialSubject(allSubjects);
    const caseId = getInitialCase(allSubjects, subjectId);
    const viewMode = getInitialView();
    configureSelectors(allSubjects, subjectId, caseId, viewMode);
    await loadSelected(subjectId, caseId, viewMode);
  } catch (error) {
    statusEl.style.background = "rgba(150,0,0,0.72)";
    statusEl.textContent = error.message;
    console.error(error);
  }
}

function animate() {
  requestAnimationFrame(animate);
  applyAtlasOrbit();
  applySurfaceMovement();
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("resize", function () {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

init();
animate();
