/* splash.js — 360° heritage landing for the reconstruction viewer.
 * Uses the global THREE r128 + THREE.OrbitControls that index.html loads.
 *
 * The background monument is shown DAMAGED and then RECONSTRUCTED: the right
 * wall has a breach that is filled with amber "reconstructed" points, so the
 * picture states the product's purpose — rebuilding wounded heritage into a
 * complete digital twin. Intact structure is coloured by elevation; AI-filled
 * structure glows amber.
 *
 * Also drives the simple multi-page navigation (Overview / Deep learning /
 * The atelier) and the Enter / Back transitions.
 */
(function () {
  if (!window.THREE) return;
  var canvas = document.getElementById("splashCanvas");
  var splash = document.getElementById("splash");
  if (!canvas || !splash) return;

  var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x120d08, 1);
  if (THREE.ACESFilmicToneMapping) {
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
  }

  var scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x120d08, 0.006);
  var camera = new THREE.PerspectiveCamera(46, 1, 0.1, 600);
  camera.position.set(34, 18, 46);

  var controls = new THREE.OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.06;
  controls.enablePan = false; controls.autoRotate = true; controls.autoRotateSpeed = 0.5;
  controls.minDistance = 28; controls.maxDistance = 110;
  controls.minPolarAngle = 0.35; controls.maxPolarAngle = Math.PI / 2.05;
  controls.target.set(0, 8, 0);

  function rng(seed) { var s = seed >>> 0; return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  var rand = rng(7);

  function discTexture() {
    var s = 64, c = document.createElement("canvas"); c.width = c.height = s;
    var g = c.getContext("2d");
    var grd = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    grd.addColorStop(0, "rgba(255,255,255,1)"); grd.addColorStop(0.4, "rgba(255,255,255,.8)"); grd.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grd; g.beginPath(); g.arc(s / 2, s / 2, s / 2, 0, 6.283); g.fill();
    return new THREE.CanvasTexture(c);
  }

  var STOPS = [[0, 0xb9532e], [0.28, 0xc45c80], [0.55, 0xe0913a], [0.78, 0x6a9c9a], [1, 0x6a64b8]];
  function elev(y) {
    var t = Math.min(Math.max(y / 34, 0), 1);
    for (var i = 0; i < STOPS.length - 1; i++) {
      var a = STOPS[i][0], ca = STOPS[i][1], b = STOPS[i + 1][0], cb = STOPS[i + 1][1];
      if (t <= b) { var k = (t - a) / (b - a || 1); return new THREE.Color(ca).lerp(new THREE.Color(cb), k); }
    }
    return new THREE.Color(STOPS[STOPS.length - 1][1]);
  }
  var AMBER = new THREE.Color(0xe0913a);

  var pos = [], col = [];
  function add(x, y, z, mul) { mul = mul == null ? 1 : mul; pos.push(x, y, z); var c = elev(y).multiplyScalar(mul); col.push(c.r, c.g, c.b); }
  function addRecon(x, y, z, mul) { mul = mul == null ? 1.35 : mul; pos.push(x, y, z); col.push(AMBER.r * mul, AMBER.g * mul, AMBER.b * mul); }

  var W = 15, L = 38, H = 17;
  function inSideWindow(z) { var bay = ((z + L / 2) % 5) - 2.5; return Math.abs(bay) < 1.3; }
  function archTop(z) { var bay = ((z + L / 2) % 5) - 2.5; return 11 + 2.4 * Math.sqrt(Math.max(0, 1 - Math.pow(bay / 1.3, 2))); }
  // breach on the right wall (sx = +1): the "damaged" region
  function inBreach(z, y) { return z > -7 && z < 7 && y > 4 && y < 13.5; }

  // side walls — right wall carries the breach
  for (var sx = -1; sx <= 1; sx += 2) {
    for (var i = 0; i < 19000; i++) {
      var z = (rand() - 0.5) * L, y = rand() * H;
      if (inSideWindow(z) && y > 4 && y < archTop(z)) continue;
      if (sx > 0 && inBreach(z, y) && rand() < 0.82) continue;        // missing (damaged)
      add(sx * W / 2 + (rand() - 0.5) * 0.08, y, z, 0.9 + rand() * 0.25);
    }
  }
  // AI-reconstructed infill across the breach (amber, slightly inferred/jittered)
  for (i = 0; i < 5200; i++) {
    var bz = (rand() - 0.5) * 13, by = 4 + rand() * 9.3;
    addRecon(W / 2 + (rand() - 0.5) * 0.5, by, bz, 1.15 + rand() * 0.5);
  }

  // pitched roof
  for (i = 0; i < 11000; i++) { var s = rand(), side = rand() < 0.5 ? -1 : 1; add(side * (W / 2) * (1 - s), H + s * 4.5, (rand() - 0.5) * L, 0.85 + rand() * 0.3); }
  // rounded apse
  for (i = 0; i < 6000; i++) { var a = (rand() - 0.5) * Math.PI, R = W / 2; add(Math.sin(a) * R, rand() * H * 0.9, L / 2 + Math.cos(a) * R * 0.7, 0.9); }
  // west facade + rose window + portal
  for (i = 0; i < 7000; i++) { var fx = (rand() - 0.5) * W, fy = rand() * H; if (Math.hypot(fx, fy - 12) < 3) continue; if (Math.abs(fx) < 2.2 && fy < 6) continue; add(fx, fy, -L / 2 - (rand() - 0.5) * 0.08, 0.9 + rand() * 0.25); }
  for (i = 0; i < 1400; i++) { var ra = rand() * 6.283, rr = 2.4 + (rand() - 0.5) * 0.5; add(Math.cos(ra) * rr, 12 + Math.sin(ra) * rr, -L / 2, 1.5); }
  for (var kk = 0; kk < 12; kk++) { var saa = kk / 12 * 6.283; for (var r = 0.3; r < 2.6; r += 0.12) add(Math.cos(saa) * r, 12 + Math.sin(saa) * r, -L / 2, 1.3); }

  // two west towers
  function tower(cx) {
    var TH = 32, hw = 2.6, cz = -L / 2 + 1;
    for (var j = 0; j < 9000; j++) {
      var f = Math.floor(rand() * 4), ty = rand() * TH, tx, tz, u;
      if (f < 2) { tx = cx + (rand() - 0.5) * 2 * hw; tz = cz + (f === 0 ? hw : -hw); u = tx - cx; }
      else { tz = cz + (rand() - 0.5) * 2 * hw; tx = cx + (f === 2 ? hw : -hw); u = tz - cz; }
      if (ty > TH - 7 && ty < TH - 1 && Math.abs(u) < hw * 0.55) continue;
      add(tx, ty, tz, 0.9 + rand() * 0.3);
    }
    for (j = 0; j < 2600; j++) { var cs = rand(), ca2 = rand() * 6.283, cr = hw * 1.1 * (1 - cs); add(cx + Math.cos(ca2) * cr, TH + cs * 8, cz + Math.sin(ca2) * cr, 1.1 + rand() * 0.4); }
  }
  tower(-(W / 2 + 0.6)); tower(W / 2 + 0.6);
  // crossing spire
  for (i = 0; i < 4200; i++) { var ss = rand(), sa2 = rand() * 6.283, sr = 2.4 * (1 - ss); add(Math.cos(sa2) * sr, H + 4.5 + ss * 15, 3 + Math.sin(sa2) * sr, 1.0 + rand() * 0.5); }
  // ground + motes
  for (i = 0; i < 5000; i++) { var gx = (rand() - 0.5) * W * 2.2, gz = (rand() - 0.5) * L * 1.4; if (Math.abs(gx) > W / 2 || Math.abs(gz) > L / 2) add(gx, 0.02, gz, 0.4 + rand() * 0.2); }
  for (i = 0; i < 600; i++) add((rand() - 0.5) * 90, rand() * 45, (rand() - 0.5) * 80, 0.5);

  // garden trees around the church
function tree(cx, cz, h, seedMul) {
  for (var t = 0; t < 900; t++) {
    var yy = rand() * h;
    var trunkR = 0.13 + rand() * 0.06;
    var aa = rand() * 6.283;
    add(cx + Math.cos(aa) * trunkR, yy, cz + Math.sin(aa) * trunkR, 0.45);
  }

  for (t = 0; t < 2600; t++) {
    var ly = h * 0.55 + rand() * h * 0.55;
    var radius = (1 - Math.abs(ly - h * 0.82) / (h * 0.45)) * (1.8 + seedMul);
    radius = Math.max(radius, 0.2);
    aa = rand() * 6.283;
    var rr = Math.sqrt(rand()) * radius;
    var green = new THREE.Color(0x5f7f55).lerp(new THREE.Color(0xe0913a), rand() * 0.18);
    pos.push(cx + Math.cos(aa) * rr, ly, cz + Math.sin(aa) * rr);
    col.push(green.r * .9, green.g * .9, green.b * .9);
  }
}

tree(-15, -10, 7, .4);
tree(-18, 8, 8, .2);
tree(15, -8, 6, .5);
tree(18, 11, 7.5, .3);

// small flowers / restoration garden lights
for (i = 0; i < 1800; i++) {
  var fx2 = (rand() - 0.5) * 42;
  var fz2 = (rand() - 0.5) * 48;
  if (Math.abs(fx2) < W / 2 + 2 && Math.abs(fz2) < L / 2 + 2) continue;
  var flower = rand() < .5 ? new THREE.Color(0xc45c80) : new THREE.Color(0xe0913a);
  pos.push(fx2, 0.08 + rand() * 0.12, fz2);
  col.push(flower.r, flower.g, flower.b);
}

  var geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  var mat = new THREE.PointsMaterial({ size: 0.17, vertexColors: true, sizeAttenuation: true, map: discTexture(), alphaTest: 0.04, transparent: true, opacity: 0.96, depthWrite: false });
  scene.add(new THREE.Points(geo, mat));

  function resize() { var w = window.innerWidth, h = window.innerHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  window.addEventListener("resize", resize); resize();

  var running = true;
  (function loop() { if (!running) return; requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

  /* ---------------- page navigation ---------------- */

  function showPage(name) {
	var pages = splash.querySelectorAll(".page");
	for (var i = 0; i < pages.length; i++) {
		pages[i].classList.toggle(
		"active",
		pages[i].getAttribute("data-page") === name
		);
	}

	var nav = splash.querySelectorAll(".splash-nav button");
	for (i = 0; i < nav.length; i++) {
		nav[i].classList.toggle(
		"active",
		nav[i].getAttribute("data-page") === name
		);
	}

	splash.setAttribute("data-page", name);
	}

	var navBtns = splash.querySelectorAll(".splash-nav button, .btn-ghost[data-page]");
	for (var n = 0; n < navBtns.length; n++) {
	(function (b) {
		b.addEventListener("click", function (e) {
		e.preventDefault();
		e.stopPropagation();
		showPage(b.getAttribute("data-page"));
		});
	})(navBtns[n]);
	}

	var backs = splash.querySelectorAll("[data-back]");
	for (var bk = 0; bk < backs.length; bk++) {
	backs[bk].addEventListener("click", function (e) {
		e.preventDefault();
		e.stopPropagation();
		showPage("home");
	});
	}

showPage("home");

  /* ---------------- enter / back to viewer ---------------- */
  function enter() {
    splash.classList.add("gone");
    var vb = document.getElementById("viewerBack");
    if (vb) vb.classList.remove("hidden");
    setTimeout(function () { running = false; controls.dispose(); renderer.dispose(); }, 900);
  }
  var btn = document.getElementById("enterBtn");
  var cue = document.getElementById("enterCue");
  if (btn) btn.addEventListener("click", enter);
  if (cue) cue.addEventListener("click", enter);
  window.addEventListener("keydown", function (e) { if (e.key === "Enter" && !splash.classList.contains("gone")) enter(); });

  var vback = document.getElementById("viewerBack");
  if (vback) vback.addEventListener("click", function () { window.location.reload(); });
})();
