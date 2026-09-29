const PIVOT = { x: 100, y: 110 };
const NEEDLE_RADIUS = 68;
const THRESHOLD_RADIUS = 82;
const TRACK_RADIUS = 80;

function pointOnArc(value, radius) {
  // value in [0,1] -> angle sweeps from 180deg (left, value=0) to 0deg (right, value=1)
  const angleDeg = 180 - value * 180;
  const angleRad = (angleDeg * Math.PI) / 180;
  return {
    x: PIVOT.x + radius * Math.cos(angleRad),
    y: PIVOT.y - radius * Math.sin(angleRad),
  };
}

export function renderGauge({ arcEl, needleEl, thresholdEl }, probability, threshold) {
  const clampedProbability = Math.min(1, Math.max(0, probability));
  const clampedThreshold = Math.min(1, Math.max(0, threshold));

  // Progress arc reveal
  const length = arcEl.getTotalLength();
  arcEl.style.strokeDasharray = `${length}`;
  arcEl.style.strokeDashoffset = `${length * (1 - clampedProbability)}`;

  // Needle rotation: 0 at value=0 (pointing left), 180deg sweep to value=1 (pointing right).
  // The needle SVG element is drawn vertical-ish from pivot; we rotate it directly.
  const needleAngle = -90 + clampedProbability * 180;
  needleEl.setAttribute("x1", PIVOT.x);
  needleEl.setAttribute("y1", PIVOT.y);
  needleEl.setAttribute("x2", PIVOT.x);
  needleEl.setAttribute("y2", PIVOT.y - NEEDLE_RADIUS);
  needleEl.style.transform = `rotate(${needleAngle}deg)`;

  // Threshold tick mark, positioned via trigonometry along the same arc.
  const inner = pointOnArc(clampedThreshold, TRACK_RADIUS - 14);
  const outer = pointOnArc(clampedThreshold, THRESHOLD_RADIUS);
  thresholdEl.setAttribute("x1", inner.x);
  thresholdEl.setAttribute("y1", inner.y);
  thresholdEl.setAttribute("x2", outer.x);
  thresholdEl.setAttribute("y2", outer.y);
}