// Seeded so server and client render the same field of stars.
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(42);
const stars = Array.from({ length: 80 }, () => ({
  left: `${(rand() * 100).toFixed(2)}%`,
  top: `${(rand() * 58).toFixed(2)}%`,
  size: rand() < 0.8 ? 1.5 : 2.5,
  opacity: (0.25 + rand() * 0.55).toFixed(2),
  duration: `${(3 + rand() * 5).toFixed(1)}s`,
  delay: `${(rand() * 6).toFixed(1)}s`,
}));

const streaks = [
  { left: "43.5%", height: 150, opacity: 0.5 },
  { left: "46%", height: 90, opacity: 0.35 },
  { left: "58%", height: 120, opacity: 0.3 },
  { left: "76.5%", height: 70, opacity: 0.45 },
];

export function AuroraBackground() {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[max(820px,100svh)] overflow-hidden">
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, #02061a 0%, #031035 14%, #06279a 28%, #0a4dff 40%, #1e7bff 50%, #8fbcff 62%, #eaf2ff 74%, #fafcff 100%)",
        }}
      />
      {/* dark dome over the top centre */}
      <div
        className="absolute left-1/2 top-[-38%] h-[78%] w-[110%] -translate-x-1/2 rounded-[50%]"
        style={{ background: "radial-gradient(closest-side, rgba(2,6,26,0.95) 30%, rgba(2,6,26,0.55) 70%, rgba(2,6,26,0) 100%)" }}
      />
      {/* horizon of light */}
      <div
        className="absolute left-1/2 top-[47%] h-[130%] w-[190%] -translate-x-1/2 rounded-[50%] blur-[40px]"
        style={{
          background: "#fafcff",
          boxShadow: "0 -50px 140px 40px rgba(30,123,255,0.95), 0 -16px 50px 10px rgba(160,200,255,0.9)",
        }}
      />
      {streaks.map((s) => (
        <span
          key={s.left}
          className="absolute top-0 w-px"
          style={{
            left: s.left,
            height: s.height,
            opacity: s.opacity,
            background: "linear-gradient(180deg, rgba(140,185,255,0), rgba(140,185,255,0.9) 40%, rgba(140,185,255,0))",
          }}
        />
      ))}
      {stars.map((s, i) => (
        <span
          key={i}
          className="absolute rounded-full bg-white motion-safe:animate-twinkle"
          style={
            {
              left: s.left,
              top: s.top,
              width: s.size,
              height: s.size,
              opacity: s.opacity,
              "--o": s.opacity,
              animationDuration: s.duration,
              animationDelay: s.delay,
            } as React.CSSProperties
          }
        />
      ))}
    </div>
  );
}
