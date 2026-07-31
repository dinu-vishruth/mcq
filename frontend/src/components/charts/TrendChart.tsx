import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

// Dark-mode line/area chart matching the accent blue used across the app.
export function TrendChart({ dates, scores }: { dates: string[]; scores: number[] }) {
  const data = dates.map((d, i) => ({ date: d.slice(5), score: scores[i] ?? 0 }));
  return (
    <div className="h-[220px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.28} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="date" tick={{ fill: "#71717a", fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} tick={{ fill: "#71717a", fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{
              background: "#17171b",
              border: "1px solid rgba(255,255,255,0.11)",
              borderRadius: 11,
              color: "#f4f4f5",
              fontSize: 13,
            }}
            labelStyle={{ color: "#a1a1aa" }}
            cursor={{ stroke: "rgba(255,255,255,0.12)" }}
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke="#3b82f6"
            strokeWidth={2.5}
            fill="url(#trendFill)"
            dot={{ r: 3, fill: "#3b82f6" }}
            activeDot={{ r: 5 }}
            name="Score %"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
