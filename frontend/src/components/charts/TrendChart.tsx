import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

// Study progress chart matching the accent blue used across the app.
export function TrendChart({ dates, scores }: { dates: string[]; scores: number[] }) {
  const data = dates.map((d, i) => ({ date: d.slice(5), score: scores[i] ?? 0 }));
  return (
    <div className="h-[220px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2855d9" stopOpacity={0.28} />
              <stop offset="100%" stopColor="#2855d9" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#e2e4df" vertical={false} />
          <XAxis dataKey="date" tick={{ fill: "#687381", fontSize: 12 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} tick={{ fill: "#687381", fontSize: 12 }} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{
              background: "#ffffff",
              border: "1px solid #e2e4df",
              borderRadius: 11,
              color: "#202d3b",
              fontSize: 13,
            }}
            labelStyle={{ color: "#526070" }}
            cursor={{ stroke: "#bdc7c0" }}
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke="#2855d9"
            strokeWidth={2.5}
            fill="url(#trendFill)"
            dot={{ r: 3, fill: "#2855d9" }}
            activeDot={{ r: 5 }}
            name="Score %"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
