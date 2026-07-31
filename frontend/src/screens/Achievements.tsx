import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Trophy, Loader2, Flame, Zap, Footprints, Calendar, Target, Award, Lock,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { apiGet } from "@/bootstrap";

interface Badge { key: string; label: string; desc: string; earned: boolean; icon: string }
interface Milestone { label: string; value: number; next: number }
interface AchData {
  xp: number; level: number; streak: number; quizzes: number;
  badges: Badge[]; milestones: Milestone[]; earned_count: number; total_badges: number;
}

const ICONS: Record<string, typeof Trophy> = {
  footprints: Footprints, flame: Flame, calendar: Calendar,
  target: Target, zap: Zap, trophy: Trophy,
};

export default function Achievements({ data }: { data: { username: string } }) {
  const [d, setD] = useState<AchData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGet<AchData>("/api/achievements")
      .then(setD)
      .catch(() => setD(null))
      .finally(() => setLoading(false));
  }, []);

  return (
    <AppShell active="achievements" username={data.username} streak={d?.streak} xp={d?.xp}>
      <div className="max-w-4xl mx-auto px-5 lg:px-8 py-8">
        <h1 className="font-display text-2xl font-semibold flex items-center gap-2.5">
          <Trophy className="w-6 h-6 text-warning" /> Achievements
        </h1>
        <p className="text-text-2 mt-1.5 mb-8">Your consistency and milestones, at a glance.</p>

        {loading || !d ? (
          <div className="grid place-items-center py-24 text-text-3"><Loader2 className="w-6 h-6 animate-spin" /></div>
        ) : (
          <>
            {/* Headline stats */}
            <div className="grid gap-4 sm:grid-cols-3 mb-8">
              <HeadStat icon={<Flame className="w-5 h-5" />} tone="text-warning bg-warning/10" value={`${d.streak}`} label="Day streak" />
              <HeadStat icon={<Zap className="w-5 h-5" />} tone="text-violet bg-violet/10" value={`${d.xp}`} label={`XP · Level ${d.level}`} />
              <HeadStat icon={<Award className="w-5 h-5" />} tone="text-accent bg-accent/10" value={`${d.earned_count}/${d.total_badges}`} label="Badges earned" />
            </div>

            {/* Badges */}
            <h2 className="font-display font-semibold mb-3">Badges</h2>
            <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 mb-8">
              {d.badges.map((b, i) => {
                const Icon = ICONS[b.icon] ?? Trophy;
                return (
                  <motion.div key={b.key} initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.04 }}>
                    <Card pad="md" className={`text-center h-full ${b.earned ? "" : "opacity-60"}`}>
                      <span className={`grid place-items-center w-12 h-12 rounded-lg mx-auto mb-3 ${
                        b.earned ? "bg-warning/12 text-warning" : "bg-white/[0.04] text-text-3"}`}>
                        {b.earned ? <Icon className="w-6 h-6" /> : <Lock className="w-5 h-5" />}
                      </span>
                      <div className="font-medium text-sm">{b.label}</div>
                      <div className="text-text-3 text-xs mt-0.5">{b.desc}</div>
                    </Card>
                  </motion.div>
                );
              })}
            </div>

            {/* Milestones */}
            <h2 className="font-display font-semibold mb-3">Milestones</h2>
            <div className="space-y-3">
              {d.milestones.map((m) => {
                const pct = m.next ? Math.min(100, (m.value / m.next) * 100) : 100;
                return (
                  <Card key={m.label} pad="md">
                    <div className="flex items-center justify-between mb-2 text-sm">
                      <span className="font-medium">{m.label}</span>
                      <span className="text-text-3">{m.value} / {m.next}</span>
                    </div>
                    <ProgressBar pct={pct} tone="accent" />
                  </Card>
                );
              })}
            </div>
          </>
        )}
      </div>
    </AppShell>
  );
}

function HeadStat({ icon, tone, value, label }: { icon: React.ReactNode; tone: string; value: string; label: string }) {
  return (
    <Card pad="md" className="flex items-center gap-4">
      <span className={`grid place-items-center w-12 h-12 rounded-lg shrink-0 ${tone}`}>{icon}</span>
      <div>
        <div className="font-display text-2xl font-semibold leading-none">{value}</div>
        <div className="text-text-3 text-xs mt-1">{label}</div>
      </div>
    </Card>
  );
}
