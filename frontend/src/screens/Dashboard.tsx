import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Play, Sparkles, Target, Trophy, Clock, Flame, Zap, BookOpen,
  TrendingUp, Brain, ArrowRight, Calendar, FileText,
} from "lucide-react";
import type { DashboardData } from "@/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { StatCard } from "@/components/ui/StatCard";
import { TrendChart } from "@/components/charts/TrendChart";
import { Onboarding } from "./Onboarding";
import { AppShell } from "@/components/AppShell";
import { csrfToken } from "@/bootstrap";

function greeting(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

function flashError(): string {
  return document.getElementById("flash-error")?.dataset.message ?? "";
}

export default function Dashboard({ data }: { data: DashboardData }) {
  // The existing /student route doesn't compute onboarding state (we don't edit
  // it), so prefs are fetched from the additive /api/prefs endpoint on mount.
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [prefs, setPrefs] = useState(data.prefs);
  const hasHistory = data.total_quizzes > 0;
  const last = data.history[0];
  const joinError = flashError();

  useEffect(() => {
    let cancelled = false;
    fetch("/api/prefs", { headers: { Accept: "application/json" } })
      .then((r) => (r.ok ? r.json() : null))
      .then((p) => {
        if (cancelled || !p) return;
        setPrefs(p.prefs);
        if (p.needs_onboarding) setShowOnboarding(true);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <AppShell active="home" username={data.username} streak={data.streak} xp={data.xp}>
      {showOnboarding && (
        <Onboarding
          onDone={() => setShowOnboarding(false)}
          initial={prefs}
        />
      )}

      <div className="max-w-6xl mx-auto px-5 md:px-8 py-8 md:py-10">
        {/* Hero */}
        <motion.header
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="mb-8"
        >
          <p className="text-accent-2 text-sm font-medium mb-2 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4" /> Your learning companion
          </p>
          <h1 className="text-[clamp(1.9rem,3.2vw,2.5rem)] font-bold leading-tight">
            {greeting()}, {data.username} 👋
          </h1>
          <p className="text-text-2 mt-2 max-w-xl">
            {hasHistory
              ? `You've completed ${data.total_quizzes} session${data.total_quizzes !== 1 ? "s" : ""} at a ${data.avg_score}% average. Ready to keep the momentum going?`
              : "Let's get started. Upload your study material and your AI coach will build a personalized learning journey."}
          </p>
          <div className="flex flex-wrap gap-3 mt-5">
            {last ? (
              <Button leftIcon={<Play className="w-4 h-4" />} onClick={() => (window.location.href = "/student#join")}>
                Continue Learning
              </Button>
            ) : (
              <Button leftIcon={<Play className="w-4 h-4" />} onClick={() => (window.location.href = "/upload")}>
                Start Learning
              </Button>
            )}
            <Button variant="secondary" leftIcon={<Sparkles className="w-4 h-4" />} onClick={() => (window.location.href = "/upload")}>
              Add Knowledge
            </Button>
          </div>
        </motion.header>

        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard index={0} value={data.total_quizzes} label="Study Sessions" icon={<BookOpen />} tone="accent" />
          <StatCard index={1} value={`${data.avg_score}%`} label="Average Score" icon={<Target />} tone="emerald" />
          <StatCard index={2} value={`${data.best_score}%`} label="Best Score" icon={<Trophy />} tone="amber" />
          <StatCard
            index={3}
            value={data.total_time >= 60 ? `${Math.floor(data.total_time / 60)}m` : `${data.total_time}s`}
            label="Study Time"
            icon={<Clock />}
            tone="violet"
          />
        </div>

        {/* Streak + XP */}
        <div className="grid md:grid-cols-3 gap-4 mb-8">
          <Card className="md:col-span-2">
            <div className="flex items-center justify-between mb-4">
              <h2 className="flex items-center gap-2 text-lg mb-0">
                <Flame className="w-5 h-5 text-warning" /> Daily Streak
              </h2>
              <Badge tone="warning">{data.streak} day{data.streak !== 1 ? "s" : ""}</Badge>
            </div>
            <div className="flex gap-2">
              {["M", "T", "W", "T", "F", "S", "S"].map((d, i) => (
                <div
                  key={i}
                  className={`flex-1 aspect-square grid place-items-center rounded-sm text-sm font-medium ${
                    i < data.streak ? "bg-warning/20 text-warning" : "bg-white/[0.04] text-text-3"
                  }`}
                >
                  {d}
                </div>
              ))}
            </div>
            <p className="text-text-3 text-sm mt-3">Practice a little every day to keep your streak alive.</p>
          </Card>
          <Card className="flex flex-col justify-center items-center text-center">
            <span className="grid place-items-center w-12 h-12 rounded-full bg-violet/10 text-violet mb-2">
              <Zap className="w-6 h-6" />
            </span>
            <div className="text-3xl font-display font-bold">{data.xp}</div>
            <div className="text-text-3 text-sm">XP · Level {Math.floor(data.xp / 100) + 1}</div>
          </Card>
        </div>

        {/* Charts + weak areas */}
        {hasHistory && (
          <div className="grid lg:grid-cols-2 gap-4 mb-8">
            <Card>
              <h2 className="flex items-center gap-2 text-lg"><TrendingUp className="w-5 h-5 text-accent" /> Performance Trend</h2>
              <p className="text-text-3 text-sm mb-4 -mt-2">Your scores over time (%)</p>
              <TrendChart dates={data.chart_dates} scores={data.chart_scores} />
            </Card>
            <Card>
              <h2 className="flex items-center gap-2 text-lg"><Target className="w-5 h-5 text-danger" /> Weak Areas</h2>
              <p className="text-text-3 text-sm mb-4 -mt-2">Concepts to review</p>
              {data.weak_topics.length ? (
                <div className="space-y-4">
                  {data.weak_topics.map((w) => (
                    <div key={w.topic}>
                      <div className="flex items-center justify-between mb-1.5">
                        <strong className="text-sm">{w.topic}</strong>
                        <Badge tone="danger">{w.wrong}/{w.total} missed</Badge>
                      </div>
                      <ProgressBar pct={w.pct} tone="danger" />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-text-3 text-sm">No weak areas detected yet. Keep practicing.</p>
              )}
            </Card>
          </div>
        )}

        {/* AI Coach nudge */}
        <Card className="mb-8 bg-gradient-to-br from-violet/[0.1] to-card border-white/[0.11]">
          <div className="flex items-start gap-4">
            <span className="grid place-items-center w-11 h-11 rounded-md bg-violet/15 text-violet shrink-0">
              <Brain className="w-6 h-6" />
            </span>
            <div className="flex-1">
              <h2 className="text-lg mb-1">Your AI Coach</h2>
              <p className="text-text-2 text-sm mb-4">
                {data.weak_topics.length
                  ? `Focus today on ${data.weak_topics[0].topic} — it's your biggest opportunity to improve.`
                  : data.recommendations.length
                  ? data.recommendations[0]
                  : hasHistory
                  ? "You're doing great. Turn fresh notes into a session to keep improving."
                  : "Upload your first document and I'll build a learning path tailored to you."}
              </p>
              <Button variant="secondary" size="sm" rightIcon={<ArrowRight className="w-4 h-4" />} onClick={() => (window.location.href = "/upload")}>
                {data.weak_topics.length ? "Practice weak topics" : "Add knowledge"}
              </Button>
            </div>
          </div>
        </Card>

        {/* Recommended topics */}
        {data.recommendations.length > 0 && (
          <div className="mb-8">
            <h2 className="text-lg mb-3">Recommended for you</h2>
            <div className="flex flex-wrap gap-2">
              {data.recommendations.map((r, i) => (
                <Badge key={i} tone="accent" className="text-sm px-3 py-1.5">{r}</Badge>
              ))}
            </div>
          </div>
        )}

        {/* Join a session — preserves the existing /student_login POST flow */}
        <Card className="mb-8" id="join">
          <h2 className="flex items-center gap-2 text-lg"><Play className="w-5 h-5 text-accent" /> Join a Session</h2>
          <p className="text-text-3 text-sm mb-4 -mt-2">Enter a session key to start a shared quiz.</p>
          {joinError && (
            <div className="mb-3 px-3.5 py-2.5 rounded-md bg-danger/10 border border-danger/20 text-danger text-sm">
              {joinError}
            </div>
          )}
          <form action="/student_login" method="post" className="flex flex-col sm:flex-row gap-3">
            <input type="hidden" name="csrf_token" value={csrfToken()} />
            <input
              type="text"
              name="session_key"
              placeholder="e.g. a1b2c3d4"
              required
              className="flex-1 h-11 px-4 rounded-md bg-inset border border-white/[0.08] text-text placeholder:text-text-3 focus:outline-none focus:border-accent/50 focus:ring-2 focus:ring-accent/20"
            />
            <Button type="submit" rightIcon={<ArrowRight className="w-4 h-4" />}>Start Quiz</Button>
          </form>
        </Card>

        {/* Recent activity */}
        <h2 className="text-lg mb-3">Recent Activity</h2>
        {data.history.length ? (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.history.map((h, i) => {
              const pct = h.total ? h.score / h.total : 0;
              return (
                <motion.div
                  key={`${h.session_key}-${i}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04, duration: 0.4 }}
                >
                  <Card hover>
                    <div className="flex items-center justify-between mb-3">
                      <span className="grid place-items-center w-9 h-9 rounded-sm bg-white/5 text-text-2">
                        <FileText className="w-4 h-4" />
                      </span>
                      <Badge tone={pct >= 0.7 ? "success" : "danger"}>{h.score} / {h.total}</Badge>
                    </div>
                    <code className="text-sm text-text-2 block mb-3">{h.session_key}</code>
                    <div className="flex flex-wrap gap-3 text-xs text-text-3">
                      <span className="flex items-center gap-1"><Calendar className="w-3.5 h-3.5" /> {h.submitted_at.slice(0, 16).replace("T", " ")}</span>
                      <span className="flex items-center gap-1"><Clock className="w-3.5 h-3.5" /> {h.time_spent || 0}s</span>
                    </div>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <Card className="text-center py-12">
            <span className="grid place-items-center w-14 h-14 rounded-full bg-white/5 text-text-2 mx-auto mb-4">
              <BookOpen className="w-7 h-7" />
            </span>
            <h2 className="text-lg mb-1">Your AI is waiting to learn</h2>
            <p className="text-text-3 text-sm mb-5">Add study material to start building your learning journey.</p>
            <Button leftIcon={<Sparkles className="w-4 h-4" />} onClick={() => (window.location.href = "/upload")}>
              Add Knowledge
            </Button>
          </Card>
        )}
      </div>
    </AppShell>
  );
}
