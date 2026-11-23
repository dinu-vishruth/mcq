import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Check, X, Target, Sparkles, Brain, Home } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { apiSend } from "@/bootstrap";

interface Detail {
  question: string;
  selected: string;
  correct: string;
  is_correct: boolean;
  confidence?: "know" | "maybe" | "no_idea" | null;
  verification?: "verified" | "unverified";
  sources?: { quote: string; document_id?: number | null; chunk_index?: number | null }[];
}
export interface ResultData {
  session_key?: string;
  enrichment_pending?: boolean;
  score: number;
  total: number;
  details: Detail[];
  explanations: string[];
  xp_awarded?: number;
  notice?: string | null;
}

function ScoreRing({ pct }: { pct: number }) {
  const R = 88;
  const C = 2 * Math.PI * R;
  const [offset, setOffset] = useState(C);
  const [shown, setShown] = useState(0);

  useEffect(() => {
    const raf = requestAnimationFrame(() => setOffset(C * (1 - pct / 100)));
    const start = performance.now();
    const dur = 1100;
    function tick(now: number) {
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(Math.round(pct * eased));
      if (t < 1) requestAnimationFrame(tick);
    }
    const r2 = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      cancelAnimationFrame(r2);
    };
  }, [pct, C]);

  const stroke = pct >= 70 ? "#10b981" : pct >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <div className="relative w-[200px] h-[200px] mx-auto">
      <svg width="200" height="200" viewBox="0 0 200 200" className="-rotate-90">
        <circle cx="100" cy="100" r={R} fill="none" strokeWidth="14" stroke="rgba(255,255,255,0.06)" />
        <circle
          cx="100" cy="100" r={R} fill="none" strokeWidth="14" stroke={stroke}
          strokeLinecap="round" strokeDasharray={C} strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1.1s cubic-bezier(0.16,1,0.3,1)" }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="text-4xl font-display font-bold">{shown}%</div>
        </div>
      </div>
    </div>
  );
}

function Confetti() {
  const colors = ["#3b82f6", "#60a5fa", "#10b981", "#a78bfa", "#f59e0b"];
  const pieces = Array.from({ length: 80 }, (_, i) => i);
  return (
    <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
      {pieces.map((i) => (
        <motion.span
          key={i}
          className="absolute w-2 h-3 rounded-sm"
          style={{ left: `${(i * 1.25) % 100}vw`, background: colors[i % colors.length] }}
          initial={{ y: -20, opacity: 1, rotate: 0 }}
          animate={{ y: "105vh", opacity: [1, 1, 0], rotate: 360 }}
          transition={{ duration: 2.6 + (i % 5) * 0.3, delay: (i % 8) * 0.05, ease: "easeIn" }}
        />
      ))}
    </div>
  );
}

export default function Result({ data }: { data: ResultData }) {
  const { score, total } = data;
  const [details, setDetails] = useState(data.details);
  const [explanations, setExplanations] = useState(data.explanations);
  const [feedback, setFeedback] = useState<"loading" | "done" | "failed">(data.enrichment_pending ? "loading" : "done");
  useEffect(() => {
    if (!data.session_key || !data.enrichment_pending) return;
    let cancelled = false;
    async function loadFeedback() {
      const until = Date.now() + 60000;
      try {
        while (!cancelled && Date.now() < until) {
          const response = await apiSend<{ details: Detail[]; explanations: string[]; pending: boolean; failed: boolean }>(
            `/api/results/${encodeURIComponent(data.session_key!)}/enrich`, "POST", {}, 55000,
          );
          if (cancelled) return;
          setDetails(response.details);
          setExplanations(response.explanations);
          if (!response.pending) { setFeedback(response.failed ? "failed" : "done"); return; }
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
        if (!cancelled) setFeedback("failed");
      } catch {
        if (!cancelled) setFeedback("failed");
      }
    }
    void loadFeedback();
    return () => { cancelled = true; };
  }, [data.session_key, data.enrichment_pending]);
  const pct = total > 0 ? (score / total) * 100 : 0;
  const wrong = total - score;
  const headline = pct >= 70 ? "Outstanding work" : pct >= 40 ? "Good effort" : "Keep practicing";

  return (
    <div className="app-glow min-h-screen">
      {pct >= 80 && <Confetti />}
      <div className="relative z-10 max-w-3xl mx-auto px-5 py-10">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <span className="grid place-items-center w-8 h-8 rounded-md bg-accent/10 text-accent">
              <Brain className="w-4 h-4" />
            </span>
            <span className="font-display font-semibold">MCQ Generator</span>
          </div>
          <Button variant="secondary" size="sm" leftIcon={<Home className="w-4 h-4" />} onClick={() => (window.location.href = "/")}>
            Home
          </Button>
        </div>

        {/* Score hero */}
        {data.notice && <p role="status" className="mb-4 text-warning">{data.notice}</p>}
        <Card pad="lg" className="text-center mb-6">
          <p className="text-accent-2 text-sm font-medium mb-4">{headline}</p>
          <ScoreRing pct={pct} />
          <div className="text-text-3 text-sm mt-2">{score} / {total} correct</div>
          {!!data.xp_awarded && <p className="mt-2 text-success">+{data.xp_awarded} XP earned</p>}
        </Card>

        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <Card>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-display font-bold text-success">{score}</div>
                <div className="text-text-3 text-sm">Correct</div>
              </div>
              <span className="grid place-items-center w-9 h-9 rounded-sm bg-success/10 text-success"><Check className="w-4 h-4" /></span>
            </div>
          </Card>
          <Card>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-display font-bold text-danger">{wrong}</div>
                <div className="text-text-3 text-sm">To Review</div>
              </div>
              <span className="grid place-items-center w-9 h-9 rounded-sm bg-danger/10 text-danger"><X className="w-4 h-4" /></span>
            </div>
          </Card>
          <Card>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-display font-bold">{Math.round(pct)}%</div>
                <div className="text-text-3 text-sm">Accuracy</div>
              </div>
              <span className="grid place-items-center w-9 h-9 rounded-sm bg-violet/10 text-violet"><Target className="w-4 h-4" /></span>
            </div>
          </Card>
        </div>

        {/* AI insight */}
        {feedback === "loading" && <p role="status" className="mb-4 text-sm text-text-2">Your score is saved. Preparing learning feedback…</p>}
        {feedback === "failed" && <p role="status" className="mb-4 text-sm text-warning">Optional learning feedback could not load. Your score and answers are saved.</p>}
        <Card className="mb-6 bg-gradient-to-br from-accent/[0.08] to-card border-accent/20">
          <div className="flex items-start gap-3">
            <span className="grid place-items-center w-9 h-9 rounded-sm bg-violet/10 text-violet shrink-0"><Sparkles className="w-4 h-4" /></span>
            <div>
              <h2 className="text-base mb-1">AI Insight</h2>
              <p className="text-text-2 text-sm">
                {pct >= 70
                  ? `Strong recall across the set. Focus your next session on the ${wrong} missed concept${wrong !== 1 ? "s" : ""} to reach mastery.`
                  : pct >= 40
                  ? "You're getting the core ideas. The explanations below target exactly where understanding slipped."
                  : "This topic needs another pass. Read each explanation, then generate a fresh quiz to reinforce what you learn."}
              </p>
            </div>
          </div>
        </Card>

        {/* Question review */}
        <h2 className="text-lg mb-3">Question Review</h2>
        <div className="space-y-3 mb-8">
          {details.map((d, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
            >
              <Card className={d.is_correct ? "border-success/20" : "border-danger/20"}>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-text-3 text-sm font-medium">Q{i + 1}</span>
                  <Badge tone={d.is_correct ? "success" : "danger"}>
                    {d.is_correct ? <Check className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
                    {d.is_correct ? "Correct" : "Incorrect"}
                  </Badge>
                </div>
                <p className="text-[1.02rem] mb-3">{d.question}</p>
                <p className="text-sm text-text-2"><strong className="text-text">Your answer:</strong> {d.selected || "Not answered"}</p>
                {d.confidence && <p className="mt-1 text-sm text-text-3">Confidence: {{ know: "I Know", maybe: "Maybe", no_idea: "No Idea" }[d.confidence]}</p>}
                {!d.is_correct && (
                  <p className="text-sm text-text-2 mt-1"><strong className="text-text">Correct answer:</strong> <span className="text-success">{d.correct}</span></p>
                )}
                {explanations[i] && (
                  <div className="mt-3 p-3 rounded-md bg-white/[0.03] border border-white/[0.06] text-sm text-text-2">
                    <strong className="text-text flex items-center gap-1.5 mb-1"><Brain className="w-4 h-4" /> Explanation</strong>
                    {explanations[i]}
                  </div>
                )}
                {!!d.sources?.length && (
                  <div className="mt-3 text-sm text-text-2">
                    <p className="font-medium text-text mb-1">Supporting excerpt</p>
                    {d.sources.map((source, index) => <blockquote key={index} className="border-l-2 border-accent/50 pl-3 mb-2">{source.quote}</blockquote>)}
                  </div>
                )}
                <p className="mt-3 text-xs text-text-2">
                  {d.verification === "verified" ? "AI source check passed. Use the excerpt to review the answer." : "AI-generated answer. Check it against your study material."}
                </p>
              </Card>
            </motion.div>
          ))}
        </div>

        <div className="flex justify-center gap-3">
          <Button leftIcon={<Sparkles className="w-4 h-4" />} onClick={() => (window.location.href = "/upload")}>
            Generate New Quiz
          </Button>
          <Button variant="secondary" leftIcon={<Home className="w-4 h-4" />} onClick={() => (window.location.href = "/")}>
            Back to Home
          </Button>
        </div>
      </div>
    </div>
  );
}
