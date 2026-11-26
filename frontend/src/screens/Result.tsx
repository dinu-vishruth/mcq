import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Check, X, Target, Sparkles, Brain, Home, BookOpen } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

interface Detail {
  question: string;
  selected: string;
  correct: string;
  is_correct: boolean;
  confidence?: "know" | "maybe" | "no_idea" | null;
  source_url?: string;
}
export interface ResultData {
  score: number;
  total: number;
  details: Detail[];
  explanations: string[];
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

  const stroke = pct >= 70 ? "#16704a" : pct >= 40 ? "#9a5b0b" : "#bc3644";

  return (
    <div className="relative w-[200px] h-[200px] mx-auto">
      <svg aria-hidden="true" width="200" height="200" viewBox="0 0 200 200" className="-rotate-90">
        <circle cx="100" cy="100" r={R} fill="none" strokeWidth="10" stroke="#e9ece5" />
        <circle
          cx="100" cy="100" r={R} fill="none" strokeWidth="10" stroke={stroke}
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

export default function Result({ data }: { data: ResultData }) {
  const { score, total, details, explanations } = data;
  const pct = total > 0 ? (score / total) * 100 : 0;
  const wrong = total - score;
  const headline = "Your learning summary";
  const misconceptions = details.filter(d => !d.is_correct && d.confidence === "know").length;
  const uncertain = details.filter(d => d.is_correct && ["maybe", "no_idea"].includes(d.confidence || "")).length;

  return (
    <div className="app-glow min-h-screen">
      <div className="relative z-10 max-w-3xl mx-auto px-5 py-10">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <span className="grid place-items-center w-8 h-8 rounded-md bg-accent/10 text-accent">
              <BookOpen className="w-4 h-4" />
            </span>
            <span className="font-display font-semibold">Examly</span>
          </div>
          <Button variant="secondary" size="sm" leftIcon={<Home className="w-4 h-4" />} onClick={() => (window.location.href = "/")}>
            Home
          </Button>
        </div>

        {/* Score hero */}
        {data.notice && <p role="status" className="mb-4 text-warning">{data.notice}</p>}
        <header className="text-center mt-10 mb-8">
          <p className="text-accent text-xs font-semibold tracking-[0.15em] uppercase mb-3">{headline}</p>
          <h1 className="study-page-title text-4xl md:text-5xl">A moment to reflect.</h1>
          <p className="text-text-2 text-sm mt-3">Take a moment to see what stuck, and what deserves another look.</p>
        </header>
        <Card pad="lg" className="text-center mb-6 bg-[#f0f2e9]">
          <ScoreRing pct={pct} />
          <div className="text-text-2 text-base mt-3">{score} of {total} correct</div>
        </Card>

        {/* Summary stats */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
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
        <Card className="mb-6 bg-gradient-to-br from-accent/[0.08] to-card border-accent/20">
          <div className="flex items-start gap-3">
            <span className="grid place-items-center w-9 h-9 rounded-sm bg-violet/10 text-violet shrink-0"><Sparkles className="w-4 h-4" /></span>
            <div>
              <h2 className="text-base mb-1">What to revisit</h2>
              <p className="text-text-2 text-sm">
                {wrong ? `${wrong} questions need another look. Read the explanations and revisit the source material before another attempt.` : "You answered this set correctly. Check your recall with different questions later."}
                {misconceptions > 0 && ` ${misconceptions} incorrect answers were marked confident; check for a possible misconception.`}
                {uncertain > 0 && ` ${uncertain} correct answers were marked unsure or guessing; revisit these to build confidence.`}

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
                {d.confidence && <p className="mt-1 text-sm text-text-3">Confidence: {{ know: "Confident", maybe: "Unsure", no_idea: "Guessing" }[d.confidence]}</p>}
                {d.confidence && <p className="mt-2 text-sm text-accent-2">{!d.is_correct && d.confidence === "know"
                  ? "Possible misconception: compare your reasoning with the correct answer."
                  : d.is_correct && d.confidence !== "know" ? "Correct with uncertainty: try this idea again later."
                  : d.is_correct ? "Correct on this attempt. Check recall again with a different question."
                  : "Revisit the explanation before another attempt."}</p>}
                {!d.is_correct && (
                  <p className="text-sm text-text-2 mt-1"><strong className="text-text">Correct answer:</strong> <span className="text-success">{d.correct}</span></p>
                )}
                {explanations[i] && (
                  <div className="mt-3 p-3 rounded-md bg-card-2 border border-line text-sm text-text-2">
                    <strong className="text-text flex items-center gap-1.5 mb-1"><Brain className="w-4 h-4" /> Explanation</strong>
                    {explanations[i]}
                    {d.source_url?.startsWith("/sources/") && <a className="block mt-2 text-accent underline" href={d.source_url}>Read the supporting passage</a>}
                  </div>
                )}
              </Card>
            </motion.div>
          ))}
        </div>

        <div className="flex flex-wrap justify-center gap-3">
          <Button leftIcon={<Sparkles className="w-4 h-4" />} onClick={() => (window.location.href = "/upload")}>
            Create another quiz
          </Button>
          <Button variant="secondary" leftIcon={<Home className="w-4 h-4" />} onClick={() => (window.location.href = "/review")}>
            Review saved mistakes
          </Button>
        </div>
      </div>
    </div>
  );
}
