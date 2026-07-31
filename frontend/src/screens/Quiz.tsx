import { useState, useEffect, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Clock, ArrowRight, Check } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { csrfToken } from "@/bootstrap";

interface Option { label: string; text: string; }
interface Question { question: string; options: Option[]; answer_text: string; }
export interface QuizData {
  mcqs: Question[];
  timer: number;
  username: string;
}

type Confidence = "know" | "maybe" | "no_idea";

const CONFIDENCE: { id: Confidence; label: string; tone: string }[] = [
  { id: "know", label: "I Know", tone: "text-success border-success/30 bg-success/10" },
  { id: "maybe", label: "Maybe", tone: "text-warning border-warning/30 bg-warning/10" },
  { id: "no_idea", label: "No Idea", tone: "text-text-3 border-white/10 bg-white/5" },
];

const OPT_KEYS = ["A", "B", "C", "D"];

export default function Quiz({ data }: { data: QuizData }) {
  const total = data.mcqs.length;
  const [idx, setIdx] = useState(0);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [confidence, setConfidence] = useState<Record<number, Confidence>>({});
  const [remaining, setRemaining] = useState(data.timer);
  const [xpFlash, setXpFlash] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);
  const startRef = useRef(data.timer);

  const answeredCount = Object.keys(answers).length;
  const q = data.mcqs[idx];
  const selected = answers[idx];

  // Countdown timer — auto-submits at zero, preserving the legacy behavior.
  useEffect(() => {
    if (remaining <= 0) {
      submit();
      return;
    }
    const t = setTimeout(() => setRemaining((r) => r - 1), 1000);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [remaining]);

  const mmss = useMemo(() => {
    const m = Math.floor(Math.max(0, remaining) / 60);
    const s = Math.max(0, remaining) % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }, [remaining]);

  function choose(text: string) {
    setAnswers((a) => ({ ...a, [idx]: text }));
    setXpFlash(true);
    setTimeout(() => setXpFlash(false), 700);
  }

  function next() {
    if (idx < total - 1) setIdx(idx + 1);
  }

  function submit() {
    if (!formRef.current) return;
    const spent = Math.max(0, startRef.current - remaining);
    (formRef.current.elements.namedItem("time_spent") as HTMLInputElement).value = String(spent);
    formRef.current.submit();
  }

  const lowTime = remaining <= 30;

  return (
    <div className="app-glow min-h-screen flex flex-col">
      {/* Top bar: brand, progress, timer */}
      <div className="relative z-10 border-b border-white/[0.06]">
        <div className="max-w-3xl mx-auto px-5 py-4 flex items-center gap-4">
          <span className="grid place-items-center w-9 h-9 rounded-md bg-accent/10 text-accent shrink-0">
            <Brain className="w-5 h-5" />
          </span>
          <div className="flex-1">
            <ProgressBar pct={(answeredCount / total) * 100} />
          </div>
          <span className="text-sm text-text-3 tabular-nums">{answeredCount} / {total}</span>
          <span className={`flex items-center gap-1.5 text-sm tabular-nums font-medium ${lowTime ? "text-danger" : "text-text-2"}`}>
            <Clock className="w-4 h-4" /> {mmss}
          </span>
        </div>
      </div>

      {/* Hidden form carrying the exact legacy /submit contract */}
      <form ref={formRef} action="/submit" method="post" className="hidden">
        <input type="hidden" name="csrf_token" value={csrfToken()} />
        <input type="hidden" name="student_name" value={data.username} />
        <input type="hidden" name="time_spent" defaultValue="0" />
        {data.mcqs.map((_, i) => (
          <input key={i} type="hidden" name={`q-${i}`} value={answers[i] ?? ""} readOnly />
        ))}
      </form>

      {/* Question */}
      <div className="relative z-10 flex-1 max-w-3xl w-full mx-auto px-5 py-10 flex flex-col">
        <AnimatePresence mode="wait">
          <motion.div
            key={idx}
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -24 }}
            transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
            className="flex-1"
          >
            <div className="text-accent-2 text-sm font-medium mb-3">Question {idx + 1}</div>
            <h1 className="text-[clamp(1.4rem,3vw,2rem)] font-display font-semibold leading-snug mb-8">
              {q.question}
            </h1>

            <div className="space-y-3">
              {q.options.map((opt, i) => {
                const on = selected === opt.text;
                return (
                  <button
                    key={i}
                    onClick={() => choose(opt.text)}
                    className={`w-full flex items-center gap-4 p-4 rounded-md border text-left transition-all duration-200 ${
                      on
                        ? "border-accent bg-accent/10 shadow-glow"
                        : "border-white/[0.08] hover:border-white/20 hover:bg-white/[0.03]"
                    }`}
                  >
                    <span className={`grid place-items-center w-8 h-8 rounded-sm text-sm font-semibold shrink-0 ${
                      on ? "bg-accent text-white" : "bg-white/5 text-text-2"
                    }`}>
                      {OPT_KEYS[i]}
                    </span>
                    <span className="text-[0.98rem]">{opt.text}</span>
                    {on && <Check className="w-5 h-5 text-accent ml-auto shrink-0" />}
                  </button>
                );
              })}
            </div>

            {/* Confidence selector */}
            <div className="mt-8">
              <div className="text-text-3 text-xs uppercase tracking-wider mb-2">How confident are you?</div>
              <div className="flex gap-2">
                {CONFIDENCE.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => setConfidence((cf) => ({ ...cf, [idx]: c.id }))}
                    className={`flex-1 py-2.5 rounded-md border text-sm font-medium transition-all ${
                      confidence[idx] === c.id ? c.tone : "border-white/[0.06] text-text-3 hover:text-text-2"
                    }`}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Nav */}
        <div className="flex items-center justify-between mt-10 pt-6 border-t border-white/[0.06]">
          <button
            onClick={() => idx > 0 && setIdx(idx - 1)}
            disabled={idx === 0}
            className="text-text-3 text-sm hover:text-text-2 disabled:opacity-40 transition-colors"
          >
            Previous
          </button>

          <div className="relative">
            <AnimatePresence>
              {xpFlash && (
                <motion.span
                  initial={{ opacity: 0, y: 0, scale: 0.8 }}
                  animate={{ opacity: 1, y: -28, scale: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5 }}
                  className="absolute left-1/2 -translate-x-1/2 -top-2 text-success text-sm font-semibold pointer-events-none"
                >
                  +10 XP
                </motion.span>
              )}
            </AnimatePresence>
            {idx < total - 1 ? (
              <Button disabled={!selected} rightIcon={<ArrowRight className="w-4 h-4" />} onClick={next}>
                Next
              </Button>
            ) : (
              <Button disabled={answeredCount === 0} leftIcon={<Check className="w-4 h-4" />} onClick={submit}>
                Submit Answers
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
