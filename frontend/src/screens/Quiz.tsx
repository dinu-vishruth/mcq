import { useState, useEffect, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Clock, ArrowRight, Check } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { apiSend, csrfToken } from "@/bootstrap";

interface Option { label: string; text: string; }
interface Question { question: string; options: Option[]; }
export interface QuizData {
  mcqs: Question[];
  timer: number;
  username: string;
  deadline: number;
  server_now: number;
  session_key: string;
  user_id: number;
  answers: Record<number, string>;
  confidence: Record<number, Confidence>;
  question_index: number;
}

type Confidence = "know" | "maybe" | "no_idea";

const CONFIDENCE: { id: Confidence; label: string; tone: string }[] = [
  { id: "know", label: "Confident", tone: "text-success border-success/30 bg-success/10" },
  { id: "maybe", label: "Unsure", tone: "text-warning border-warning/30 bg-warning/10" },
  { id: "no_idea", label: "Guessing", tone: "text-text-3 border-white/10 bg-white/5" },
];

const OPT_KEYS = ["A", "B", "C", "D"];

export default function Quiz({ data }: { data: QuizData }) {
  const total = data.mcqs.length;
  const storageKey = `quiz:${data.user_id}:${data.session_key}:${data.deadline}`;
  const [initial] = useState(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey) || "null");
      if (saved && saved.deadline === data.deadline && data.timer > 0) {
        const answers: Record<number, string> = {};
        const confidence: Record<number, Confidence> = {};
        data.mcqs.forEach((q, i) => {
          if (q.options.some((o) => o.text === saved.answers?.[i])) answers[i] = saved.answers[i];
          if (["know", "maybe", "no_idea"].includes(saved.confidence?.[i])) confidence[i] = saved.confidence[i];
        });
        return { answers, confidence, idx: Math.max(0, Math.min(total - 1, Math.floor(Number(saved.idx)) || 0)) };
      }
    } catch { /* Storage may be unavailable; the server draft remains usable. */ }
    return { answers: data.answers, confidence: data.confidence, idx: data.question_index };
  });
  const [idx, setIdx] = useState(initial.idx);
  const [answers, setAnswers] = useState<Record<number, string>>(initial.answers);
  const [confidence, setConfidence] = useState<Record<number, Confidence>>(initial.confidence);
  const [remaining, setRemaining] = useState(Math.ceil(data.timer));
  const [saveError, setSaveError] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);
  const submittingRef = useRef(false);
  const clockRef = useRef({ at: performance.now(), seconds: data.deadline - data.server_now });
  const saveQueue = useRef<Promise<unknown>>(Promise.resolve());

  const answeredCount = Object.keys(answers).length;
  const q = data.mcqs[idx];
  const selected = answers[idx];

  // Measure elapsed time rather than counting callbacks; background tabs can throttle timers.
  useEffect(() => {
    const tick = () => setRemaining(Math.max(0, Math.ceil(
      clockRef.current.seconds - (performance.now() - clockRef.current.at) / 1000,
    )));
    tick();
    const t = setInterval(tick, 250);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (remaining <= 0) submit();
  }, [remaining]);

  useEffect(() => {
    if (submittingRef.current || remaining <= 0) return;
    try {
      sessionStorage.setItem(storageKey, JSON.stringify({ answers, confidence, idx, deadline: data.deadline }));
    } catch { /* Server autosave also protects progress when storage is unavailable. */ }
    const timer = setTimeout(() => {
      // Serialize saves so a slower previous request cannot overwrite newer answers.
      saveQueue.current = saveQueue.current.catch(() => undefined).then(() => apiSend(
        "/api/quiz-attempt", "POST",
        { session_key: data.session_key, answers, confidence, question_index: idx },
      )).then(() => setSaveError(false)).catch(() => setSaveError(true));
    }, 200);
    return () => clearTimeout(timer);
  }, [answers, confidence, idx]);

  const mmss = useMemo(() => {
    const m = Math.floor(Math.max(0, remaining) / 60);
    const s = Math.max(0, remaining) % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }, [remaining]);

  function choose(text: string) {
    if (remaining <= 0 || submittingRef.current) return;
    setAnswers((a) => ({ ...a, [idx]: text }));
  }

  function next() {
    if (idx < total - 1) setIdx(idx + 1);
  }

  function submit() {
    if (!formRef.current || submittingRef.current) return;
    submittingRef.current = true;
    setSubmitting(true);
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
        {data.mcqs.map((_, i) => (
          <span key={i}>
            <input type="hidden" name={`q-${i}`} value={answers[i] ?? ""} readOnly />
            <input type="hidden" name={`confidence-${i}`} value={confidence[i] ?? ""} readOnly />
          </span>
        ))}
      </form>

      {/* Question */}
      <div className="relative z-10 flex-1 max-w-3xl w-full mx-auto px-5 py-10 flex flex-col">
        {saveError && <p role="alert" className="mb-4 text-warning">Progress could not sync. Keep this tab open and submit your answers when finished.</p>}
        <p className="mb-4 text-xs text-text-3">Use this set to find what to revisit. Confidence is optional and never changes your score.</p>
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
                    disabled={submitting || remaining <= 0}
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
              <div className="text-text-3 text-xs uppercase tracking-wider mb-2">How confident are you? (optional)</div>
              <div className="flex gap-2">
                {CONFIDENCE.map((c) => (
                  <button
                    key={c.id}
                    disabled={submitting || remaining <= 0}
                    aria-pressed={confidence[idx] === c.id}
                    onClick={() => setConfidence((cf) => {
                      const next = { ...cf };
                      if (next[idx] === c.id) delete next[idx];
                      else next[idx] = c.id;
                      return next;
                    })}
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
            {idx < total - 1 ? (
              <Button disabled={!selected} rightIcon={<ArrowRight className="w-4 h-4" />} onClick={next}>
                Next
              </Button>
            ) : (
              <Button disabled={submitting || answeredCount === 0} leftIcon={<Check className="w-4 h-4" />} onClick={submit}>
                {submitting ? "Submitting…" : "Submit Answers"}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
