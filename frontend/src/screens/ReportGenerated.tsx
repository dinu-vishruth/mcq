import { useState } from "react";
import { motion } from "framer-motion";
import { Check, Copy, Play, ArrowLeft } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { csrfToken } from "@/bootstrap";

interface Option { label: string; text: string; }
interface Question { question: string; options: Option[]; answer_text: string; }
export interface ReportData {
  session_key: string;
  mcqs: Question[];
}

export default function ReportGenerated({ data }: { data: ReportData }) {
  const [copied, setCopied] = useState(false);

  function copyKey() {
    navigator.clipboard.writeText(data.session_key).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="app-glow min-h-screen">
      <div className="relative z-10 max-w-2xl mx-auto px-5 py-10">
        <Card pad="lg" className="text-center mb-6">
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 260, damping: 18 }}
            className="grid place-items-center w-16 h-16 rounded-full bg-success/10 text-success mx-auto mb-4"
          >
            <Check className="w-8 h-8" />
          </motion.span>
          <h1 className="text-2xl font-bold">Your practice set is ready</h1>
          <p className="text-text-2 mt-1">Start now, or save this key to resume later.</p>

          <div className="mt-6 p-4 rounded-md bg-inset border border-white/[0.08]">
            <div className="text-text-3 text-xs uppercase tracking-wider mb-2">Session Key</div>
            <div className="flex items-center justify-center gap-3">
              <code className="text-lg text-accent-2">{data.session_key}</code>
              <Button variant="secondary" size="sm" leftIcon={copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />} onClick={copyKey}>
                {copied ? "Copied!" : "Copy"}
              </Button>
            </div>
          </div>

          <div className="flex justify-center gap-3 mt-6">
            <form action="/student_login" method="post">
              <input type="hidden" name="csrf_token" value={csrfToken()} />
              <input type="hidden" name="session_key" value={data.session_key} />
              <Button type="submit" leftIcon={<Play className="w-4 h-4" />}>Start Quiz Now</Button>
            </form>
            <Button variant="secondary" leftIcon={<ArrowLeft className="w-4 h-4" />} onClick={() => (window.location.href = "/")}>
              Back to Home
            </Button>
          </div>
        </Card>

        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg">Question Preview</h2>
          <Badge tone="accent">{data.mcqs.length} questions</Badge>
        </div>
        <div className="space-y-3">
          {data.mcqs.map((q, i) => (
            <Card key={i}>
              <div className="flex items-start gap-2 mb-3">
                <span className="text-text-3 text-sm font-medium mt-0.5">Q{i + 1}</span>
                <p className="text-[1.02rem]">{q.question}</p>
              </div>
              <div className="space-y-1.5">
                {q.options.map((opt, j) => {
                  const correct = opt.text === q.answer_text;
                  return (
                    <div key={j} className={`flex items-center gap-2 px-3 py-2 rounded-sm text-sm ${
                      correct ? "bg-success/10 text-text border border-success/20" : "bg-white/[0.03] text-text-2"
                    }`}>
                      <span className={`grid place-items-center w-6 h-6 rounded-sm text-xs font-semibold ${correct ? "bg-success/20 text-success" : "bg-white/5 text-text-3"}`}>
                        {opt.label}
                      </span>
                      {opt.text}
                      {correct && <Check className="w-4 h-4 text-success ml-auto" />}
                    </div>
                  );
                })}
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
