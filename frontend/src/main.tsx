import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { readRoot } from "./bootstrap";

const { page, bootstrap } = readRoot();

// Lazy-load each screen so the initial bundle only carries what a page needs.
const SCREENS: Record<string, React.LazyExoticComponent<React.ComponentType<{ data: any }>>> = {
  dashboard: React.lazy(() => import("./screens/Dashboard")),
  // "knowledge" and "practice" both mount the Make Quiz & Test hub: knowledge is
  // its nav home; practice is the deep-link (/practice?doc=ID) used to make a
  // quiz from a saved Learning Journey resource.
  knowledge: React.lazy(() => import("./screens/MakeQuiz")),
  practice: React.lazy(() => import("./screens/MakeQuiz")),
  journey: React.lazy(() => import("./screens/Journey")),
  "add-resource": React.lazy(() => import("./screens/AddResource")),
  quiz: React.lazy(() => import("./screens/Quiz")),
  result: React.lazy(() => import("./screens/Result")),
  upload: React.lazy(() => import("./screens/Upload")),
  progress: React.lazy(() => import("./screens/Progress")),
  "weak-topics": React.lazy(() => import("./screens/WeakTopics")),
  achievements: React.lazy(() => import("./screens/Achievements")),
  settings: React.lazy(() => import("./screens/Settings")),
};

function Fallback() {
  return (
    <div className="min-h-screen grid place-items-center text-text-3">
      <div className="animate-pulse">Loading…</div>
    </div>
  );
}

const el = document.getElementById("root");
if (el) {
  const Screen = SCREENS[page];
  createRoot(el).render(
    <React.StrictMode>
      {Screen ? (
        <React.Suspense fallback={<Fallback />}>
          <Screen data={bootstrap} />
        </React.Suspense>
      ) : (
        <div className="p-8 text-text-2">Unknown page: {page}</div>
      )}
    </React.StrictMode>,
  );
}
