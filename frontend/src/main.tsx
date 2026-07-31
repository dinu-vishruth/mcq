import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { readRoot } from "./bootstrap";

const { page, bootstrap } = readRoot();

// Lazy-load each screen so the initial bundle only carries what a page needs.
const SCREENS: Record<string, React.LazyExoticComponent<React.ComponentType<{ data: any }>>> = {
  dashboard: React.lazy(() => import("./screens/Dashboard")),
  knowledge: React.lazy(() => import("./screens/Knowledge")),
  journey: React.lazy(() => import("./screens/Journey")),
  practice: React.lazy(() => import("./screens/Practice")),
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
