// Reads the JSON payload and page key that the Jinja mount shell embeds on
// #root. Every server-rendered template that hosts a React island writes:
//   <div id="root" data-page="dashboard" data-bootstrap='{{ ...|tojson }}'></div>
// so the backend stays the source of truth for data and auth, and React just
// renders it. No client-side fetching is required for the initial paint.

export interface RootData {
  page: string;
  bootstrap: Record<string, unknown>;
}

export function readRoot(): RootData {
  const el = document.getElementById("root");
  const page = el?.dataset.page ?? "unknown";
  let bootstrap: Record<string, unknown> = {};
  const raw = el?.dataset.bootstrap;
  if (raw) {
    try {
      bootstrap = JSON.parse(raw);
    } catch (e) {
      console.error("Failed to parse bootstrap data", e);
    }
  }
  return { page, bootstrap };
}

// CSRF token is rendered into a meta tag by the shell so POSTs to existing
// Flask-WTF-protected endpoints keep working from React.
export function csrfToken(): string {
  return (
    document
      .querySelector('meta[name="csrf-token"]')
      ?.getAttribute("content") ?? ""
  );
}
