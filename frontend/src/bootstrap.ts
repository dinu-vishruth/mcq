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

// Shared JSON API helpers. Every screen used to hand-roll fetch + CSRF; these
// centralize it. Mutations automatically attach the CSRF header expected by the
// Flask-WTF-protected /api endpoints.
export async function apiGet<T = any>(path: string): Promise<T> {
  const res = await fetchWithTimeout(path, { headers: { Accept: "application/json" } }, 15000);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

export async function apiSend<T = any>(
  path: string,
  method: "POST" | "DELETE" | "PUT",
  body?: unknown,
  timeoutMs = 65000,
): Promise<T> {
  const res = await fetchWithTimeout(path, {
    method,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      "X-CSRFToken": csrfToken(),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  }, timeoutMs);
  if (!res.ok) {
    let msg = `${method} ${path} failed: ${res.status}`;
    try {
      const data = await res.json();
      if (data?.error) msg = data.error;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  return res.json();
}

async function fetchWithTimeout(path: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(path, { ...init, signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted) throw new Error("The request took too long. Your material is kept; please retry shortly.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
