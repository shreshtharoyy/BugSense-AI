"use client";

import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useState } from "react";

/* ================= types ================= */

type SearchResult = {
  bug_id: string;
  title: string;
  project: string;
  resolution: string | null;
  created_at: string | null;
  resolved_at: string | null;
  document: string;
  snippet: string;
  similarity: number;
  rrf_score: number | null;
  score: number;
};

type SearchResponse = {
  query: {
    bug_id?: string;
    title: string;
    query_text: string;
    snippet: string;
    known_duplicate_bug_id?: string | null;
  };
  results: SearchResult[];
  ground_truth_found: boolean | null;
  ground_truth_rank: number | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const EXAMPLES = [
  "Firefox crashes when I open Settings",
  "Bookmarks stop syncing after sign in",
  "PDF viewer shows a blank page",
];

/* ================= smart helpers ================= */

const STOPWORDS = new Set([
  "the","a","an","and","or","but","if","then","when","while","of","to","in","on",
  "for","with","at","by","from","is","are","was","were","be","been","it","this",
  "that","these","those","i","my","we","you","after","before","not","no","does",
  "do","did","can","cannot","cant","dont","doesnt","has","have","had","will",
  "would","should","could","there","here","its","into","about","also","only",
  "title","description","error","log","bug","firefox","browser","issue","user",
]);

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9_]{3,}/g) ?? []).filter(
    (t) => !STOPWORDS.has(t)
  );
}

function sharedTerms(query: string, doc: string, limit = 6): string[] {
  const queryTokens = new Set(tokenize(query));
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of tokenize(doc)) {
    if (queryTokens.has(token) && !seen.has(token)) {
      seen.add(token);
      out.push(token);
    }
  }
  return out.sort((a, b) => b.length - a.length).slice(0, limit);
}

type Tone = "emerald" | "indigo" | "neutral" | "amber";

function verdictFor(result: SearchResult): { label: string; tone: Tone } {
  const sim = result.similarity || 0;
  if (sim >= 0.8) return { label: "Likely duplicate", tone: "emerald" };
  if (sim >= 0.65) return { label: "Closely related", tone: "indigo" };
  if (sim > 0) return { label: "Possibly related", tone: "neutral" };
  return { label: "Keyword match", tone: "amber" };
}

const TONE = {
  emerald: "bg-emerald-50 text-emerald-700 border-emerald-200",
  indigo: "bg-indigo-50 text-indigo-700 border-indigo-200",
  neutral: "bg-neutral-100 text-neutral-600 border-neutral-200",
  amber: "bg-amber-50 text-amber-700 border-amber-200",
} as const;

/** The product's real answer: should you file this bug or not? */
function recommendation(results: SearchResult[]): {
  tone: Tone;
  title: string;
  body: string;
} {
  if (!results.length) {
    return {
      tone: "emerald",
      title: "Looks like a new bug",
      body: "No similar reports found in the index. Safe to file.",
    };
  }
  const top = results[0].similarity || 0;
  if (top >= 0.8) {
    return {
      tone: "amber",
      title: "Possible duplicate — check before filing",
      body: `The top result is a ${Math.round(top * 100)}% match. Review it first; this bug may already be reported.`,
    };
  }
  if (top >= 0.65) {
    return {
      tone: "indigo",
      title: "Related bugs exist",
      body: "Nothing identical, but the top matches are close — worth a look for context or workarounds.",
    };
  }
  return {
    tone: "emerald",
    title: "Probably a new bug",
    body: "Only weak matches found — likely safe to file a new report.",
  };
}

function parseDocument(doc: string): { title: string; errorLog: string; description: string } {
  let title = "";
  let errorLog = "";
  let description = doc;
  if (doc.startsWith("Title: ")) {
    const rest = doc.slice("Title: ".length);
    const errSep = ". Error Log: ";
    const descSep = ". Description: ";
    const errAt = rest.indexOf(errSep);
    const descAt = rest.indexOf(descSep);
    if (errAt !== -1 && (descAt === -1 || errAt < descAt)) {
      title = rest.slice(0, errAt);
      const afterErr = rest.slice(errAt + errSep.length);
      const dAt = afterErr.indexOf(descSep);
      if (dAt !== -1) {
        errorLog = afterErr.slice(0, dAt);
        description = afterErr.slice(dAt + descSep.length);
      } else {
        errorLog = afterErr;
        description = "";
      }
    } else if (descAt !== -1) {
      title = rest.slice(0, descAt);
      description = rest.slice(descAt + descSep.length);
    } else {
      title = rest;
      description = "";
    }
  }
  return { title: title.trim(), errorLog: errorLog.trim(), description: description.trim() };
}

function formatDate(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

/** "resolved in 12 days" — how long this bug took historically. */
function resolutionTime(created: string | null, resolved: string | null): string | null {
  if (!created || !resolved) return null;
  const a = new Date(created).getTime();
  const b = new Date(resolved).getTime();
  if (isNaN(a) || isNaN(b) || b <= a) return null;
  const days = Math.round((b - a) / 86_400_000);
  if (days < 1) return "resolved same day";
  if (days < 31) return `resolved in ${days} day${days > 1 ? "s" : ""}`;
  const months = Math.round(days / 30);
  if (months < 12) return `resolved in ${months} month${months > 1 ? "s" : ""}`;
  const years = Math.round(months / 12);
  return `resolved in ${years} year${years > 1 ? "s" : ""}`;
}

function Highlighted({
  text,
  terms,
  dark = false,
}: {
  text: string;
  terms: string[];
  dark?: boolean;
}) {
  const parts = useMemo(() => {
    if (!terms.length) return [text];
    const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const re = new RegExp(`(${escaped.join("|")})`, "gi");
    return text.split(re);
  }, [text, terms]);
  const markClass = dark
    ? "rounded bg-amber-400/25 px-0.5 text-amber-200"
    : "rounded bg-indigo-100/80 px-0.5 text-indigo-900";
  return (
    <>
      {parts.map((part, i) =>
        terms.some((t) => t.toLowerCase() === part.toLowerCase()) ? (
          <mark key={i} className={markClass}>{part}</mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}

/* ================= page ================= */

export default function HomePage() {
  const [mode, setMode] = useState<"text" | "bug-id">("text");
  const [bugId, setBugId] = useState("");
  const [queryText, setQueryText] = useState("");
  const [topK, setTopK] = useState(5);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [corpus, setCorpus] = useState<number | null>(null);
  const [selected, setSelected] = useState(0);
  const [mobileDetail, setMobileDetail] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.json())
      .then((d) => setCorpus(d.parent_bugs ?? null))
      .catch(() => {});
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setMobileDetail(false);
      const inField =
        e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement;
      if (e.key === "/" && !inField) {
        e.preventDefault();
        document.getElementById("bs-query")?.focus();
        return;
      }
      if (!data?.results.length || inField) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelected((s) => Math.min(s + 1, data.results.length - 1));
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelected((s) => Math.max(s - 1, 0));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [data]);

  async function runSearch(payload: Record<string, unknown>) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail ?? "Something went wrong.");
      setData(body);
      setSelected(0);
      setMobileDetail(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  function submit() {
    if (mode === "text") {
      if (queryText.trim()) runSearch({ query_text: queryText.trim(), top_k: topK });
    } else {
      if (bugId.trim()) runSearch({ bug_id: bugId.trim(), top_k: topK });
    }
  }

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    submit();
  }

  function onTextareaKey(e: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      submit();
    }
  }

  const queryForMatch = data?.query.query_text ?? "";
  const rec = data ? recommendation(data.results) : null;
  const corpusLabel = corpus ? corpus.toLocaleString() : "24,824";
  const hasResults = !!data && !loading;

  return (
    <div className="flex min-h-screen flex-col">
      {/* Top bar */}
      <header className="sticky top-0 z-20 border-b border-neutral-200/70 bg-neutral-50/85 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-5">
          <div className="flex items-center gap-2.5">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 text-white">
              <BugIcon className="h-4 w-4" />
            </span>
            <span className="text-[15px] font-semibold tracking-tight text-neutral-900">BugSense</span>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="#whats-next"
              className="hidden text-[13px] font-medium text-neutral-500 transition hover:text-neutral-900 sm:block"
            >
              What&apos;s next
            </a>
            <div className="flex items-center gap-2 rounded-full border border-neutral-200 bg-white px-3 py-1 text-xs text-neutral-500">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
              </span>
              {corpusLabel} bugs indexed
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-5 pb-16">
        {!hasResults && (
          <section className="pt-14 pb-8 text-center">
            <h1 className="text-[34px] font-semibold leading-[1.1] tracking-tight text-neutral-900 sm:text-[40px]">
              Find similar bugs,
              <br />
              before you file a new one.
            </h1>
            <p className="mx-auto mt-4 max-w-md text-[15px] leading-7 text-neutral-500">
              Describe a bug in plain language and BugSense surfaces the closest
              matches from thousands of past reports.
            </p>
          </section>
        )}

        {/* Search */}
        <section
          className={[
            "mx-auto max-w-3xl rounded-2xl border border-neutral-200 bg-white p-2",
            "shadow-[0_1px_2px_rgba(0,0,0,0.04),0_8px_24px_-12px_rgba(0,0,0,0.12)]",
            hasResults ? "mt-6" : "",
          ].join(" ")}
        >
          <div className="flex gap-1 rounded-xl bg-neutral-100 p-1">
            <Segment active={mode === "text"} onClick={() => setMode("text")}>
              <TextIcon className="h-3.5 w-3.5" /> Describe a bug
            </Segment>
            <Segment active={mode === "bug-id"} onClick={() => setMode("bug-id")}>
              <HashIcon className="h-3.5 w-3.5" /> Bug number
            </Segment>
          </div>

          <form onSubmit={onSubmit} className="p-2.5">
            {mode === "text" ? (
              <textarea
                id="bs-query"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onKeyDown={onTextareaKey}
                rows={hasResults ? 2 : 3}
                placeholder="e.g. Firefox freezes when playing a video in fullscreen…"
                className="w-full resize-none rounded-xl border-0 bg-transparent px-1.5 py-1 text-[15px] leading-7 text-neutral-800 outline-none placeholder:text-neutral-400"
              />
            ) : (
              <input
                id="bs-query"
                value={bugId}
                onChange={(e) => setBugId(e.target.value)}
                placeholder="e.g. 1702772"
                className="w-full rounded-xl border-0 bg-transparent px-1.5 py-2 text-[15px] text-neutral-800 outline-none placeholder:text-neutral-400"
              />
            )}

            <div className="mt-2 flex items-center justify-between gap-3 border-t border-neutral-100 pt-2.5">
              <div className="flex items-center gap-3 pl-1 text-xs text-neutral-400">
                <span className="flex items-center gap-1.5">
                  Show
                  <select
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="cursor-pointer rounded-md bg-neutral-100 px-1.5 py-1 font-medium text-neutral-600 outline-none"
                  >
                    {[3, 5, 10].map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </span>
                <span className="hidden items-center gap-1 sm:flex">
                  <Kbd>Ctrl</Kbd>
                  <Kbd>Enter</Kbd>
                  to search
                </span>
              </div>
              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-700 active:scale-[0.98] disabled:opacity-50"
              >
                {loading ? "Searching" : "Search"}
                {!loading && <ArrowIcon className="h-3.5 w-3.5" />}
              </button>
            </div>
          </form>
        </section>

        {mode === "text" && !data && !loading && (
          <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => setQueryText(ex)}
                className="rounded-full border border-neutral-200 bg-white px-3 py-1.5 text-xs text-neutral-500 transition hover:border-neutral-300 hover:text-neutral-800"
              >
                {ex}
              </button>
            ))}
          </div>
        )}

        {error && (
          <div className="mx-auto mt-6 max-w-3xl rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        )}

        {loading && (
          <div className="mx-auto mt-8 max-w-3xl">
            <ResultsSkeleton count={topK} />
          </div>
        )}

        {/* ============ results ============ */}
        {hasResults && data && rec && (
          <section className="mt-6">
            {/* recommendation — the actual answer */}
            <div
              className={[
                "fade-up mx-auto mb-6 flex max-w-3xl items-start gap-3 rounded-2xl border px-4 py-3.5",
                rec.tone === "amber"
                  ? "border-amber-200 bg-amber-50"
                  : rec.tone === "indigo"
                    ? "border-indigo-200 bg-indigo-50"
                    : "border-emerald-200 bg-emerald-50",
              ].join(" ")}
            >
              <span
                className={[
                  "mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full",
                  rec.tone === "amber"
                    ? "bg-amber-100 text-amber-700"
                    : rec.tone === "indigo"
                      ? "bg-indigo-100 text-indigo-700"
                      : "bg-emerald-100 text-emerald-700",
                ].join(" ")}
              >
                {rec.tone === "amber" ? (
                  <AlertIcon className="h-3.5 w-3.5" />
                ) : (
                  <CheckIcon className="h-3.5 w-3.5" />
                )}
              </span>
              <div>
                <div className="text-sm font-semibold text-neutral-900">{rec.title}</div>
                <p className="mt-0.5 text-[13px] leading-6 text-neutral-600">{rec.body}</p>
                {data.ground_truth_found && (
                  <p className="mt-1 text-xs font-medium text-emerald-700">
                    Verified: the labeled duplicate of this bug appears at #{data.ground_truth_rank}.
                  </p>
                )}
              </div>
            </div>

            {data.results.length > 0 && (
              <>
                <div className="mb-3 flex items-baseline justify-between px-1">
                  <h2 className="text-sm font-medium text-neutral-900">
                    {data.results.length} similar bug{data.results.length > 1 ? "s" : ""}
                  </h2>
                  <span className="hidden text-xs text-neutral-400 sm:inline">
                    <Kbd>↑</Kbd> <Kbd>↓</Kbd> to navigate
                  </span>
                </div>

                <div className="grid items-start gap-4 lg:grid-cols-[minmax(300px,380px)_minmax(0,1fr)]">
                  <ul className="divide-y divide-neutral-100 overflow-hidden rounded-2xl border border-neutral-200 bg-white lg:sticky lg:top-[72px] lg:max-h-[calc(100vh-100px)] lg:overflow-y-auto">
                    {data.results.map((r, i) => (
                      <ResultRow
                        key={`${r.bug_id}-${i}`}
                        rank={i + 1}
                        result={r}
                        active={selected === i}
                        isDuplicate={data.query.known_duplicate_bug_id === r.bug_id}
                        delayMs={i * 45}
                        onClick={() => {
                          setSelected(i);
                          setMobileDetail(true);
                        }}
                      />
                    ))}
                  </ul>

                  <div className="hidden lg:block">
                    <DetailPanel
                      key={data.results[selected]?.bug_id}
                      result={data.results[selected]}
                      rank={selected + 1}
                      query={queryForMatch}
                      isDuplicate={data.query.known_duplicate_bug_id === data.results[selected]?.bug_id}
                    />
                  </div>
                </div>

                {mobileDetail && data.results[selected] && (
                  <div className="fixed inset-0 z-30 flex flex-col bg-neutral-50 lg:hidden">
                    <div className="flex h-14 items-center gap-3 border-b border-neutral-200 bg-white px-4">
                      <button
                        onClick={() => setMobileDetail(false)}
                        className="flex items-center gap-1.5 text-sm font-medium text-neutral-600"
                      >
                        <BackIcon className="h-4 w-4" /> Results
                      </button>
                    </div>
                    <div className="flex-1 overflow-y-auto p-4">
                      <DetailPanel
                        key={data.results[selected].bug_id}
                        result={data.results[selected]}
                        rank={selected + 1}
                        query={queryForMatch}
                        isDuplicate={data.query.known_duplicate_bug_id === data.results[selected]?.bug_id}
                      />
                    </div>
                  </div>
                )}
              </>
            )}
          </section>
        )}

        {!data && !loading && !error && (
          <div className="flex flex-col items-center py-14 text-center">
            <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-xl border border-neutral-200 bg-white text-neutral-300">
              <SearchIcon className="h-5 w-5" />
            </div>
            <p className="text-sm text-neutral-400">
              Your results will appear here. Press <Kbd>/</Kbd> to start typing.
            </p>
          </div>
        )}

        {!hasResults && <HowItWorks />}

        <ComingSoon />
      </main>

      <footer className="border-t border-neutral-200/70 py-6">
        <p className="text-center text-xs text-neutral-400">
          BugSense — hybrid semantic + keyword search over {corpusLabel} Firefox bug reports
        </p>
      </footer>
    </div>
  );
}

/* ================= list row ================= */

function ResultRow({
  rank,
  result,
  active,
  isDuplicate,
  delayMs,
  onClick,
}: {
  rank: number;
  result: SearchResult;
  active: boolean;
  isDuplicate: boolean;
  delayMs: number;
  onClick: () => void;
}) {
  const pct = Math.round((result.similarity || 0) * 100);
  const verdict = verdictFor(result);

  return (
    <li className="row-in" style={{ animationDelay: `${delayMs}ms` }}>
      <button
        onClick={onClick}
        className={[
          "relative block w-full px-4 py-3.5 text-left transition",
          active ? "bg-indigo-50/60" : "hover:bg-neutral-50",
        ].join(" ")}
      >
        {active && <span className="absolute inset-y-0 left-0 w-0.5 bg-indigo-500" />}
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <span
              className={[
                "text-xs tabular-nums",
                active ? "font-semibold text-indigo-600" : "text-neutral-300",
              ].join(" ")}
            >
              {String(rank).padStart(2, "0")}
            </span>
            <h3
              className={[
                "truncate text-[14px] leading-6",
                active ? "font-semibold text-neutral-900" : "font-medium text-neutral-800",
              ].join(" ")}
            >
              {result.title || "Untitled bug"}
            </h3>
          </div>
          <span className="shrink-0 text-xs font-semibold tabular-nums text-neutral-500">
            {pct > 0 ? `${pct}%` : "kw"}
          </span>
        </div>
        <div className="mt-1.5 flex items-center gap-2 pl-[26px]">
          <span className={`rounded-full border px-2 py-px text-[11px] font-medium ${TONE[verdict.tone]}`}>
            {verdict.label}
          </span>
          {isDuplicate && (
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-600">
              <CheckIcon className="h-3 w-3" /> confirmed
            </span>
          )}
          <span className="truncate text-[11px] text-neutral-400">#{result.bug_id}</span>
        </div>
      </button>
    </li>
  );
}

/* ================= detail panel ================= */

function DetailPanel({
  result,
  rank,
  query,
  isDuplicate,
}: {
  result: SearchResult | undefined;
  rank: number;
  query: string;
  isDuplicate: boolean;
}) {
  const [copied, setCopied] = useState(false);
  if (!result) return null;

  const pct = Math.round((result.similarity || 0) * 100);
  const verdict = verdictFor(result);
  const parsed = parseDocument(result.document);
  const terms = sharedTerms(query, result.document);
  const created = formatDate(result.created_at);
  const resTime = resolutionTime(result.created_at, result.resolved_at);

  function copyLog() {
    navigator.clipboard.writeText(parsed.errorLog).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <article className="fade-up overflow-hidden rounded-2xl border border-neutral-200 bg-white lg:sticky lg:top-[72px]">
      <div className="border-b border-neutral-100 p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${TONE[verdict.tone]}`}>
                {verdict.label}
              </span>
              {isDuplicate && (
                <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700">
                  <CheckIcon className="h-3 w-3" /> confirmed duplicate
                </span>
              )}
              <span className="text-xs text-neutral-400">result #{rank}</span>
            </div>

            <h2 className="mt-2.5 text-lg font-semibold leading-snug text-neutral-900">
              {parsed.title || result.title || "Untitled bug"}
            </h2>

            <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-neutral-500">
              <a
                href={`https://bugzilla.mozilla.org/show_bug.cgi?id=${result.bug_id}`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 font-medium text-indigo-600 hover:text-indigo-800"
              >
                Bug #{result.bug_id} on Bugzilla <ExternalIcon className="h-3 w-3" />
              </a>
              {result.resolution && (
                <span className="rounded bg-neutral-100 px-1.5 py-0.5 font-medium text-neutral-600">
                  {result.resolution}
                </span>
              )}
              {created && <span>filed {created}</span>}
              {resTime && (
                <span className="inline-flex items-center gap-1 text-neutral-500">
                  <ClockIcon className="h-3 w-3" /> {resTime}
                </span>
              )}
            </div>
          </div>

          <MatchRing pct={pct} />
        </div>

        {terms.length > 0 && (
          <div className="mt-4">
            <div className="mb-1.5 text-xs font-medium text-neutral-600">
              Why this matched — shared terms
            </div>
            <div className="flex flex-wrap gap-1.5">
              {terms.map((t) => (
                <span key={t} className="rounded-md bg-indigo-50 px-2 py-0.5 font-mono text-xs text-indigo-700">
                  {t}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="max-h-[52vh] overflow-y-auto p-5">
        {parsed.errorLog && (
          <section className="mb-5">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
                Error log
              </h3>
              <button
                onClick={copyLog}
                className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium text-neutral-400 transition hover:bg-neutral-100 hover:text-neutral-700"
              >
                {copied ? (
                  <>
                    <CheckIcon className="h-3 w-3 text-emerald-500" /> Copied
                  </>
                ) : (
                  <>
                    <CopyIcon className="h-3 w-3" /> Copy
                  </>
                )}
              </button>
            </div>
            <pre className="rounded-xl bg-neutral-900 p-3.5 font-mono text-xs leading-6 text-neutral-100">
              <Highlighted text={parsed.errorLog} terms={terms} dark />
            </pre>
          </section>
        )}

        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">
            Description
          </h3>
          <p className="whitespace-pre-wrap text-[13.5px] leading-7 text-neutral-700">
            <Highlighted text={parsed.description || result.snippet} terms={terms} />
          </p>
        </section>
      </div>
    </article>
  );
}

/** Small circular match indicator — reads at a glance. */
function MatchRing({ pct }: { pct: number }) {
  const color = pct >= 80 ? "stroke-emerald-500" : pct >= 65 ? "stroke-indigo-500" : "stroke-neutral-400";
  const textColor = pct >= 80 ? "text-emerald-600" : pct >= 65 ? "text-indigo-600" : "text-neutral-500";
  return (
    <div className="relative h-12 w-12 shrink-0">
      <svg viewBox="0 0 36 36" className="h-12 w-12 -rotate-90">
        <circle cx="18" cy="18" r="15.5" fill="none" className="stroke-neutral-100" strokeWidth="3.5" />
        {pct > 0 && (
          <circle
            cx="18" cy="18" r="15.5" fill="none"
            className={color}
            strokeWidth="3.5"
            strokeLinecap="round"
            pathLength={100}
            strokeDasharray={`${pct} 100`}
          />
        )}
      </svg>
      <span className={`absolute inset-0 flex items-center justify-center text-[11px] font-bold tabular-nums ${textColor}`}>
        {pct > 0 ? `${pct}%` : "kw"}
      </span>
    </div>
  );
}

/* ================= shared bits ================= */

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border border-neutral-200 bg-white px-1.5 py-0.5 font-sans text-[10px] font-medium text-neutral-500 shadow-[0_1px_0_rgba(0,0,0,0.06)]">
      {children}
    </kbd>
  );
}

function ResultsSkeleton({ count }: { count: number }) {
  return (
    <ul className="divide-y divide-neutral-100 overflow-hidden rounded-2xl border border-neutral-200 bg-white">
      {Array.from({ length: Math.min(count, 5) }).map((_, i) => (
        <li key={i} className="flex gap-4 px-4 py-4">
          <div className="skeleton h-3 w-5 rounded" />
          <div className="flex-1 space-y-2.5">
            <div className="skeleton h-3.5 w-2/3 rounded" />
            <div className="skeleton h-3 w-1/3 rounded" />
          </div>
        </li>
      ))}
    </ul>
  );
}

function Segment({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "flex flex-1 items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] font-medium transition",
        active ? "bg-white text-neutral-900 shadow-sm" : "text-neutral-500 hover:text-neutral-800",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

function HowItWorks() {
  const steps = [
    { n: "1", t: "Describe the bug", d: "Plain words or an error log — no format needed." },
    { n: "2", t: "AI reads the meaning", d: "Text becomes a vector, so wording differences do not matter." },
    { n: "3", t: "Closest bugs surface", d: "Similar meaning and shared error tokens rank highest." },
  ];
  return (
    <section className="mx-auto mt-16 max-w-3xl border-t border-neutral-200 pt-10">
      <div className="grid gap-8 sm:grid-cols-3">
        {steps.map((s) => (
          <div key={s.n}>
            <div className="mb-2.5 flex h-6 w-6 items-center justify-center rounded-full border border-neutral-200 text-xs font-semibold text-neutral-400">
              {s.n}
            </div>
            <div className="text-sm font-medium text-neutral-900">{s.t}</div>
            <p className="mt-1 text-[13px] leading-6 text-neutral-500">{s.d}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ================= coming soon ================= */

function ComingSoon() {
  const features = [
    {
      icon: <GaugeIcon className="h-4 w-4" />,
      title: "Severity prediction",
      desc: "AI suggests a P1–P5 priority straight from the description.",
      status: "In progress",
      live: true,
    },
    {
      icon: <ImageIcon className="h-4 w-4" />,
      title: "Screenshot search",
      desc: "Attach a screenshot and find bugs that look the same.",
      status: "Planned",
      live: false,
    },
    {
      icon: <BranchIcon className="h-4 w-4" />,
      title: "Root-cause hints",
      desc: "Likely component and cause, learned from similar fixed bugs.",
      status: "In progress",
      live: true,
    },
    {
      icon: <FileIcon className="h-4 w-4" />,
      title: "AI ticket draft",
      desc: "One click turns your description into a well-formatted report.",
      status: "Planned",
      live: false,
    },
  ];

  return (
    <section id="whats-next" className="mx-auto mt-16 max-w-3xl scroll-mt-20 border-t border-neutral-200 pt-10 pb-4">
      <div className="mb-6 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold text-neutral-900">What&apos;s next</h2>
        <span className="text-xs text-neutral-400">the roadmap, honestly</span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {features.map((f) => (
          <div
            key={f.title}
            className="group rounded-2xl border border-neutral-200 bg-white p-4 transition hover:border-neutral-300"
          >
            <div className="flex items-start justify-between gap-3">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-neutral-100 text-neutral-500 transition group-hover:bg-indigo-50 group-hover:text-indigo-600">
                {f.icon}
              </span>
              <span
                className={[
                  "rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                  f.live
                    ? "border-indigo-200 bg-indigo-50 text-indigo-600"
                    : "border-neutral-200 bg-neutral-50 text-neutral-400",
                ].join(" ")}
              >
                {f.status}
              </span>
            </div>
            <div className="mt-3 text-sm font-medium text-neutral-900">{f.title}</div>
            <p className="mt-1 text-[13px] leading-6 text-neutral-500">{f.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ================= icons ================= */

function BugIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 2l1.5 1.5M16 2l-1.5 1.5" />
      <path d="M9 7a3 3 0 0 1 6 0" />
      <rect x="7" y="7" width="10" height="12" rx="5" />
      <path d="M12 11v6M3 10h4M17 10h4M3 16h4M17 16h4M4 6l3 2M20 6l-3 2" />
    </svg>
  );
}
function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="7" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}
function TextIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 6h16M4 12h16M4 18h10" />
    </svg>
  );
}
function HashIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 9h16M4 15h16M10 3 8 21M16 3l-2 18" />
    </svg>
  );
}
function ArrowIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}
function BackIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 12H5M11 18l-6-6 6-6" />
    </svg>
  );
}
function CheckIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}
function AlertIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 9v4M12 17h.01" />
      <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
    </svg>
  );
}
function ClockIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </svg>
  );
}
function CopyIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="12" height="12" rx="2" />
      <path d="M5 15V5a2 2 0 0 1 2-2h10" />
    </svg>
  );
}
function ExternalIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 3h6v6M21 3l-9 9M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    </svg>
  );
}
function GaugeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 14 15.5 9" />
      <path d="M3.34 18a10 10 0 1 1 17.32 0" />
    </svg>
  );
}
function ImageIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-5-5L5 21" />
    </svg>
  );
}
function BranchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="6" cy="6" r="2.5" />
      <circle cx="6" cy="18" r="2.5" />
      <circle cx="18" cy="9" r="2.5" />
      <path d="M6 8.5V15.5M6 8.5c0 4 3 4.5 8.5 4.5M14.5 13c2 0 3.5-1.5 3.5-2.5" />
    </svg>
  );
}
function FileIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6M9 13h6M9 17h6" />
    </svg>
  );
}
