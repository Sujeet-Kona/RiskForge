import { useEffect, useMemo, useState, type FormEvent } from "react";
import {
  Activity, AlertCircle, ArrowDownRight, ArrowUpRight, CheckCircle2,
  ChevronDown, CircleHelp, Clock3, Command, FileText, Gauge, RefreshCw,
  Shield, ShieldAlert, ShieldCheck, SlidersHorizontal, Sparkles, Wallet,
} from "lucide-react";
import {
  getDecisionHistory,
  getHealth,
  getReviewQueue,
  resolveReviewCase,
  scoreTransaction,
} from "./api";
import type {
  AuditDecisionSummary,
  Decision,
  HealthResponse,
  HistoryTransaction,
  ReviewCase,
  ReviewOutcome,
  ScoreResponse,
} from "./types";

const decisionMeta: Record<Decision, { label: string; className: string; icon: typeof ShieldCheck }> = {
  ALLOW: { label: "Allow", className: "allow", icon: ShieldCheck },
  REVIEW: { label: "Review", className: "review", icon: ShieldAlert },
  BLOCK: { label: "Block", className: "block", icon: Shield },
};

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState("");
  const [historyRows, setHistoryRows] = useState<AuditDecisionSummary[]>([]);
  const [historyError, setHistoryError] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [reviewerKey, setReviewerKey] = useState("");
  const [reviewCases, setReviewCases] = useState<ReviewCase[]>([]);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewLoaded, setReviewLoaded] = useState(false);
  const [reviewError, setReviewError] = useState("");
  const [reviewNotice, setReviewNotice] = useState("");
  const [reviewActionId, setReviewActionId] = useState<number | null>(null);
  const [reviewComments, setReviewComments] = useState<Record<number, string>>({});
  const [loading, setLoading] = useState(false);
  const [idempotencyKey, setIdempotencyKey] = useState(() => crypto.randomUUID());
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState("");
  const [historyText, setHistoryText] = useState("[]");
  const [form, setForm] = useState({
    transactionId: `RF-${Date.now().toString().slice(-8)}`,
    transactionDT: "10000000",
    amount: "245.50",
    card1: "13926",
    addr1: "",
    addr2: "",
    productCD: "W",
    card6: "debit",
    email: "",
    recipientEmail: "",
    topK: "3",
  });

  async function refreshHealth() {
    try {
      const next = await getHealth();
      setHealth(next);
      setHealthError("");
    } catch (err) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : "API unavailable");
    }
  }

  async function refreshHistory() {
    setHistoryLoading(true);
    try {
      const page = await getDecisionHistory();
      setHistoryRows(page.items);
      setHistoryError("");
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "Could not load decision history");
    } finally {
      setHistoryLoading(false);
    }
  }

  async function loadReviewQueue() {
    if (!reviewerKey.trim()) {
      setReviewError("Enter the reviewer API key configured on the FastAPI server.");
      return;
    }
    setReviewLoading(true);
    setReviewLoaded(false);
    setReviewError("");
    setReviewNotice("");
    setReviewCases([]);
    try {
      const page = await getReviewQueue(reviewerKey.trim());
      setReviewCases(page.items);
      setReviewLoaded(true);
    } catch (err) {
      setReviewError(err instanceof Error ? err.message : "Could not load review queue");
    } finally {
      setReviewLoading(false);
    }
  }

  async function resolveCase(caseId: number, outcome: ReviewOutcome) {
    if (!reviewerKey.trim()) return;
    setReviewActionId(caseId);
    setReviewError("");
    setReviewNotice("");
    try {
      const response = await resolveReviewCase(
        caseId,
        reviewerKey.trim(),
        outcome,
        reviewComments[caseId] ?? "",
      );
      const successMessage = `${response.case.transaction_id} recorded as ${response.case.status}. The original model decision remains REVIEW.`;
      setReviewComments((current) => ({ ...current, [caseId]: "" }));
      await loadReviewQueue();
      setReviewNotice(successMessage);
      await refreshHistory();
    } catch (err) {
      setReviewError(err instanceof Error ? err.message : "Could not record review outcome");
    } finally {
      setReviewActionId(null);
    }
  }

  useEffect(() => {
    void refreshHealth();
    void refreshHistory();
  }, []);

  const apiReady = health?.status === "ok" && health.database === "ready";
  const systemStatusLabel = health?.status !== "ok"
    ? "API unavailable"
    : health.database !== "ready"
      ? "Audit store unavailable"
      : "Scoring ready";
  const probability = result ? result.risk_probability * 100 : null;
  const decision = result ? decisionMeta[result.decision] : null;
  const ResultIcon = decision?.icon;
  const artifactFeatures = health?.artifact?.n_features;
  const featureContractLabel = typeof artifactFeatures === "number"
    ? String(artifactFeatures)
    : "—";
  const resultAnnouncement = loading
    ? "Scoring transaction."
    : result
      ? `Decision ${decision?.label}. Calibrated fraud probability ${probability?.toFixed(2)} percent. ${result.decision_explanation}`
      : "";

  const historyCount = useMemo(() => {
    try {
      const parsed = JSON.parse(historyText.trim() || "[]") as unknown;
      return Array.isArray(parsed) ? parsed.length : 0;
    } catch {
      return 0;
    }
  }, [historyText]);

  function update<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((current) => ({ ...current, [key]: value }));
    setIdempotencyKey(crypto.randomUUID());
  }

  function changeReviewerKey(value: string) {
    setReviewerKey(value);
    setReviewCases([]);
    setReviewComments({});
    setReviewLoaded(false);
    setReviewError("");
    setReviewNotice("");
  }

  async function submitScore(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);

    let history: HistoryTransaction[];
    try {
      const parsed: unknown = JSON.parse(historyText.trim() || "[]");
      if (!Array.isArray(parsed)) throw new Error("History must be a JSON array.");
      history = parsed as HistoryTransaction[];
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid history JSON.");
      return;
    }

    const transactionDT = Number(form.transactionDT);
    const amount = Number(form.amount);
    const card1 = Number(form.card1);
    if (![transactionDT, amount, card1].every(Number.isFinite)) {
      setError("Timestamp, amount and card identifier must be valid numbers.");
      return;
    }

    setLoading(true);
    try {
      const response = await scoreTransaction({
        transaction: {
          TransactionID: form.transactionId.trim(),
          TransactionDT: transactionDT,
          TransactionAmt: amount,
          card1,
          addr1: form.addr1 === "" ? null : Number(form.addr1),
          addr2: form.addr2 === "" ? null : Number(form.addr2),
          ProductCD: form.productCD || null,
          card6: form.card6 || null,
          P_emaildomain: form.email || null,
          R_emaildomain: form.recipientEmail || null,
        },
        history,
        top_k: Number(form.topK),
      }, idempotencyKey);
      setResult(response);
      // Keep this key for retries of the exact same form payload. Editing any
      // field rotates it in update(), so a changed request gets a fresh key.
      void refreshHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Scoring request failed.");
    } finally {
      setLoading(false);
      void refreshHealth();
    }
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#overview">Skip to main content</a>
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="RiskForge home">
          <span className="brand-mark"><Shield size={19} strokeWidth={2.4} /></span>
          <span><strong>riskforge</strong><small>RISK INTELLIGENCE</small></span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Main navigation">
          <a className="nav-item active" href="#overview"><Gauge size={17} /> Overview</a>
          <a className="nav-item" href="#score"><Activity size={17} /> Transaction scoring</a>
          <a className="nav-item" href="#activity"><FileText size={17} /> Decision history</a>
          <a className="nav-item" href="#review"><ShieldAlert size={17} /> Review queue</a>
        </nav>
        <div className="sidebar-bottom">
          <div className="environment-card">
            <span className="environment-dot" />
            <div><strong>Local environment</strong><small>Development workspace</small></div>
          </div>
          <div className="sidebar-foot"><span>RISKFORGE</span><span>v0.1.0</span></div>
        </div>
      </aside>

      <main className="main-content" id="overview" tabIndex={-1}>
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="slash">/</span><strong>Overview</strong></div>
          <div className="topbar-actions">
            <div className={`system-status ${apiReady ? "online" : "offline"}`}>
              <span className="status-dot" /> {systemStatusLabel}
            </div>
            <button className="icon-button" type="button" onClick={() => void refreshHealth()} aria-label="Refresh API status"><RefreshCw size={16} /></button>
            <div className="avatar">RF</div>
          </div>
        </header>

        <section className="page-heading">
          <div>
            <div className="eyebrow"><Sparkles size={13} /> FRAUD RISK OPERATIONS</div>
            <h1>Risk overview</h1>
            <p>Score transactions, inspect model signals and understand policy decisions.</p>
          </div>
          <div className="date-chip"><Clock3 size={15} /> Local workspace</div>
        </section>

        <section className="metric-grid" aria-label="System status">
          <article className="metric-card">
            <div className="metric-top"><span>MODEL STATUS</span><span className="metric-icon mint"><Activity size={17} /></span></div>
            <div className="metric-value">{health?.status === "ok" ? "Ready" : "Offline"}</div>
            <div className="metric-caption">{health?.model_version ?? "Waiting for API connection"}</div>
          </article>
          <article className="metric-card">
            <div className="metric-top"><span>FEATURE CONTRACT</span><span className="metric-icon blue"><SlidersHorizontal size={17} /></span></div>
            <div className="metric-value">{featureContractLabel}<small> features</small></div>
            <div className="metric-caption">{health?.feature_version ?? "Version unavailable"}</div>
          </article>
          <article className="metric-card">
            <div className="metric-top"><span>CALIBRATION</span><span className="metric-icon mint"><SlidersHorizontal size={17} /></span></div>
            <div className="metric-value policy-value">{health?.calibration_method ?? "—"}</div>
            <div className="metric-caption">Probability calibration method</div>
          </article>
          <article className="metric-card">
            <div className="metric-top"><span>DECISION POLICY</span><span className="metric-icon violet"><ShieldCheck size={17} /></span></div>
            <div className="metric-value policy-value">{health?.policy_version ?? "—"}</div>
            <div className="metric-caption">Calibrated probability thresholds</div>
          </article>
          <article className="metric-card">
            <div className="metric-top"><span>AUDIT DATABASE</span><span className="metric-icon amber"><Wallet size={17} /></span></div>
            <div className="metric-value">{health?.database === "ready" ? "Connected" : "Unavailable"}</div>
            <div className="metric-caption">PostgreSQL decision persistence</div>
          </article>
        </section>

        {healthError && <div className="notice warning" role="alert"><AlertCircle size={17} aria-hidden="true" /><span>{healthError} — start the API and confirm the serving artifact and database are available.</span></div>}

        <div className="content-grid">
          <section className="panel scoring-panel" id="score">
            <div className="panel-heading">
              <div><div className="section-kicker">INFERENCE</div><h2 id="scoring-heading">Score a transaction</h2><p>Submit a transaction and strictly-prior history to the frozen model.</p></div>
              <span className="panel-symbol"><Command size={18} aria-hidden="true" /></span>
            </div>
            <form onSubmit={submitScore} aria-labelledby="scoring-heading">
              <div className="form-section-title">TRANSACTION DETAILS</div>
              <div className="field-grid">
                <label className="field full"><span>Transaction ID</span><input required maxLength={64} value={form.transactionId} onChange={(e) => update("transactionId", e.target.value)} /></label>
                <label className="field"><span>Transaction amount</span><div className="input-prefix"><b>$</b><input required min="0" step="0.01" type="number" value={form.amount} onChange={(e) => update("amount", e.target.value)} /></div></label>
                <label className="field"><span>TransactionDT <em>elapsed seconds</em></span><input required min="0" step="1" type="number" value={form.transactionDT} onChange={(e) => update("transactionDT", e.target.value)} /></label>
                <label className="field"><span>Card identifier (card1)</span><input required type="number" value={form.card1} onChange={(e) => update("card1", e.target.value)} /></label>
                <label className="field"><span>Product category</span><select value={form.productCD} onChange={(e) => update("productCD", e.target.value)}><option value="">Not provided</option><option value="W">W</option><option value="C">C</option><option value="R">R</option><option value="H">H</option><option value="S">S</option></select><ChevronDown className="select-chevron" size={14} /></label>
                <label className="field"><span>Card type</span><select value={form.card6} onChange={(e) => update("card6", e.target.value)}><option value="">Not provided</option><option value="debit">Debit</option><option value="credit">Credit</option></select><ChevronDown className="select-chevron" size={14} /></label>
                <label className="field"><span>Address 1 <em>optional</em></span><input type="number" value={form.addr1} onChange={(e) => update("addr1", e.target.value)} /></label>
                <label className="field"><span>Address 2 <em>optional</em></span><input type="number" value={form.addr2} onChange={(e) => update("addr2", e.target.value)} /></label>
                <label className="field"><span>Payer email domain <em>optional</em></span><input placeholder="example.com" value={form.email} onChange={(e) => update("email", e.target.value)} /></label>
                <label className="field"><span>Recipient email domain <em>optional</em></span><input placeholder="example.com" value={form.recipientEmail} onChange={(e) => update("recipientEmail", e.target.value)} /></label>
              </div>

              <div className="form-section-title history-title">BEHAVIORAL CONTEXT <span>{historyCount} records</span></div>
              <label className="field full"><span>Prior transaction history <em>JSON array · strictly earlier timestamps only</em></span><textarea rows={4} spellCheck={false} value={historyText} onChange={(e) => { setHistoryText(e.target.value); setIdempotencyKey(crypto.randomUUID()); }} /></label>
              <div className="form-hint"><CircleHelp size={14} aria-hidden="true" /> History rows require TransactionDT and TransactionAmt. Include card1 / addr1 / addr2 to match the current entity.</div>

              {error && <div className="notice error" role="alert"><AlertCircle size={17} aria-hidden="true" /><span>{error}</span></div>}
              <button className="primary-button" type="submit" disabled={loading || !apiReady}>
                {loading ? <><span className="spinner" aria-hidden="true" /> Scoring transaction…</> : <><Activity size={16} aria-hidden="true" /> Run risk assessment</>}
              </button>
              {!apiReady && <p className="disabled-note">Scoring is disabled until the API, model artifact and audit database are ready.</p>}
            </form>
          </section>

          <section className="panel result-panel" aria-labelledby="result-heading">
            <div className="panel-heading">
              <div><div className="section-kicker">ASSESSMENT OUTPUT</div><h2 id="result-heading">Decision result</h2><p>Model risk signals and policy outcome.</p></div>
              <span className="panel-symbol"><ShieldCheck size={18} aria-hidden="true" /></span>
            </div>
            {!result ? (
              <div className="empty-result">
                <div className="empty-art"><Shield size={27} /></div>
                <h3>Awaiting assessment</h3>
                <p>Submit a transaction to see its calibrated fraud probability, decision and local explanation.</p>
                <div className="empty-checks"><span><CheckCircle2 size={14} /> Calibrated probability</span><span><CheckCircle2 size={14} /> Policy decision</span><span><CheckCircle2 size={14} /> TreeSHAP signals</span></div>
              </div>
            ) : (
              <div className="result-content">
                <div className={`decision-banner ${decision?.className ?? ""}`}>
                  <div className="decision-icon">{ResultIcon && <ResultIcon size={22} />}</div>
                  <div><span>POLICY OUTCOME</span><strong>{decision?.label}</strong><small>{result.decision_explanation}</small></div>
                </div>
                <div className="probability-block">
                  <div className="probability-head"><span>CALIBRATED FRAUD PROBABILITY</span><strong>{probability?.toFixed(2)}%</strong></div>
                  <div className="probability-track"><span className={decision?.className} style={{ width: `${probability ?? 0}%` }} /></div>
                  <div className="thresholds"><span>Review { (result.policy_detail.review_threshold * 100).toFixed(1) }%</span><span>Block { (result.policy_detail.block_threshold * 100).toFixed(1) }%</span></div>
                </div>
                <div className="result-meta"><div><span>MODEL</span><strong>{result.model_version}</strong></div><div><span>CALIBRATION</span><strong>{result.calibration_method}</strong></div><div><span>POLICY</span><strong>{result.policy_version}</strong></div><div><span>HISTORY USED</span><strong>{result.history_transactions_used}</strong></div></div>
                <div className="signals-heading"><h3>Top model signals</h3><span>TreeSHAP · risk increasing</span></div>
                {result.reason_codes.length ? result.reason_codes.map((item) => <div className="signal-row" key={`${item.direction}:${item.feature}`}><div><strong>{item.title}</strong><small>{item.feature}</small></div><span className="signal-value up"><ArrowUpRight size={13} /> {item.contribution.toFixed(3)}</span></div>) : <p className="muted-copy">No risk-increasing signals returned.</p>}
                {result.risk_decreasing_signals.length > 0 && <><div className="signals-heading secondary-signals"><h3>Risk-reducing signals</h3></div>{result.risk_decreasing_signals.map((item) => <div className="signal-row" key={`${item.direction}:${item.feature}`}><div><strong>{item.title}</strong><small>{item.feature}</small></div><span className="signal-value down"><ArrowDownRight size={13} /> {item.contribution.toFixed(3)}</span></div>)}</>}
                <div className="policy-explanation"><span>POLICY EXPLANATION</span><p>{result.decision_reason_code.description}</p></div>
                <div className="result-foot">Transaction {result.transaction_id} · {result.feature_count} features · {result.feature_version}</div>
              </div>
            )}
            <p className="visually-hidden" role="status">{resultAnnouncement}</p>
          </section>
        </div>

        <section className="panel history-panel" id="activity" aria-labelledby="history-heading">
          <div className="panel-heading history-heading">
            <div><div className="section-kicker">AUDIT TRAIL</div><h2 id="history-heading">Recent decisions</h2><p>Persisted scoring outcomes returned by the PostgreSQL audit store.</p></div>
            <button className="icon-button" type="button" onClick={() => void refreshHistory()} aria-label="Refresh decision history" aria-busy={historyLoading} disabled={historyLoading}><RefreshCw size={15} aria-hidden="true" /></button>
          </div>
          {historyError && <div className="notice warning" role="alert"><AlertCircle size={16} aria-hidden="true" /><span>{historyError}</span></div>}
          <p className="visually-hidden" role="status">{historyLoading ? "Loading recorded decisions…" : historyError ? "" : historyRows.length ? `${historyRows.length} recent decisions loaded.` : "No persisted decisions yet."}</p>
          {historyRows.length === 0 ? (
            <div className="history-empty">{historyLoading ? "Loading recorded decisions…" : historyError ? "Decision history could not be loaded." : "No persisted decisions yet. Score a transaction to create the first audit record."}</div>
          ) : (
            <div className="history-table-wrap">
              <table className="history-table">
                <thead><tr><th scope="col">TRANSACTION</th><th scope="col">OUTCOME</th><th scope="col">RISK</th><th scope="col">MODEL / POLICY</th><th scope="col">RECORDED</th></tr></thead>
                <tbody>
                  {historyRows.map((item) => (
                    <tr key={item.id ?? item.transaction_id}>
                      <td><strong>{item.transaction_id}</strong><small>Audit #{item.id ?? "session"}</small></td>
                      <td><span className={`history-decision ${decisionMeta[item.decision].className}`}>{item.decision}</span></td>
                      <td className="history-probability">{(item.risk_probability * 100).toFixed(2)}%</td>
                      <td><strong>{item.model_version}</strong><small>{item.calibration_method} · {item.policy_version} · {item.feature_version}</small></td>
                      <td className="history-time">{item.created_at ? new Date(item.created_at).toLocaleString() : "Not timestamped"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="panel review-panel" id="review" aria-labelledby="review-heading">
          <div className="panel-heading">
            <div><div className="section-kicker">HUMAN-IN-THE-LOOP</div><h2 id="review-heading">Manual review queue</h2><p>Inspect REVIEW decisions and record a separate human outcome.</p></div>
            <span className="panel-symbol"><ShieldAlert size={18} aria-hidden="true" /></span>
          </div>
          <div className="review-access">
            <label className="field"><span>Reviewer API key <em>held in memory only</em></span><input type="password" autoComplete="off" placeholder="Enter the server-configured reviewer key" value={reviewerKey} disabled={reviewLoading || reviewActionId !== null} onChange={(e) => changeReviewerKey(e.target.value)} /></label>
            <button className="secondary-button" type="button" onClick={() => void loadReviewQueue()} aria-busy={reviewLoading} disabled={reviewLoading || reviewActionId !== null || !reviewerKey.trim()}>{reviewLoading ? "Loading…" : "Load pending cases"}</button>
          </div>
          <p className="form-hint"><CircleHelp size={14} aria-hidden="true" /> The key is sent only in the request header and is not saved in browser storage. Configure a local reviewer key on the API before using this workflow.</p>
          {reviewError && <div className="notice error" role="alert"><AlertCircle size={16} aria-hidden="true" /><span>{reviewError}</span></div>}
          <p className={reviewNotice ? "notice success" : "visually-hidden"} role="status">{reviewLoading ? "Loading review cases…" : reviewActionId !== null ? "Recording review outcome…" : reviewNotice || (reviewLoaded ? `${reviewCases.length} pending review cases loaded.` : "")}</p>
          {reviewCases.length === 0 ? (
            <div className="history-empty">{reviewLoading ? "Loading review cases…" : reviewError ? "Review queue could not be loaded." : reviewLoaded ? "No pending review cases." : "No pending cases loaded. Authenticate to retrieve the review queue."}</div>
          ) : (
            <div className="review-list">
              {reviewCases.map((item) => (
                <article className="review-card" key={item.case_id} aria-labelledby={`review-case-${item.case_id}`}>
                  <div className="review-card-top">
                    <div><span className="review-case-id">CASE #{item.case_id} · DECISION #{item.decision_id}</span><h3 id={`review-case-${item.case_id}`}>{item.transaction_id}</h3><p>Automated outcome <span className="history-decision review">REVIEW</span></p></div>
                    <div className="review-risk"><strong>{(item.risk_probability * 100).toFixed(2)}%</strong><small>CALIBRATED RISK</small></div>
                  </div>
                  <div className="review-details">
                    <div><span>MODEL / CALIBRATION / POLICY</span><strong>{item.model_version} · {item.calibration_method} · {item.policy_version}</strong><small>{item.feature_version} · {item.scored_inputs.ProductCD ? String(item.scored_inputs.ProductCD) : "Product not supplied"}</small></div>
                    <div><span>TRANSACTION AMOUNT</span><strong>{typeof item.scored_inputs.TransactionAmt === "number" ? `${item.scored_inputs.TransactionAmt.toFixed(2)}` : "Not supplied"}</strong><small>History rows used: recorded in audit decision</small></div>
                  </div>
                  <p className="review-explanation">{item.decision_explanation}</p>
                  <div className="review-signals">
                    {item.reason_codes.slice(0, 3).map((signal) => <span key={`${signal.direction}:${signal.feature}`}>{signal.title}</span>)}
                  </div>
                  <label className="field review-comment"><span>Reviewer notes <em>optional</em></span><textarea rows={2} value={reviewComments[item.case_id] ?? ""} onChange={(e) => setReviewComments((current) => ({ ...current, [item.case_id]: e.target.value }))} placeholder="Record the evidence or rationale for this outcome…" /></label>
                  <div className="review-actions">
                    <button className="approve-button" type="button" disabled={reviewActionId !== null} aria-busy={reviewActionId === item.case_id} onClick={() => void resolveCase(item.case_id, "APPROVED")}><CheckCircle2 size={15} aria-hidden="true" /> Clear transaction</button>
                    <button className="reject-button" type="button" disabled={reviewActionId !== null} aria-busy={reviewActionId === item.case_id} onClick={() => void resolveCase(item.case_id, "REJECTED")}><Shield size={15} aria-hidden="true" /> Confirm suspected fraud</button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <footer className="footer"><span>RiskForge <b>·</b> Fraud detection &amp; risk decisioning</span><span>Research prototype · Not connected to a live payment network</span></footer>
      </main>
    </div>
  );
}

export default App;