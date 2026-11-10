import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL

const ORDER = ['negative', 'neutral', 'positive']
const NAMES = { negative: 'Negative', neutral: 'Neutral', positive: 'Positive' }

const EXAMPLES = [
  { label: 'Mixed', text: 'Bad experience today-but the cabin crew was so humble' },
  { label: 'Three parts', text: 'Flight 482 now boards at gate B12. The crew was lovely. Still no bag after 3 days.' },
  { label: 'Thanks, but', text: '@united thanks for the quick help, but I waited two hours on hold' },
]

function percent(value) {
  return Math.round(value * 100)
}

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function analyse() {
    if (!text.trim()) {
      setError('Type or paste a tweet first.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        const message =
          typeof body.detail === 'string'
            ? body.detail
            : 'Please check your text (1 to 500 characters).'
        throw new Error(message)
      }
      setResult(await res.json())
    } catch (err) {
      setResult(null)
      setError(err.message || 'Could not reach the server. Try again in a moment.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page">
      <h1>Tweet sentiment analyser</h1>
      <p className="subtitle">
        Paste a tweet about an airline. The model reads it as a whole and part by part.
      </p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        maxLength={500}
        placeholder="Type or paste a tweet here"
      />
      <div className="row">
        <span className="count">{text.length} / 500</span>
        <button onClick={analyse} disabled={loading}>
          {loading ? 'Analysing...' : 'Analyse'}
        </button>
      </div>

      <div className="examples">
        <span>Try:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex.label} type="button" className="chip" onClick={() => setText(ex.text)}>
            {ex.label}
          </button>
        ))}
      </div>

      {loading && (
        <p className="note">
          The first request after a quiet period can take up to 30 seconds while the model wakes up.
        </p>
      )}
      {error && <p className="error">{error}</p>}

      {result && (
        <section className="card">
          <p className="label">Tweet, part by part</p>
          <p className="tweet">
            {result.parts.map((p, i) => (
              <span
                key={i}
                className={`part ${p.label}`}
                title={`${NAMES[p.label]} ${percent(p.confidence)}%`}
              >
                {p.text}
              </span>
            ))}
          </p>

          <p className="label">Overall</p>
          <div className="verdict">
            <span className={`pill ${result.label}`}>{NAMES[result.label]}</span>
            <span className="muted">{percent(result.confidence)}% confidence</span>
          </div>

          {ORDER.map((l) => (
            <div className="bar-row" key={l}>
              <span>{NAMES[l]}</span>
              <div className="track">
                <div className={`fill ${l}`} style={{ width: `${percent(result.probabilities[l])}%` }} />
              </div>
              <span className="pct">{percent(result.probabilities[l])}%</span>
            </div>
          ))}
        </section>
      )}

      <p className="footnote">
        Model: DistilBERT fine-tuned on the Kaggle Twitter US Airline Sentiment data (macro-F1 about
        0.80 on a held-out test set). It can be fooled by sarcasm, and the part colours were checked
        by eye, not scored.
      </p>
    </main>
  )
}

export default App