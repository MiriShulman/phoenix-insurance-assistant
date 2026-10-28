
import { useState, type FormEvent } from 'react'
import './App.css'

const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

type Source = {
  title: string
  url: string
  snippet: string
}

type Message = {
  id: number
  role: 'assistant' | 'user'
  text: string
  sources?: Source[]
}

const suggestions = [
  'מה כולל ביטוח חובה לרכב?',
  'מה כולל ביטוח מקיף לרכב?',
  'איך עובד ביטוח יומי לרכב?',
]

function App() {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      role: 'assistant',
      text: 'שלום! אני העוזרת הדיגיטלית של הפניקס. אפשר לשאול אותי שאלות על המידע שנאסף מאתר החברה. כשיימצא מידע רלוונטי, אציג גם קישורים למקורות.',
    },
  ])

  async function sendQuestion(question: string) {
    const text = question.trim()

    if (!text || loading) return

    setMessages((previous) => [
      ...previous,
      { id: Date.now(), role: 'user', text },
    ])

    setInput('')
    setLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: text,
          history: messages.slice(-8).map(({ role, text }) => ({ role, content: text })),
        }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data: {
        answer: string
        sources?: Source[]
      } = await response.json()

      setMessages((previous) => [
        ...previous,
        {
          id: Date.now(),
          role: 'assistant',
          text: data.answer,
          sources: data.sources ?? [],
        },
      ])
    } catch (error) {
      console.error('Chat request failed:', error)

      setMessages((previous) => [
        ...previous,
        {
          id: Date.now(),
          role: 'assistant',
          text: 'לא הצלחתי להתחבר לשרת. בדקי שהשרת פועל ונסי שוב.',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    sendQuestion(input)
  }

  return (
    <main className="app" dir="rtl">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">פ</div>
          <div>
            <h1>הפניקס</h1>
            <p>המידע שלך, פשוט יותר</p>
          </div>
        </div>

        <span className="status">
          <span className="status-dot" />
          גרסת הדגמה
        </span>
      </header>

      <section className="hero">
        <div className="eyebrow">העוזרת הדיגיטלית</div>
        <h2>איך אפשר לעזור לך היום?</h2>
        <p>
          שאלי שאלה על המידע שנאסף מאתר הפניקס.
          התשובות יוצגו לצד קטעי מקור וקישורים לעמודים הרלוונטיים.
        </p>
      </section>

      <section className="chat" aria-label="צ׳אט">
        <div className="messages" aria-live="polite">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`message-row ${message.role}`}
            >
              {message.role === 'assistant' && (
                <div className="avatar">פ</div>
              )}

              <div className="message">
                <div className="message-label">
                  {message.role === 'assistant' ? 'הפניקס' : 'את'}
                </div>

                <p>
                  {message.role === 'assistant' &&
                  message.sources?.length
                    ? message.text.split('\n\n')[0]
                    : message.text}
                </p>

                {message.role === 'assistant' &&
                  message.sources &&
                  message.sources.length > 0 && (
                    <div className="source-list">
                      <div className="source-heading">
                        מקורות מהאתר
                      </div>

                      {message.sources.map((source) => (
                        <article
                          className="source-card"
                          key={source.url}
                        >
                          <div className="source-title">
                            {source.title}
                          </div>

                          <p className="source-snippet">
                            {source.snippet}
                          </p>

                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            פתיחת המקור באתר הפניקס ↗
                          </a>
                        </article>
                      ))}
                    </div>
                  )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message-row assistant">
              <div className="avatar">פ</div>
              <div className="message">
                <div className="message-label">הפניקס</div>
                <p>מחפשת מידע במקורות...</p>
              </div>
            </div>
          )}
        </div>

        {messages.length === 1 && (
          <div className="suggestions">
            <span>אפשר להתחיל בשאלה:</span>

            {suggestions.map((suggestion) => (
              <button
                className="suggestion"
                key={suggestion}
                type="button"
                onClick={() => sendQuestion(suggestion)}
                disabled={loading}
              >
                {suggestion}
                <span aria-hidden="true">←</span>
              </button>
            ))}
          </div>
        )}

        <form className="composer" onSubmit={handleSubmit}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="כתבי כאן את השאלה שלך..."
            aria-label="השאלה שלך"
          />

          <button
            className="send-button"
            type="submit"
            disabled={!input.trim() || loading}
          >
            {loading ? 'מחפשת...' : 'שליחה'}
          </button>
        </form>

        <p className="disclaimer">
          גרסת הדגמה — החיפוש מכסה כרגע רק את הדפים שנאספו,
          ולא בהכרח את כל אתר הפניקס.
        </p>
      </section>

      <footer>פרויקט האקתון AI · הפניקס</footer>
    </main>
  )
}

export default App
