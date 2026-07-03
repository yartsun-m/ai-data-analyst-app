"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";
import { useSession } from "@/lib/session";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const suggestions = [
  "What are the key trends in this dataset?",
  "Which features matter most for predictions?",
  "Summarize data quality issues.",
  "Why might performance differ across segments?",
];

export default function ChatPage() {
  const router = useRouter();
  const { sessionId } = useSession();
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) router.push("/");
  }, [sessionId, router]);

  const ask = async (text: string) => {
    if (!sessionId || !text.trim()) return;
    setLoading(true);
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setQuestion("");
    try {
      const result = await api.ask(sessionId, text);
      setMessages((prev) => [...prev, { role: "assistant", content: result.answer }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Question failed");
    } finally {
      setLoading(false);
    }
  };

  if (!sessionId) return <p className="text-muted-foreground">Upload a dataset first.</p>;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">AI Assistant</h2>
        <p className="mt-2 text-muted-foreground">
          Ask questions in natural language. The backend sends summarized statistics — never raw rows — to the LLM.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {suggestions.map((item) => (
          <Button key={item} variant="outline" size="sm" onClick={() => ask(item)} disabled={loading}>
            {item}
          </Button>
        ))}
      </div>

      <Card className="min-h-[420px]">
        <CardHeader>
          <CardTitle>Conversation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {messages.length === 0 && (
            <p className="text-sm text-muted-foreground">No messages yet. Ask a question about your dataset.</p>
          )}
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`rounded-lg px-4 py-3 text-sm ${
                msg.role === "user" ? "bg-primary text-primary-foreground ml-12" : "bg-muted mr-12"
              }`}
            >
              {msg.content}
            </div>
          ))}
        </CardContent>
      </Card>

      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          ask(question);
        }}
      >
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Why did sales drop in Q3?"
          disabled={loading}
        />
        <Button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Thinking..." : "Ask"}
        </Button>
      </form>

      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
}
