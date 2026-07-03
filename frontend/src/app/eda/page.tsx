"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { PlotlyChart } from "@/components/plotly-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type ChartItem } from "@/lib/api";
import { useSession } from "@/lib/session";

export default function EDAPage() {
  const router = useRouter();
  const { sessionId } = useSession();
  const [charts, setCharts] = useState<ChartItem[]>([]);
  const [insights, setInsights] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) router.push("/");
  }, [sessionId, router]);

  useEffect(() => {
    if (!sessionId) return;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await api.eda(sessionId);
        setCharts(result.eda.charts || []);
        setInsights(result.eda.insights || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load EDA");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [sessionId]);

  if (!sessionId) return <p className="text-muted-foreground">Upload a dataset first.</p>;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Exploratory Data Analysis</h2>
        <p className="mt-2 text-muted-foreground">Interactive charts generated automatically from your dataset.</p>
      </div>

      {loading && <p className="text-sm text-muted-foreground">Generating charts...</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {insights.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Insights</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
              {insights.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {charts.map((chart) => (
          <Card key={chart.id}>
            <CardHeader>
              <CardTitle className="text-base">{chart.column}</CardTitle>
              <CardDescription>{chart.type}</CardDescription>
            </CardHeader>
            <CardContent>
              <PlotlyChart figure={chart.figure} />
            </CardContent>
          </Card>
        ))}
      </div>

      {!loading && charts.length === 0 && (
        <p className="text-sm text-muted-foreground">No charts could be generated for this dataset.</p>
      )}

      <Button onClick={() => router.push("/dashboard")}>View Dashboard</Button>
    </div>
  );
}
