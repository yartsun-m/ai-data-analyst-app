"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { PlotlyChart } from "@/components/plotly-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type ChartItem } from "@/lib/api";
import { useSession } from "@/lib/session";

export default function DashboardPage() {
  const router = useRouter();
  const { sessionId, filename } = useSession();
  const [kpis, setKpis] = useState<Array<{ label: string; value: string | number }>>([]);
  const [charts, setCharts] = useState<ChartItem[]>([]);
  const [sections, setSections] = useState<Array<{ title: string; content: string }>>([]);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
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
        const result = await api.dashboard(sessionId);
        setKpis(result.dashboard.kpis || []);
        setCharts(result.dashboard.charts || []);
        setSections(result.dashboard.report_sections || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load dashboard");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [sessionId]);

  const downloadReport = async () => {
    if (!sessionId) return;
    setDownloading(true);
    setError(null);
    try {
      await api.downloadReport(sessionId, filename ?? undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to download report");
    } finally {
      setDownloading(false);
    }
  };

  if (!sessionId) return <p className="text-muted-foreground">Upload a dataset first.</p>;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
          <p className="mt-2 text-muted-foreground">Auto-generated KPIs, charts, and report sections.</p>
        </div>
        <Button variant="outline" onClick={downloadReport} disabled={downloading || loading}>
          {downloading ? "Preparing report…" : "Download Report"}
        </Button>
      </div>

      {loading && <p className="text-sm text-muted-foreground">Building dashboard...</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="grid gap-4 md:grid-cols-4">
        {kpis.map((kpi) => (
          <Card key={kpi.label}>
            <CardHeader className="pb-2">
              <CardDescription>{kpi.label}</CardDescription>
              <CardTitle className="text-xl">{kpi.value}</CardTitle>
            </CardHeader>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {sections.map((section) => (
          <Card key={section.title}>
            <CardHeader>
              <CardTitle className="text-base">{section.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">{section.content}</p>
            </CardContent>
          </Card>
        ))}
      </div>

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
    </div>
  );
}
