"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { PlotlyChart } from "@/components/plotly-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { api, type ChartItem } from "@/lib/api";
import { useSession } from "@/lib/session";

export default function EDAPage() {
  const router = useRouter();
  const { sessionId, profile } = useSession();
  const [charts, setCharts] = useState<ChartItem[]>([]);
  const [customCharts, setCustomCharts] = useState<ChartItem[]>([]);
  const [insights, setInsights] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [customLoading, setCustomLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [xColumn, setXColumn] = useState("");
  const [yColumn, setYColumn] = useState("");
  const [chartType, setChartType] = useState<"scatter" | "line" | "box" | "histogram">("scatter");

  const columns = profile?.columns || [];

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
        setCustomCharts(result.eda.custom_charts || []);
        setInsights(result.eda.insights || []);
        if (!xColumn && result.eda.charts?.[0]) {
          setXColumn(result.eda.charts[0].column.split("/")[0] || columns[0] || "");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load EDA");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [sessionId]);

  useEffect(() => {
    if (!xColumn && columns.length) setXColumn(columns[0]);
    if (!yColumn && columns.length > 1) setYColumn(columns[1]);
  }, [columns, xColumn, yColumn]);

  const buildCustomChart = async () => {
    if (!sessionId || !xColumn) return;
    setCustomLoading(true);
    setError(null);
    try {
      const result = await api.customEda(
        sessionId,
        xColumn,
        chartType === "histogram" ? undefined : yColumn || undefined,
        chartType,
      );
      setCustomCharts((prev) => [...prev, result.chart]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to build chart");
    } finally {
      setCustomLoading(false);
    }
  };

  if (!sessionId) return <p className="text-muted-foreground">Upload a dataset first.</p>;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Exploratory Data Analysis</h2>
        <p className="mt-2 text-muted-foreground">Auto charts plus interactive column picker.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Interactive chart builder</CardTitle>
          <CardDescription>Pick X/Y columns and chart type to explore relationships.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-end gap-3">
          <div>
            <label className="mb-1 block text-sm text-muted-foreground">X column</label>
            <Select value={xColumn} onChange={(e) => setXColumn(e.target.value)}>
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </Select>
          </div>
          {chartType !== "histogram" && (
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">Y column</label>
              <Select value={yColumn} onChange={(e) => setYColumn(e.target.value)}>
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </Select>
            </div>
          )}
          <div>
            <label className="mb-1 block text-sm text-muted-foreground">Chart type</label>
            <Select
              value={chartType}
              onChange={(e) => setChartType(e.target.value as typeof chartType)}
            >
              <option value="scatter">Scatter</option>
              <option value="line">Line</option>
              <option value="box">Box</option>
              <option value="histogram">Histogram</option>
            </Select>
          </div>
          <Button onClick={buildCustomChart} disabled={customLoading}>
            {customLoading ? "Building..." : "Build chart"}
          </Button>
        </CardContent>
      </Card>

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

      {customCharts.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          {customCharts.map((chart) => (
            <Card key={chart.id}>
              <CardHeader>
                <CardTitle className="text-base">{chart.column}</CardTitle>
                <CardDescription>Custom · {chart.type}</CardDescription>
              </CardHeader>
              <CardContent>
                <PlotlyChart figure={chart.figure} />
              </CardContent>
            </Card>
          ))}
        </div>
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
