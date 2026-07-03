"use client";

import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface PlotlyChartProps {
  figure: Record<string, unknown>;
  title?: string;
}

export function PlotlyChart({ figure, title }: PlotlyChartProps) {
  const data = (figure.data as Record<string, unknown>[]) || [];
  const baseLayout = (figure.layout as Record<string, unknown>) || {};
  const layout = {
    ...baseLayout,
    autosize: true,
    margin: { l: 40, r: 20, t: title ? 40 : 20, b: 40 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    title: title || (figure.layout as { title?: { text?: string } })?.title,
  };

  return (
    <div className="h-[360px] w-full">
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
