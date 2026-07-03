const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = error.detail;
    const message =
      typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d: { msg?: string }) => d.msg).join(", ") : "Request failed";
    throw new Error(message || "Request failed");
  }
  return response.json();
}

export interface UploadResponse {
  session_id: string;
  filename: string;
  preview: Record<string, unknown>[];
  columns: string[];
  profile: ProfileData;
}

export interface ProfileData {
  shape: { rows: number; columns: number };
  columns: string[];
  column_types: Record<string, string>;
  column_roles?: Record<string, string>;
  columns_summary: Record<string, unknown>;
  missing_by_column: Record<string, number>;
  total_missing_cells: number;
  duplicate_rows: number;
  suggested_targets: string[];
  ml_excluded_columns?: string[];
  target_column: string | null;
  task_type: string | null;
  validation_report?: {
    passed: boolean;
    issues: string[];
    checks_run: number;
    rows_validated: number;
    quality_report?: {
      framework: string;
      columns: Array<Record<string, unknown>>;
      summary: Record<string, number>;
    };
  };
}

export interface CleaningReport {
  steps: string[];
  rows_before: number;
  rows_after: number;
  columns_before: number;
  columns_after: number;
  imputation: Record<string, string>;
  encoded_categorical_columns: string[];
}

export interface ChartItem {
  id: string;
  type: string;
  column: string;
  figure: Record<string, unknown>;
}

export interface FeatureImportanceItem {
  feature: string;
  importance?: number;
  mean_abs_shap?: number;
}

export interface MLResults {
  task_type: string;
  best_model: string;
  best_metrics: Record<string, number>;
  leaderboard: Array<{ model: string; metrics?: Record<string, number>; error?: string }>;
  feature_importance: FeatureImportanceItem[];
  warnings?: string[];
  excluded_feature_columns?: string[];
  feature_columns?: string[];
  cross_validation?: { mean?: number; std?: number; scoring?: string; scores?: number[] };
  diagnostics?: Record<string, unknown>;
  model_saved?: boolean;
  explainability?: {
    method: string;
    top_features: FeatureImportanceItem[];
  };
}

export interface TrainJobResponse {
  session_id: string;
  job_id?: string;
  status?: string;
  message?: string;
  ml_results?: MLResults;
}

export interface TrainStatusResponse {
  job_id: string;
  session_id: string;
  status: string;
  progress: number;
  error?: string;
  ml_results?: MLResults;
}

export interface DashboardData {
  title?: string;
  kpis: Array<{ label: string; value: string | number }>;
  charts: ChartItem[];
  profile_summary: Record<string, unknown>;
  cleaning_summary: Record<string, unknown>;
  ml_summary: Record<string, unknown>;
  report_sections: Array<{ title: string; content: string }>;
}

export interface DatasetColumn {
  name: string;
  type: string;
}

export interface DatasetPageResponse {
  session_id: string;
  variant: "raw" | "cleaned";
  filename: string;
  has_cleaned: boolean;
  columns: DatasetColumn[];
  rows: Record<string, unknown>[];
  page: number;
  page_size: number;
  total_rows: number;
  filtered_rows: number;
  total_pages: number;
  row_offset: number;
  sort_by: string | null;
  sort_order: "asc" | "desc";
  search: string | null;
}

export interface DatasetQuery {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
  search?: string;
  variant?: "raw" | "cleaned";
}

export const api = {
  upload: async (file: File): Promise<UploadResponse> => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResponse>("/upload", { method: "POST", body: form });
  },

  profile: (sessionId: string, targetColumn?: string) => {
    const params = new URLSearchParams({ session_id: sessionId });
    if (targetColumn) params.set("target_column", targetColumn);
    return request<{
      profile: ProfileData;
      preview: Record<string, unknown>[];
      active_columns: string[];
    }>(`/profile?${params}`);
  },

  dataset: (sessionId: string, query: DatasetQuery = {}) => {
    const params = new URLSearchParams({ session_id: sessionId });
    if (query.page) params.set("page", String(query.page));
    if (query.pageSize) params.set("page_size", String(query.pageSize));
    if (query.sortBy) params.set("sort_by", query.sortBy);
    if (query.sortOrder) params.set("sort_order", query.sortOrder);
    if (query.search) params.set("search", query.search);
    if (query.variant) params.set("variant", query.variant);
    return request<DatasetPageResponse>(`/dataset?${params}`);
  },

  clean: (sessionId: string, targetColumn?: string, outlierStrategy = "winsorize") =>
    request<{ cleaning_report: CleaningReport }>("/clean", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        target_column: targetColumn || null,
        outlier_strategy: outlierStrategy,
      }),
    }),

  eda: (sessionId: string) =>
    request<{ eda: { charts: ChartItem[]; insights: string[]; chart_count: number; custom_charts?: ChartItem[] } }>(
      `/eda?session_id=${sessionId}`,
    ),

  customEda: (
    sessionId: string,
    xColumn: string,
    yColumn?: string,
    chartType: "scatter" | "line" | "box" | "histogram" = "scatter",
  ) =>
    request<{ chart: ChartItem }>("/eda/custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        x_column: xColumn,
        y_column: yColumn || null,
        chart_type: chartType,
      }),
    }),

  clustering: (sessionId: string, nClusters?: number) =>
    request<{
      clustering: {
        n_clusters: number;
        silhouette_score: number | null;
        cluster_sizes: Record<string, number>;
        scatter: Array<{ x: number; y: number; cluster: number }>;
      };
    }>("/clustering", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, n_clusters: nClusters || null }),
    }),

  anomaly: (sessionId: string, contamination = 0.05) =>
    request<{
      anomaly: {
        anomaly_count: number;
        total_rows: number;
        anomaly_rate: number;
        method: string;
        anomaly_indices: number[];
      };
    }>("/anomaly", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, contamination }),
    }),

  train: (sessionId: string, targetColumn: string, asyncMode = true) =>
    request<TrainJobResponse>("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        target_column: targetColumn,
        async_mode: asyncMode,
      }),
    }),

  trainStatus: (jobId: string) =>
    request<TrainStatusResponse>(`/train/status?job_id=${encodeURIComponent(jobId)}`),

  ask: (sessionId: string, question: string, stream = false) =>
    request<{ question: string; answer: string; model_used?: string }>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question, stream }),
    }),

  askStream: async function* (sessionId: string, question: string) {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question, stream: true }),
    });
    if (!response.ok || !response.body) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || "Stream failed");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.token) yield payload.token as string;
        if (payload.done) return;
      }
    }
  },

  exportDataset: async (sessionId: string, variant: "raw" | "cleaned" = "raw", filename?: string) => {
    const response = await fetch(
      `${API_BASE}/export?session_id=${encodeURIComponent(sessionId)}&variant=${variant}`,
    );
    if (!response.ok) throw new Error("Export failed");
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename || `dataset-${variant}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  },

  dashboard: (sessionId: string) =>
    request<{ dashboard: DashboardData }>(`/dashboard?session_id=${sessionId}`),

  downloadReport: async (sessionId: string, filename?: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/report?session_id=${encodeURIComponent(sessionId)}`);
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || "Failed to download report");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const base = (filename || "dataset").replace(/\.[^.]+$/, "");
    link.download = `${base}-report.html`;
    link.click();
    URL.revokeObjectURL(url);
  },
};
