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
  explainability?: {
    method: string;
    top_features: FeatureImportanceItem[];
  };
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

  clean: (sessionId: string, targetColumn?: string) =>
    request<{ cleaning_report: CleaningReport }>("/clean", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, target_column: targetColumn || null }),
    }),

  eda: (sessionId: string) =>
    request<{ eda: { charts: ChartItem[]; insights: string[]; chart_count: number } }>(
      `/eda?session_id=${sessionId}`,
    ),

  train: (sessionId: string, targetColumn: string) =>
    request<{ ml_results: MLResults }>("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, target_column: targetColumn }),
    }),

  ask: (sessionId: string, question: string) =>
    request<{ question: string; answer: string; model_used?: string }>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question }),
    }),

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
