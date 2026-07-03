import fs from "fs";
import path from "path";

import { test, expect } from "@playwright/test";

const API_URL = process.env.API_URL || "http://localhost:8000";
const FRONTEND_URL = process.env.FRONTEND_URL || "http://localhost:3000";

test.describe("AI Data Analyst demo flow", () => {
  test("upload → clean → train → dashboard (API)", async ({ request }) => {
    const samplePath = path.join(__dirname, "..", "sample-data", "sales_sample.csv");
    const uploadRes = await request.post(`${API_URL}/upload`, {
      multipart: {
        file: {
          name: "sales_sample.csv",
          mimeType: "text/csv",
          buffer: fs.readFileSync(samplePath),
        },
      },
    });
    expect(uploadRes.ok()).toBeTruthy();
    const upload = await uploadRes.json();
    const sessionId = upload.session_id as string;
    expect(sessionId).toBeTruthy();

    const cleanRes = await request.post(`${API_URL}/clean`, {
      data: { session_id: sessionId, outlier_strategy: "winsorize" },
    });
    expect(cleanRes.ok()).toBeTruthy();

    const customEdaRes = await request.post(`${API_URL}/eda/custom`, {
      data: { session_id: sessionId, x_column: "sales", y_column: "units_sold", chart_type: "scatter" },
    });
    expect(customEdaRes.ok()).toBeTruthy();

    const clusterRes = await request.post(`${API_URL}/clustering`, {
      data: { session_id: sessionId },
    });
    expect(clusterRes.ok()).toBeTruthy();

    const trainRes = await request.post(`${API_URL}/train`, {
      data: { session_id: sessionId, target_column: "region", async_mode: false },
    });
    expect(trainRes.ok()).toBeTruthy();
    const train = await trainRes.json();
    expect(train.ml_results?.best_model).toBeTruthy();

    const dashRes = await request.get(`${API_URL}/dashboard?session_id=${sessionId}`);
    expect(dashRes.ok()).toBeTruthy();
    const dash = await dashRes.json();
    expect(dash.dashboard?.kpis?.length).toBeGreaterThan(0);

    const metricsRes = await request.get(`${API_URL}/metrics`);
    expect(metricsRes.ok()).toBeTruthy();
  });

  test("frontend home loads", async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await expect(page.getByRole("heading", { name: "AI Data Analyst" })).toBeVisible();
  });
});
