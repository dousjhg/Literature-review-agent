import { test as base, expect } from "@playwright/test";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:net";
import { mkdtemp, rm } from "node:fs/promises";
import path from "node:path";
import os from "node:os";

const root = path.resolve(import.meta.dirname, "../..");
const test = base.extend<{ appUrl: string }>({
  appUrl: async ({}, use) => {
    const directory = await mkdtemp(
      path.join(os.tmpdir(), "litagent-browser-"),
    );
    const sock = createServer();
    sock.listen(0, "127.0.0.1");
    await once(sock, "listening");
    const port = (sock.address() as { port: number }).port;
    await new Promise<void>((resolve) => sock.close(() => resolve()));
    const child = spawn(
      process.env.LITAGENT_PYTHON || path.join(root, ".venv/bin/python"),
      [path.join(root, "frontend/e2e/serve_fixture.py")],
      {
        cwd: root,
        env: {
          ...process.env,
          LITAGENT_QA_LOGS: directory,
          LITAGENT_QA_PORT: String(port),
        },
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    let log = "";
    child.stdout.on("data", (c) => {
      log = (log + c.toString()).slice(-6000);
    });
    child.stderr.on("data", (c) => {
      log = (log + c.toString()).slice(-6000);
    });
    const appUrl = `http://127.0.0.1:${port}`;
    try {
      let ready = false;
      for (let n = 0; n < 100; n++) {
        if (child.exitCode !== null) throw Error(log);
        try {
          if ((await fetch(appUrl + "/health")).ok) {
            ready = true;
            break;
          }
        } catch {}
        await new Promise((r) => setTimeout(r, 100));
      }
      if (!ready) throw Error("Sample API did not start: " + log);
      await use(appUrl);
    } finally {
      if (child.exitCode === null) {
        child.kill("SIGTERM");
        await Promise.race([
          once(child, "exit"),
          new Promise((r) => setTimeout(r, 3000)),
        ]);
        if (child.exitCode === null) child.kill("SIGKILL");
      }
      await rm(directory, { recursive: true, force: true });
    }
  },
});

test("sample API: analyze, inspect, move, rollback and explicitly save review", async ({
  page,
  request,
  appUrl,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await page.goto(appUrl);
  await expect(
    page.getByRole("heading", { name: "研究主题", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".paper-node")).toHaveCount(6);
  await expect(
    page.getByText("样例模式：分组、相似度和评分为占位结果。"),
  ).toBeVisible();
  await page.getByRole("button", { name: "运行分析", exact: true }).click();
  await expect(page.locator(".version-selector select")).not.toHaveValue(
    "qa-root",
  );
  const analyzed = await page.locator(".version-selector select").inputValue();
  await page.getByRole("button", { name: "关联", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "关联证据", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".inspector blockquote")).toHaveCount(2);
  const original = (
    await (await request.get(appUrl + "/cluster/qa-root")).json()
  ).output;
  expect(original.connections).toHaveLength(0);
  await page.getByRole("button", { name: "全部", exact: true }).click();
  await page.locator(".paper-node").first().click();
  const target = page.getByLabel("选择移动目标");
  await expect(target.locator("option")).not.toHaveCount(1);
  const option = await target
    .locator('option:not([value=""]):not([disabled])')
    .first()
    .getAttribute("value");
  expect(option).toBeTruthy();
  await target.selectOption(option!);
  await page.getByRole("button", { name: "预览移动", exact: true }).click();
  await expect(
    page.getByRole("dialog", { name: "预览分组变更" }),
  ).toBeVisible();
  await expect(page.locator(".change-row")).toHaveCount(2);
  await page
    .getByRole("button", { name: "应用并生成新版本", exact: true })
    .click();
  await expect(page.locator(".version-selector select")).not.toHaveValue(
    analyzed,
  );
  const moved = await page.locator(".version-selector select").inputValue();
  const changed = (
    await (await request.get(appUrl + `/cluster/${moved}`)).json()
  ).output;
  expect(changed.clusters.flatMap((c: any) => c.paper_ids).sort()).toEqual(
    original.clusters.flatMap((c: any) => c.paper_ids).sort(),
  );
  await page.getByRole("button", { name: "版本记录", exact: true }).click();
  const row = page
    .locator(".version-row")
    .filter({ has: page.getByText(analyzed, { exact: true }) });
  await row.getByRole("button", { name: "设为当前版本", exact: true }).click();
  await expect(page.locator(".version-selector select")).toHaveValue(analyzed);
  const history = (
    await (await request.get(appUrl + `/cluster/${analyzed}/versions`)).json()
  ).output;
  expect(history.active_head.run_id).toBe(analyzed);
  expect(history.versions).toHaveLength(3);
  await page.getByRole("button", { name: "人工验收", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "验收材料", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("0 / 6 已完成")).toBeVisible();
  await page.getByRole("button", { name: "保存人工判断", exact: true }).click();
  await expect(page.getByRole("alert")).toHaveText(
    "请选择一项判断：有依据、不支持或资料不足。",
  );
  await page.getByLabel("有依据", { exact: true }).check();
  await page.getByRole("button", { name: "保存人工判断", exact: true }).click();
  await expect(page.getByRole("alert")).toHaveText(
    "请填写审核者姓名或审核标识，可使用单字或缩写。",
  );
  await page.getByLabel("审核者", { exact: true }).fill("X");
  await page.getByRole("button", { name: "保存人工判断", exact: true }).click();
  await expect(page.getByRole("alert")).toHaveText(
    "判断理由至少需要 10 个字符，请补充具体依据。",
  );
  await page
    .getByLabel("判断理由", { exact: true })
    .fill(
      "Sample-only UI regression: the displayed original source excerpts are selected for this temporary test.",
    );
  const quotes = page.getByRole("button", {
    name: "引用选中文字或首段原文",
    exact: true,
  });
  await quotes.nth(0).click();
  await quotes.nth(1).click();
  await page.getByRole("button", { name: "保存人工判断", exact: true }).click();
  await expect(page.getByText("判断已保存至当前版本。")).toBeVisible();
  const review = (
    await (await request.get(appUrl + `/cluster/${analyzed}/review`)).json()
  ).output;
  expect(review.sample_mode).toBe(true);
  expect(review.summary.reviewed).toBe(1);
  expect(
    review.items.find((item: any) => item.annotation)?.annotation.reviewer,
  ).toBe("X");
  expect(review.summary.semantic_accuracy).toBeNull();
  await page.screenshot({ path: info.outputPath("review-saved.png") });
  await page.getByRole("button", { name: "查看已有评分", exact: true }).click();
  await expect(
    page.getByText("当前版本暂无 AI 评分，可点击「AI 自动评分」生成。"),
  ).toBeVisible();
  await page.getByRole("button", { name: "AI 自动评分", exact: true }).click();
  await expect(
    page.getByText("样例流程 · 未评分", { exact: true }),
  ).toBeVisible();
  const afterAgent = (
    await (await request.get(appUrl + `/cluster/${analyzed}/review`)).json()
  ).output;
  expect(afterAgent).toEqual(review);
  await page.getByRole("button", { name: "AI 自动评分", exact: true }).click();
  await expect(
    page.getByText("已读取本版本的已有评审，未重复调用模型。"),
  ).toBeVisible();
  await page.getByRole("button", { name: "保存人工判断", exact: true }).click();
  await expect(page.getByText("判断已保存至当前版本。")).toBeVisible();
  const seen = (
    await (await request.get(appUrl + `/cluster/${analyzed}/review`)).json()
  ).output;
  expect(
    seen.items.find((item: any) => item.annotation)?.annotation
      .saw_model_assessment,
  ).toBe(true);
  await page.locator(".review-item").nth(3).click();
  await expect(
    page.getByText("待明确具体假设 · 未评分", { exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: info.outputPath("agent-review-sample.png") });
  await page.getByRole("button", { name: "论文地图", exact: true }).click();
  await page.getByRole("button", { name: "新建地图", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "新建研究地图" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "导入记录", exact: true }).click();
  await dialog
    .getByRole("textbox", { name: "研究问题", exact: true })
    .fill("新的样例问题：如何比较两种适配方法？");
  await dialog.getByRole("button", { name: "建立地图", exact: true }).click();
  await expect(page.locator(".version-selector select")).not.toHaveValue(
    analyzed,
  );
  const next = await page.locator(".version-selector select").inputValue();
  const newReview = (
    await (await request.get(appUrl + `/cluster/${next}/review`)).json()
  ).output;
  expect(newReview.summary.reviewed).toBe(0);
  await page.getByRole("button", { name: "新建地图", exact: true }).click();
  const importDialog = page.getByRole("dialog", { name: "新建研究地图" });
  await importDialog.getByRole("button", { name: "导入记录", exact: true }).click();
  await importDialog
    .locator('input[type="file"]')
    .setInputFiles(
      path.join(root, "input_examples/cluster-map/lora-translation.json"),
    );
  await expect(
    importDialog.getByRole("textbox", { name: "研究问题", exact: true }),
  ).toHaveValue(
    "How does LoRA fine-tuning improve machine translation for low-resource languages?",
  );
  await importDialog
    .getByRole("button", { name: "建立地图", exact: true })
    .click();
  await expect(page.locator(".paper-node")).toHaveCount(3);
  // Question -> search -> Human review 1 (keep/remove, feedback re-plan) -> map.
  await page.getByRole("button", { name: "新建地图", exact: true }).click();
  const searchDialog = page.getByRole("dialog", { name: "新建研究地图" });
  await searchDialog
    .getByRole("textbox", { name: "研究问题", exact: true })
    .fill("How do large language models handle low-resource languages?");
  await searchDialog.getByRole("button", { name: "检索论文", exact: true }).click();
  const results = searchDialog.getByRole("region", { name: "检索结果审核" });
  await expect(results.getByText("保留 30 / 30 篇")).toBeVisible();
  await expect(results.locator(".plan-criteria")).toContainText("（默认）");
  await results.getByRole("textbox").fill("Only papers since 2023");
  const replan = page.waitForRequest(
    (r) => r.url().endsWith("/search") && !!r.postDataJSON().input.previous_plan,
  );
  await results.getByRole("button", { name: "按反馈重新检索" }).click();
  expect((await replan).postDataJSON().input.feedback).toBe("Only papers since 2023");
  await expect(results.getByRole("textbox")).toHaveValue("");
  await results.getByRole("button", { name: "全部取消" }).click();
  for (const box of (await results.getByRole("checkbox").all()).slice(0, 4))
    await box.check();
  await searchDialog
    .getByRole("button", { name: "用保留的 4 篇建立地图", exact: true })
    .click();
  await expect(searchDialog).toBeHidden();
  await expect(page.locator(".paper-node")).toHaveCount(4);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});

test("mobile: readable map, topic navigation, evidence and no horizontal overflow", async ({
  page,
  appUrl,
}, info) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(appUrl);
  await expect(page.locator(".paper-node")).toHaveCount(6);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "切换主题导航", exact: true }).click();
  await expect(page.getByLabel("搜索论文或主题")).toBeVisible();
  await page.locator(".topic-row").first().click();
  await expect(page.locator(".topic-sidebar")).not.toBeVisible();
  await page.locator(".inspector .paper-list button").first().click();
  await expect(
    page.getByRole("heading", { name: "论文详情", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".inspector blockquote")).toHaveCount(1);
  await page.getByRole("button", { name: "放大地图", exact: true }).click();
  const canvas = page.locator(".map-svg");
  await canvas.scrollIntoViewIfNeeded();
  const beforePan = await canvas
    .locator(":scope > g")
    .getAttribute("transform");
  const bounds = (await canvas.boundingBox())!;
  const touch = await page.context().newCDPSession(page);
  await touch.send("Input.dispatchTouchEvent", {
    type: "touchStart",
    touchPoints: [{ x: bounds.x + 25, y: bounds.y + bounds.height / 2 }],
  });
  await touch.send("Input.dispatchTouchEvent", {
    type: "touchMove",
    touchPoints: [{ x: bounds.x + 65, y: bounds.y + bounds.height / 2 + 20 }],
  });
  await touch.send("Input.dispatchTouchEvent", {
    type: "touchEnd",
    touchPoints: [],
  });
  await expect(canvas.locator(":scope > g")).not.toHaveAttribute(
    "transform",
    beforePan!,
  );
  await touch.detach();
  await page.getByRole("button", { name: "重置视图", exact: true }).click();
  await page.screenshot({
    path: info.outputPath("mobile-map.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "人工验收", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "验收材料", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});

test("stale preview conflict is visible inside the dialog and does not move papers", async ({
  page,
  request,
  appUrl,
}) => {
  await page.goto(appUrl);
  await page.getByRole("button", { name: "运行分析", exact: true }).click();
  await expect(page.locator(".version-selector select")).not.toHaveValue(
    "qa-root",
  );
  const id = await page.locator(".version-selector select").inputValue();
  await page.locator(".paper-node").first().click();
  const target = page.getByLabel("选择移动目标");
  await expect(target.locator("option")).not.toHaveCount(1);
  await target.selectOption(
    (await target
      .locator('option:not([value=""]):not([disabled])')
      .first()
      .getAttribute("value"))!,
  );
  await page.getByRole("button", { name: "预览移动", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "预览分组变更" });
  await expect(dialog).toBeVisible();
  const current = (await (await request.get(appUrl + `/cluster/${id}`)).json())
    .output;
  expect(
    (
      await request.post(appUrl + "/cluster/qa-root/activate", {
        data: {
          target_run_id: "qa-root",
          expected_head_run_id: id,
          expected_head_version_token: current.version_token,
        },
      })
    ).ok(),
  ).toBe(true);
  await dialog
    .getByRole("button", { name: "应用并生成新版本", exact: true })
    .click();
  await expect(dialog.getByRole("alert")).toContainText("no longer active");
  await expect(
    dialog.getByRole("button", { name: "应用并生成新版本", exact: true }),
  ).toBeDisabled();
  const versions = (
    await (await request.get(appUrl + "/cluster/qa-root/versions")).json()
  ).output;
  expect(versions.versions).toHaveLength(2);
  expect(versions.active_head.run_id).toBe("qa-root");
});
