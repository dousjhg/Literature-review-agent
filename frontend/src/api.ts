const prefix = import.meta.env.DEV ? "/api" : "";
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
export async function request<T>(
  path: string,
  options: {
    method?: string;
    body?: unknown;
    signal?: AbortSignal;
    timeout?: number;
  } = {},
): Promise<T> {
  const signal = options.signal
    ? AbortSignal.any([
        options.signal,
        AbortSignal.timeout(options.timeout ?? 25000),
      ])
    : AbortSignal.timeout(options.timeout ?? 25000);
  const response = await fetch(prefix + path, {
    method: options.method ?? "GET",
    signal,
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const result = await response.json();
  if (!response.ok || result.status === "error")
    throw new ApiError(
      result.error?.message ?? `请求失败 (${response.status})`,
      response.status,
    );
  return result.output as T;
}
export const messageOf = (error: unknown) =>
  error instanceof Error
    ? error.name === "TimeoutError"
      ? "请求等待超时，服务可能仍在处理。请刷新版本列表，避免重复执行。"
      : error.message
    : "请求未完成，请重试。";
