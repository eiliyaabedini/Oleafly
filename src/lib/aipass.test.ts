import { describe, expect, it, vi } from "vitest";
import { createAiPassFetch, type AiPassStreamEvent } from "./aipass";

function encoded(value: string) {
  return btoa(value);
}

describe("AI Pass native fetch bridge", () => {
  it("streams bounded response chunks from native IPC without forwarding browser headers", async () => {
    let receive: ((event: AiPassStreamEvent) => void) | undefined;
    const invoke = vi.fn(async (command: string, args?: Record<string, unknown>) => {
      expect(command).toBe("aipass_chat_completions");
      expect(args).not.toHaveProperty("headers");
      receive?.({ event: "chunk", data: encoded("data: first\n\n") });
      receive?.({ event: "chunk", data: encoded("data: [DONE]\n\n") });
      receive?.({ event: "end" });
      return {
        status: 200,
        content_type: "text/event-stream",
      };
    });
    const fetch = createAiPassFetch(
      invoke,
      (handler) => {
        receive = handler;
        return { onmessage: handler };
      },
    );

    const response = await fetch("https://browser.invalid/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: "Bearer must-not-cross-ipc" },
      body: '{"model":"live","messages":[],"stream":true}',
    });

    expect(response.status).toBe(200);
    expect(await response.text()).toBe("data: first\n\ndata: [DONE]\n\n");
  });

  it("propagates AbortSignal to the native cancellation command", async () => {
    let receive: ((event: AiPassStreamEvent) => void) | undefined;
    const invoke = vi.fn(async (command: string) => {
      if (command === "aipass_chat_completions") {
        receive?.({ event: "chunk", data: encoded("data: partial\n\n") });
        return { status: 200, content_type: "text/event-stream" };
      }
      return true;
    });
    const fetch = createAiPassFetch(
      invoke,
      (handler) => {
        receive = handler;
        return { onmessage: handler };
      },
    );
    const controller = new AbortController();

    await fetch("https://browser.invalid", {
      method: "POST",
      body: '{"model":"live","messages":[],"stream":true}',
      signal: controller.signal,
    });
    controller.abort();

    await vi.waitFor(() => {
      expect(invoke).toHaveBeenCalledWith(
        "aipass_cancel_request",
        expect.objectContaining({ requestId: expect.any(String) }),
      );
    });
  });

  it("fails the browser stream with a sanitized native transport error", async () => {
    let receive: ((event: AiPassStreamEvent) => void) | undefined;
    const fetch = createAiPassFetch(
      async () => {
        receive?.({ event: "error", message: "AI Pass response exceeded the size limit." });
        return { status: 200, content_type: "text/event-stream" };
      },
      (handler) => {
        receive = handler;
        return { onmessage: handler };
      },
    );
    const response = await fetch("https://browser.invalid", {
      method: "POST",
      body: '{"model":"live","messages":[],"stream":true}',
    });

    await expect(response.text()).rejects.toThrow("exceeded the size limit");
  });

  it("cancels native work when a response chunk cannot be decoded", async () => {
    let receive: ((event: AiPassStreamEvent) => void) | undefined;
    const invoke = vi.fn(async (command: string) => {
      if (command === "aipass_chat_completions") {
        return { status: 200, content_type: "text/event-stream" };
      }
      return true;
    });
    const fetch = createAiPassFetch(
      invoke,
      (handler) => {
        receive = handler;
        return { onmessage: handler };
      },
    );
    const response = await fetch("https://browser.invalid", {
      method: "POST",
      body: '{"model":"live","messages":[],"stream":true}',
    });

    receive?.({ event: "chunk", data: "not valid base64!" });

    await expect(response.text()).rejects.toThrow("invalid response chunk");
    expect(invoke).toHaveBeenCalledWith(
      "aipass_cancel_request",
      expect.objectContaining({ requestId: expect.any(String) }),
    );
  });
});
