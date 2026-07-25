import { createOpenAI } from "@ai-sdk/openai";
import { Channel, invoke } from "@tauri-apps/api/core";

export interface AiPassProfile {
  subject: string;
  display_name: string | null;
  email: string | null;
}

export interface AiPassStatus {
  available: boolean;
  connected: boolean;
  profile: AiPassProfile | null;
}

export interface AiPassModel {
  id: string;
  name: string;
  supports_vision: boolean;
}

export interface AiPassDisconnectResult {
  revoked: boolean;
}

interface NativeResponseMeta {
  status: number;
  content_type: string;
}

export type AiPassStreamEvent =
  | { event: "chunk"; data: string }
  | { event: "end" }
  | { event: "error"; message: string };

type InvokeCommand = (
  command: string,
  args?: Record<string, unknown>,
) => Promise<unknown>;

type ChannelLike = { onmessage: (event: AiPassStreamEvent) => void };
type ChannelFactory = (
  handler: (event: AiPassStreamEvent) => void,
) => ChannelLike;

const defaultChannelFactory: ChannelFactory = (handler) =>
  new Channel<AiPassStreamEvent>(handler);

function decodeBase64(value: string): Uint8Array {
  const binary = atob(value);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function abortError() {
  return new DOMException("The operation was aborted.", "AbortError");
}

export function createAiPassFetch(
  invokeCommand: InvokeCommand = invoke,
  createChannel: ChannelFactory = defaultChannelFactory,
): typeof fetch {
  return async (_input, init) => {
    if (init?.signal?.aborted) throw abortError();
    if (typeof init?.body !== "string") {
      throw new Error("AI Pass native transport requires a JSON request body.");
    }

    const requestId = crypto.randomUUID();
    let streamController!: ReadableStreamDefaultController<Uint8Array>;
    let settled = false;
    let removeAbortListener = () => {};
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        streamController = controller;
      },
      cancel() {
        if (settled) return;
        settled = true;
        removeAbortListener();
        void invokeCommand("aipass_cancel_request", { requestId }).catch(() => {});
      },
    });
    const channel = createChannel((event) => {
      if (settled) return;
      if (event.event === "chunk") {
        try {
          streamController.enqueue(decodeBase64(event.data));
        } catch {
          settled = true;
          removeAbortListener();
          streamController.error(new Error("AI Pass returned an invalid response chunk."));
        }
        return;
      }
      settled = true;
      removeAbortListener();
      if (event.event === "end") {
        streamController.close();
      } else {
        streamController.error(new Error(event.message));
      }
    });

    const onAbort = () => {
      if (settled) return;
      void invokeCommand("aipass_cancel_request", { requestId }).catch(() => {});
    };
    if (init.signal) {
      init.signal.addEventListener("abort", onAbort, { once: true });
      removeAbortListener = () => init.signal?.removeEventListener("abort", onAbort);
    }

    let meta: NativeResponseMeta;
    try {
      meta = (await invokeCommand("aipass_chat_completions", {
        requestId,
        body: init.body,
        onEvent: channel,
      })) as NativeResponseMeta;
    } catch (error) {
      if (!settled) {
        settled = true;
        removeAbortListener();
        streamController.error(error);
      }
      if (init.signal?.aborted) throw abortError();
      throw error;
    }
    if (init.signal?.aborted) {
      if (!settled) {
        settled = true;
        removeAbortListener();
        streamController.error(abortError());
      }
      throw abortError();
    }
    if (!Number.isInteger(meta.status) || meta.status < 200 || meta.status > 599) {
      throw new Error("AI Pass native transport returned an invalid HTTP status.");
    }
    return new Response(body, {
      status: meta.status,
      headers: { "content-type": meta.content_type },
    });
  };
}

const aipassFetch = createAiPassFetch();

export function buildAiPassModel(model: string) {
  if (!model) throw new Error("Choose a discovered AI Pass model first.");
  return createOpenAI({
    name: "aipass",
    apiKey: "native-managed-credential",
    // No request reaches this sentinel origin. The injected fetch passes only
    // the JSON body to Rust, which pins the real AI Pass endpoint.
    baseURL: "https://native.aipass.invalid/v1",
    fetch: aipassFetch,
  }).chat(model);
}

export const aipassStatus = () => invoke<AiPassStatus>("aipass_status");
export const connectAiPass = () => invoke<AiPassStatus>("aipass_connect");
export const disconnectAiPass = () =>
  invoke<AiPassDisconnectResult>("aipass_disconnect");
export const listAiPassModels = () =>
  invoke<AiPassModel[]>("aipass_list_models");
