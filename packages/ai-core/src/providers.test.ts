import { describe, it, expect } from "vitest";
import {
  PROVIDERS,
  getProvider,
  defaultModel,
  credentialMeta,
  hasConfiguredProvider,
  pickActiveProvider,
} from "./providers";

describe("ai-providers", () => {
  it("ships a non-empty provider catalog, each with id/name/models", () => {
    expect(PROVIDERS.length).toBeGreaterThan(0);
    for (const p of PROVIDERS) {
      expect(p.id).toBeTruthy();
      expect(p.name).toBeTruthy();
      expect(Array.isArray(p.models)).toBe(true);
    }
  });

  it("getProvider resolves a known id and is undefined for an unknown one", () => {
    expect(getProvider("openai")?.id).toBe("openai");
    expect(getProvider("does-not-exist")).toBeUndefined();
  });

  it("defaultModel returns the provider's first model, or a safe fallback", () => {
    const first = getProvider("openai")?.models[0]?.id;
    expect(defaultModel("openai")).toBe(first);
    expect(defaultModel("does-not-exist")).toBe("gpt-4o-mini");
  });

  it("credentialMeta asks for a host URL for local Ollama, an API key otherwise", () => {
    expect(credentialMeta("ollama").label).toBe("Host URL");
    expect(credentialMeta("ollama").placeholder).toContain("localhost");
    expect(credentialMeta("openai").label).toBe("API key");
  });

  it("describes AI Pass as an OAuth account with no hardcoded models", () => {
    const provider = getProvider("aipass");
    expect(provider?.auth).toBe("oauth");
    expect(provider?.models).toEqual([]);
    expect(defaultModel("aipass")).toBe("");
  });
});

describe("pickActiveProvider", () => {
  it("uses the saved provider + saved model when the saved provider has a key", () => {
    const r = pickActiveProvider({
      ai_provider: "anthropic",
      ai_model: "claude-3-5-haiku-20241022",
      ai_keys: { anthropic: "sk-ant" },
    });
    expect(r).toEqual({
      providerId: "anthropic",
      modelId: "claude-3-5-haiku-20241022",
      credential: "sk-ant",
    });
  });

  it("falls back to the first configured provider (default model) when the saved one has no key", () => {
    const r = pickActiveProvider({
      ai_provider: "openai",
      ai_model: "gpt-4o",
      ai_keys: { groq: "gsk-x" },
    });
    expect(r.providerId).toBe("groq");
    expect(r.modelId).toBe(defaultModel("groq"));
    expect(r.credential).toBe("gsk-x");
  });

  it("folds the legacy single ai_api_key into the saved provider", () => {
    const r = pickActiveProvider({
      ai_provider: "openai",
      ai_api_key: "sk-legacy",
      ai_keys: {},
    });
    expect(r).toEqual({ providerId: "openai", modelId: defaultModel("openai"), credential: "sk-legacy" });
  });

  it("defaults to openai with an empty credential when nothing is configured", () => {
    const r = pickActiveProvider({});
    expect(r.providerId).toBe("openai");
    expect(r.modelId).toBe(defaultModel("openai"));
    expect(r.credential).toBe("");
  });

  it("selects a connected AI Pass account without putting a token in config", () => {
    const r = pickActiveProvider({
      ai_provider: "aipass",
      ai_model: "discovered-live-model",
      ai_keys: {},
      aipass_connected: true,
    });
    expect(r).toEqual({
      providerId: "aipass",
      modelId: "discovered-live-model",
      credential: "",
    });
    expect(hasConfiguredProvider({
      ai_provider: "aipass",
      ai_model: "discovered-live-model",
      aipass_connected: true,
    })).toBe(true);
  });
});
