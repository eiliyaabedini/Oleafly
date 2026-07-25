// @vitest-environment jsdom

import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AISection } from "./AISection";

const { getConfig, aipassStatus } = vi.hoisted(() => ({
  getConfig: vi.fn(),
  aipassStatus: vi.fn(),
}));

vi.mock("@/lib/tauri", () => ({
  getConfig,
  setConfig: vi.fn(async () => {}),
}));
vi.mock("@/lib/aipass", () => ({
  aipassStatus,
  connectAiPass: vi.fn(),
  disconnectAiPass: vi.fn(),
  listAiPassModels: vi.fn(async () => []),
}));
vi.mock("@/lib/ollama", () => ({
  DEFAULT_OLLAMA_HOST: "http://localhost:11434",
  listOllamaModels: vi.fn(async () => []),
}));
vi.mock("@tauri-apps/plugin-shell", () => ({
  open: vi.fn(async () => {}),
}));

beforeEach(() => {
  getConfig.mockResolvedValue({
    github_token: "",
    github_user: "",
    github_connected: false,
    ai_api_key: "",
    ai_provider: "openai",
    ai_model: "gpt-4o-mini",
    aipass_connected: false,
    ai_keys: {},
    ai_system_prompt: "",
    ai_pdf_capture: true,
    mcp_enabled: false,
    mcp_port: 5323,
    mcp_read_only: false,
    mcp_approval_policy: "ask",
  });
  aipassStatus.mockResolvedValue({
    available: true,
    connected: false,
    profile: null,
  });
});

describe("AI Pass settings", () => {
  it("renders Connect AI Pass as an account action and never an API-key field", async () => {
    render(<AISection />);
    const card = await screen.findByTestId("ai-provider-card-aipass");
    const disclosure = within(card).getByRole("button", { name: "AI Pass" });
    fireEvent.click(disclosure);

    expect(await within(card).findByTestId("ai-provider-connect-aipass")).toHaveTextContent(
      "Connect AI Pass",
    );
    expect(card.querySelector('input[type="password"]')).toBeNull();
    expect(within(card).queryByPlaceholderText("Paste your API key here")).toBeNull();
  });
});
