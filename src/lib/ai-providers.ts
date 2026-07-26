// The AI provider engine now lives in @oleafly/ai-core; kept here so
// existing `@/lib/ai-providers` imports (and their test mocks) keep working
// while consumers migrate to the package directly.
export * from "@oleafly/ai-core";
import {
  buildModel as buildCoreModel,
  getProvider,
  pickActiveProvider,
  type AIConfigLike,
} from "@oleafly/ai-core";
import { buildAiPassModel } from "@/lib/aipass";

export function buildModel(provider: string, model: string, credential: string) {
  return provider === "aipass"
    ? buildAiPassModel(model)
    : buildCoreModel(provider, model, credential);
}

export function resolveActiveModel(cfg: AIConfigLike): {
  model: ReturnType<typeof buildModel>;
  providerId: string;
  modelId: string;
  label: string;
} {
  const { providerId, modelId, credential } = pickActiveProvider(cfg);
  const label =
    getProvider(providerId)?.models.find((candidate) => candidate.id === modelId)?.name ??
    modelId;
  return {
    model: buildModel(providerId, modelId, credential),
    providerId,
    modelId,
    label,
  };
}
