use llama_cpp_2::{GrammarError, model::LlamaModel, sampling::LlamaSampler};
use serde::{Deserialize, Serialize};
use strum::Display;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimpleSamplingParams {
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub temperature: Option<f32>,
    pub seed: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl SimpleSamplingParams {
    pub fn to_llama(&self) -> LlamaSampler {
        let mut samplers = Vec::new();
        if let Some(k) = self.top_k {
            samplers.push(LlamaSampler::top_k(k));
        }
        if let Some(p) = self.top_p {
            samplers.push(LlamaSampler::top_p(p, 8));
        }
        if let Some(p) = self.presence_penalty
            && let Some(r) = self.repetition_penalty
        {
            samplers.push(LlamaSampler::penalties(-1, p, 0.0, r));
        }
        samplers.push(LlamaSampler::dist(
            self.seed.unwrap_or_else(|| rand::random()),
        ));
        LlamaSampler::chain_simple(samplers)
    }
}

#[derive(Debug, Clone)]
pub struct LlguidanceSamplingParams {
    pub schema: LlguidanceSchema,
    pub data: String,
}

impl LlguidanceSamplingParams {
    pub fn to_llama(&self, model: &LlamaModel) -> Result<LlamaSampler, GrammarError> {
        LlamaSampler::llguidance(model, self.schema.to_string().as_str(), &self.data)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum LlguidanceSchema {
    #[strum(to_string = "json")]
    Json,
    #[strum(to_string = "lark")]
    Lark,
}
