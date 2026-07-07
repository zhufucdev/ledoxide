use std::sync::Arc;

use ollama_rs::Ollama;
use smol_str::ToSmolStr;

use crate::{args, ext::FromEnvVars, schedule::Scheduler, task::ollama::OllamaRunTask};

#[derive(Clone)]
pub struct AppState {
    auth_key: String,
    scheduler: Arc<Scheduler<OllamaRunTask>>,
}

impl AppState {
    pub fn new(args: &args::App) -> Self {
        let caption_model = args.caption_model.to_smolstr();
        let extract_model = args.extract_model.to_smolstr();
        let runner = OllamaRunTask {
            ollama: Ollama::from_env_vars(),
            caption_model: caption_model.clone(),
            extract_model: extract_model.clone(),
            offline: args.offline,
        };
        Self {
            auth_key: args.auth_key.clone(),
            scheduler: Arc::new(Scheduler::new(
                args.max_concurrency,
                args.max_memory_size,
                args.model_timeout,
                runner,
            )),
        }
    }

    pub fn auth_key(&self) -> &str {
        &self.auth_key
    }

    pub fn scheduler(&self) -> &Scheduler<OllamaRunTask> {
        self.scheduler.as_ref()
    }
}
