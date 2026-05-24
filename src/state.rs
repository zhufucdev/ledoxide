use std::sync::Arc;

use crate::{args, models::ModelProducer, schedule::Scheduler, task::ollama::OllamaRunTask};

#[derive(Clone)]
pub struct AppState {
    auth_key: String,
    scheduler: Arc<Scheduler<OllamaRunTask>>,
}

impl AppState {
    pub fn new(args: &args::App) -> Self {
        let runner = if args.offline {
            if args.large_model {
                ModelProducer::new(async || Ok(OllamaRunTask::default().pull_models().await?))
            } else {
                ModelProducer::new(async || Ok(OllamaRunTask::default().pull_models().await?))
            }
        } else if args.large_model {
            ModelProducer::new(async || Ok(OllamaRunTask::default().pull_models().await?))
        } else {
            ModelProducer::new(async || Ok(OllamaRunTask::default().pull_models().await?))
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
