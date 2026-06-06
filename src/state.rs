use std::sync::Arc;

use ollama_rs::Ollama;

use crate::{
    args, ext::FromEnvVars, models::ModelProducer, schedule::Scheduler, task::ollama::OllamaRunTask,
};

#[derive(Clone)]
pub struct AppState {
    auth_key: String,
    scheduler: Arc<Scheduler<OllamaRunTask>>,
}

impl AppState {
    pub fn new(args: &args::App) -> Self {
        const GEMMA_4_12B: &str = "gemma4:12b";
        let runner = if args.offline {
            if args.large_model {
                ModelProducer::new(async || {
                    Ok(OllamaRunTask {
                        ollama: Ollama::from_env_vars(),
                        caption_model: GEMMA_4_12B.into(),
                        extract_model: GEMMA_4_12B.into(),
                    })
                })
            } else {
                ModelProducer::new(async || Ok(OllamaRunTask::default()))
            }
        } else if args.large_model {
            ModelProducer::new(async || {
                Ok(OllamaRunTask {
                    ollama: Ollama::from_env_vars(),
                    caption_model: GEMMA_4_12B.into(),
                    extract_model: GEMMA_4_12B.into(),
                }
                .pull_models()
                .await?)
            })
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
