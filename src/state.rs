use std::sync::Arc;

use crate::{
    args,
    models::ModelProducer,
    schedule::{Scheduler, default_lm_model, default_vlm_model, large_lm_model, large_vlm_model},
};

#[derive(Clone)]
pub struct AppState {
    auth_key: String,
    scheduler: Arc<Scheduler>,
}

impl AppState {
    pub fn new(args: &args::App) -> Self {
        let (lm, vlm) = if args.large_model {
            (
                ModelProducer::new(large_lm_model),
                ModelProducer::new(large_vlm_model),
            )
        } else {
            (
                ModelProducer::new(default_lm_model),
                ModelProducer::new(default_vlm_model),
            )
        };
        Self {
            auth_key: args.auth_key.clone(),
            scheduler: Arc::new(Scheduler::new(
                args.max_concurrency,
                args.max_memory_size,
                args.model_timeout,
                lm,
                vlm,
            )),
        }
    }

    pub fn auth_key(&self) -> &str {
        &self.auth_key
    }

    pub fn scheduler(&self) -> &Scheduler {
        self.scheduler.as_ref()
    }
}
