use std::sync::Arc;

use crate::{
    args,
    models::ModelProducer,
    schedule::{
        Scheduler, default_lm_model, default_vlm_model, large_vlm_model, offline_large_vlm_model,
        offline_lm_model, offline_vlm_model,
    },
};

#[derive(Clone)]
pub struct AppState {
    auth_key: String,
    scheduler: Arc<Scheduler>,
}

impl AppState {
    pub fn new(args: &args::App) -> Self {
        let scheduler = if args.offline {
            if args.large_model {
                Scheduler::new_singular(
                    args.max_concurrency,
                    args.max_memory_size,
                    args.model_timeout,
                    ModelProducer::new(offline_large_vlm_model),
                )
            } else {
                Scheduler::new(
                    args.max_concurrency,
                    args.max_memory_size,
                    args.model_timeout,
                    ModelProducer::new(offline_vlm_model),
                    ModelProducer::new(offline_lm_model),
                )
            }
        } else if args.large_model {
            Scheduler::new_singular(
                args.max_concurrency,
                args.max_memory_size,
                args.model_timeout,
                ModelProducer::new(large_vlm_model),
            )
        } else {
            Scheduler::new(
                args.max_concurrency,
                args.max_memory_size,
                args.model_timeout,
                ModelProducer::new(default_vlm_model),
                ModelProducer::new(default_lm_model),
            )
        };
        Self {
            auth_key: args.auth_key.clone(),
            scheduler: Arc::new(scheduler),
        }
    }

    pub fn auth_key(&self) -> &str {
        &self.auth_key
    }

    pub fn scheduler(&self) -> &Scheduler {
        self.scheduler.as_ref()
    }
}
