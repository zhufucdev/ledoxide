use std::sync::Arc;

use crate::{
    args,
    models::ModelProducer,
    schedule::{
        Scheduler, default_vlm_model, large_vlm_model, offline_large_vlm_model, offline_vlm_model,
    },
};

#[derive(Clone)]
pub struct AppState {
    auth_key: String,
    scheduler: Arc<Scheduler>,
}

impl AppState {
    pub fn new(args: &args::App) -> Self {
        let vlm = if args.offline {
            if args.large_model {
                ModelProducer::new(offline_large_vlm_model)
            } else {
                ModelProducer::new(offline_vlm_model)
            }
        } else if args.large_model {
            ModelProducer::new(large_vlm_model)
        } else {
            ModelProducer::new(default_vlm_model)
        };
        Self {
            auth_key: args.auth_key.clone(),
            scheduler: Arc::new(Scheduler::new_singular(
                args.max_concurrency,
                args.max_memory_size,
                args.model_timeout,
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
