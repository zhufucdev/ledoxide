use std::sync::Arc;

use crate::schedule::Scheduler;

#[derive(Clone)]
pub struct AppState {
    auth_key: String,
    scheduler: Arc<Scheduler>,
}

impl AppState {
    pub fn new(auth_key: String) -> Self {
        Self {
            auth_key,
            scheduler: Default::default(),
        }
    }

    pub fn auth_key(&self) -> &str {
        &self.auth_key
    }

    pub fn scheduler(&self) -> &Scheduler {
        self.scheduler.as_ref()
    }
}
