use crate::args::App;
use crate::schedule::Scheduler;
use crate::task::RunTask;

#[derive(Clone)]
pub struct AppState<Runner: RunTask + Clone + Send + Sync + 'static> {
    auth_key: String,
    scheduler: Scheduler<Runner>,
}

impl<Runner> AppState<Runner>
where
    Runner: RunTask + Clone + Send + Sync + 'static,
{
    pub fn new(args: &App, runner: Runner) -> Self {
        Self {
            auth_key: args.auth_key.clone(),
            scheduler: Scheduler::new(
                args.max_concurrency,
                args.max_memory_size,
                args.model_timeout,
                runner,
            ),
        }
    }

    pub fn auth_key(&self) -> &str {
        &self.auth_key
    }

    pub fn scheduler(&self) -> &Scheduler<Runner> {
        &self.scheduler
    }
}
