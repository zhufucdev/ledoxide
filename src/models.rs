use std::{sync::Arc, time::Duration};

use anyhow::Ok;
use futures::future::BoxFuture;
use tokio::{sync::RwLock, task::JoinHandle};
use tracing::{Level, event};

pub struct ModelProducer<Model>(
    Box<dyn Fn() -> BoxFuture<'static, anyhow::Result<Model>> + Send + Sync>,
);

/// unloads the model when not in use
pub struct TimedModel<Model> {
    timeout: Duration,
    cache: Arc<RwLock<Option<Arc<Model>>>>,
    timeout_job: RwLock<Option<JoinHandle<()>>>,
    builder: ModelProducer<Model>,
}

impl<Model> TimedModel<Model> {
    pub fn new(timeout: Duration, builder: ModelProducer<Model>) -> Self {
        Self {
            timeout,
            cache: Arc::new(RwLock::new(None)),
            timeout_job: RwLock::new(None),
            builder,
        }
    }
}

impl<Model> TimedModel<Model>
where
    Model: Send + Sync + 'static,
{
    pub async fn get_model(&self) -> anyhow::Result<Arc<Model>> {
        if let Some(timeout_job) = self.timeout_job.write().await.take() {
            event!(target: "model manager", Level::TRACE, "aborting timeout job");
            timeout_job.abort();
        }
        self.add_timeout_job().await;
        if let Some(cached) = self.cache.read().await.as_ref() {
            event!(target: "model manager", Level::TRACE, "cache hit");
            return Ok(cached.clone());
        }
        event!(target: "model manager", Level::DEBUG, "cache missed, building");
        let model = Arc::new(self.builder.0().await?);
        *self.cache.write().await = Some(model.clone());
        Ok(model.clone())
    }

    async fn add_timeout_job(&self) {
        let timeout = self.timeout.clone();
        let cache = self.cache.clone();
        self.timeout_job
            .write()
            .await
            .replace(tokio::task::spawn(async move {
                tokio::time::sleep(timeout).await;
                event!(target: "timed model", Level::TRACE, "dropping model");
                *cache.write().await = None;
            }));
    }
}

impl<Model> ModelProducer<Model> {
    pub fn new<F, Fut>(f: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<Model>> + Send + 'static,
    {
        Self(Box::new(move || Box::pin(f())))
    }
}
