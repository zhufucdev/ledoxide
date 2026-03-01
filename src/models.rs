use std::{sync::Arc, time::Duration};

use anyhow::Ok;
use futures::future::BoxFuture;
use tokio::{sync::RwLock, task::JoinHandle};

use crate::runner::{Gemma3TextRunner, Gemma3VisionRunner};

pub struct ModelProducer<Model>(
    Box<dyn Fn() -> BoxFuture<'static, anyhow::Result<Model>> + Send + Sync>,
);
pub type VisionModel = Gemma3VisionRunner;
pub type TextModel = Gemma3TextRunner;
pub type VisionModelProducer = ModelProducer<VisionModel>;
pub type TextModelProducer = ModelProducer<TextModel>;

/// unloads the model when not in use
pub struct TimedModel<Model> {
    name: String,
    timeout: Duration,
    cache: Arc<RwLock<Option<Arc<Model>>>>,
    timeout_job: RwLock<Option<JoinHandle<()>>>,
    builder: ModelProducer<Model>,
}

impl<Model> TimedModel<Model> {
    pub fn new(name: impl ToString, timeout: Duration, builder: ModelProducer<Model>) -> Self {
        Self {
            name: name.to_string(),
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
            log::debug!(target: "model manager", "aborting timeout job for {}", self.name);
            timeout_job.abort();
        }
        self.add_timeout_job().await;
        if let Some(cached) = self.cache.read().await.as_ref() {
            log::debug!(target: "model manager", "cache hit for model {}", self.name);
            return Ok(cached.clone());
        }
        log::debug!(target: "model manager", "cache missed, building model {}", self.name);
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
                log::debug!(target: "timed model", "dropping model");
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
