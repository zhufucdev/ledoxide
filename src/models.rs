use std::{collections::HashMap, sync::Arc, time::Duration};

use futures::future::BoxFuture;
use mistralrs::Model;
use tokio::{
    sync::{Mutex, RwLock},
    task::JoinHandle,
};

pub struct ModelProducer(Box<dyn Fn() -> BoxFuture<'static, anyhow::Result<Model>> + Send + Sync>);

/// unloads the model when not in use
pub struct ModelManager {
    timeout: Duration,
    cache: Arc<RwLock<HashMap<String, Arc<Model>>>>,
    timeout_jobs: Arc<Mutex<HashMap<String, JoinHandle<()>>>>,
    model_builders: Arc<RwLock<HashMap<String, ModelProducer>>>,
}

impl ModelManager {
    pub fn new(timeout: Duration, model_builders: HashMap<String, ModelProducer>) -> Self {
        Self {
            timeout,
            cache: Arc::new(RwLock::new(HashMap::with_capacity(model_builders.len()))),
            timeout_jobs: Default::default(),
            model_builders: Arc::new(RwLock::new(model_builders)),
        }
    }

    pub async fn get_model(
        &self,
        model_id: impl AsRef<str>,
    ) -> Result<Option<Arc<Model>>, anyhow::Error> {
        if let Some(timeout_job) = self.timeout_jobs.lock().await.get(model_id.as_ref()) {
            log::debug!(target: "model manager", "aborting timeout job for {}", model_id.as_ref());
            timeout_job.abort();
        }
        self.add_timeout_job(model_id.as_ref()).await;
        if let Some(cached) = self.cache.read().await.get(model_id.as_ref()) {
            log::debug!(target: "model manager", "cache hit for model {}", model_id.as_ref());
            return Ok(Some(cached.clone()));
        }
        log::debug!(target: "model manager", "cache missed, building model {}", model_id.as_ref());
        let model_builders = self.model_builders.read().await;
        let Some(builder) = model_builders.get(model_id.as_ref()) else {
            return Ok(None);
        };
        let model = Arc::<Model>::new(builder.0().await?);
        self.cache
            .write()
            .await
            .insert(model_id.as_ref().to_string(), model.clone());
        Ok(Some(model.clone()))
    }

    async fn add_timeout_job(&self, model_id: impl AsRef<str>) {
        let timeout = self.timeout.clone();
        let model_id = model_id.as_ref().to_string();
        let cache = self.cache.clone();
        self.timeout_jobs.lock().await.insert(
            model_id.clone(),
            tokio::task::spawn(async move {
                tokio::time::sleep(timeout).await;
                log::debug!(target: "model manager", "dropping model {}", model_id);
                cache.write().await.remove(model_id.as_str());
            }),
        );
    }
}

impl ModelProducer {
    pub fn new<F, Fut>(f: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<Model>> + Send + 'static,
    {
        Self(Box::new(move || Box::pin(f())))
    }
}
