use std::{collections::HashMap, sync::Arc, time::Duration};

use mistralrs::{PagedAttentionMetaBuilder, TextModelBuilder, VisionModelBuilder};
use tokio::{sync::Mutex, task::JoinHandle};

use crate::{
    models::{ModelBuilder, ModelManager},
    task::{self, TaskControlBlock, TaskDescriptor},
};

pub struct Scheduler {
    active_queue: Arc<Mutex<Vec<(TaskControlBlock, JoinHandle<()>)>>>,
    pending_queue: Arc<Mutex<Vec<(TaskControlBlock, Arc<TaskDescriptor>)>>>,
    max_concurrency: usize,
    vlm_id: String,
    lm_id: String,
    model_manager: Arc<ModelManager>,
}

impl Scheduler {
    pub fn new(
        max_concurrency: usize,
        model_timeout: Duration,
        vlm_id: impl ToString + Send + Sync + 'static,
        lm_id: impl ToString + Send + Sync + 'static,
    ) -> Self {
        let (vlm_id_copy, lm_id_copy) = (vlm_id.to_string(), lm_id.to_string());
        let vlm_builder: ModelBuilder = Box::new(move || {
            let vlm_id = vlm_id.to_string();
            Box::pin(async move {
                VisionModelBuilder::new(vlm_id)
                    .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
                    .build()
                    .await
            })
        });
        let lm_builder: ModelBuilder = Box::new(move || {
            let lm_id = lm_id.to_string();
            Box::pin(async move {
                TextModelBuilder::new(lm_id)
                    .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
                    .build()
                    .await
            })
        });
        Self {
            active_queue: Arc::new(Mutex::new(Vec::new())),
            pending_queue: Arc::new(Mutex::new(Vec::new())),
            max_concurrency,
            vlm_id: vlm_id_copy.clone(),
            lm_id: lm_id_copy.clone(),
            model_manager: Arc::new(ModelManager::new(
                model_timeout,
                HashMap::from([(vlm_id_copy, vlm_builder), (lm_id_copy, lm_builder)]),
            )),
        }
    }

    pub async fn create_task(&self, descriptor: TaskDescriptor) -> TaskControlBlock {
        let task = TaskControlBlock::new();
        self.pending_queue
            .lock()
            .await
            .push((task.clone(), Arc::new(descriptor)));
        let task_run = self.try_run_topmost().await;
        log::info!(target: "scheduler", "running topmost {} tasks", task_run);
        task
    }

    /// returns the number of tasks that were run
    async fn try_run_topmost(&self) -> usize {
        let mut active_queue = self.active_queue.lock().await;
        let original_active_tasks = active_queue.len();
        let mut pending_queue = self.pending_queue.lock().await;
        log::debug!(target: "scheduler",
            "try running topmost {}, active count = {}, max concurrency = {}",
            pending_queue.len(), original_active_tasks, self.max_concurrency);
        for _ in 0..self.max_concurrency - active_queue.len() {
            if let Some((tcb, descriptor)) = pending_queue.pop() {
                tcb.set_state(task::State::Running);
                let (mm, vlm, lm) = (
                    self.model_manager.clone(),
                    self.vlm_id.clone(),
                    self.lm_id.clone(),
                );
                active_queue.push((
                    tcb.clone(),
                    tokio::spawn(async move {
                        tcb.set_state(task::State::Finished(
                            match descriptor.run(mm.as_ref(), vlm, lm).await {
                                Ok(bill) => Ok(task::Success(bill)),
                                Err(err) => Err(Arc::new(err)),
                            },
                        ));
                    }),
                ));
            }
        }

        active_queue.len() - original_active_tasks
    }

    pub async fn get_task(&self, task_id: String) -> Option<TaskControlBlock> {
        std::iter::chain(
            self.active_queue.lock().await.iter().map(|(task, _)| task),
            self.pending_queue.lock().await.iter().map(|(task, _)| task),
        )
        .collect::<Vec<&TaskControlBlock>>()
        .into_iter()
        .find(|task| task.id() == task_id)
        .cloned()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(4, Duration::from_mins(5), "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-4B-Instruct-2507")
    }
}
