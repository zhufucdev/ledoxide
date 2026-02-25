use std::{
    collections::HashMap,
    io::{self, SeekFrom},
    ops::Deref,
    sync::{Arc, LazyLock},
    time::Duration,
};

use anyhow::anyhow;
use async_stream::try_stream;
use futures::{Stream, StreamExt, TryStreamExt, stream};
use mistralrs::{
    IsqBits, Model, ModelBuilder, ModelDType, PagedAttentionMetaBuilder, TextModelBuilder,
    VisionModelBuilder,
};
use tempfile::tempfile;
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    pin,
    sync::Mutex,
    task::JoinHandle,
};

use crate::{
    models::{ModelManager, ModelProducer},
    task::{self, TaskControlBlock, TaskDescriptor},
};

struct ScheduleQueues {
    active: Arc<Mutex<Vec<(TaskControlBlock, JoinHandle<()>)>>>,
    pending: Arc<Mutex<Vec<(TaskControlBlock, Arc<TaskDescriptor>)>>>,
    finished: Arc<Mutex<Vec<TaskControlBlock>>>,
}

pub struct Scheduler {
    queues: Arc<ScheduleQueues>,
    swap_file: Arc<Mutex<File>>,
    max_memory_size: usize,
    max_concurrency: usize,
    model_manager: Arc<ModelManager>,
}

impl Scheduler {
    pub fn new(
        max_concurrency: usize,
        max_memory_size: usize,
        model_timeout: Duration,
        vlm_builder: ModelProducer,
        lm_builder: ModelProducer,
    ) -> Self {
        Self {
            queues: Default::default(),
            max_memory_size,
            swap_file: Arc::new(Mutex::new(tempfile().map(File::from_std).unwrap())),
            max_concurrency,
            model_manager: Arc::new(ModelManager::new(
                model_timeout,
                HashMap::from([
                    ("vlm".to_string(), vlm_builder),
                    ("lm".to_string(), lm_builder),
                ]),
            )),
        }
    }

    pub async fn create_task(&self, descriptor: TaskDescriptor) -> TaskControlBlock {
        let task = TaskControlBlock::new();
        self.queues
            .pending
            .lock()
            .await
            .push((task.clone(), Arc::new(descriptor)));
        let task_run = self.try_run_topmost().await;
        log::info!(target: "scheduler", "running topmost {} tasks", task_run);
        task
    }

    /// returns the number of tasks that were run
    async fn try_run_topmost(&self) -> usize {
        let mut active_queue = self.queues.active.lock().await;
        let original_active_tasks = active_queue.len();
        let mut pending_queue = self.queues.pending.lock().await;
        log::debug!(target: "scheduler",
            "try running topmost {}, active count = {}, max concurrency = {}",
            pending_queue.len(), original_active_tasks, self.max_concurrency);
        for _ in 0..self.max_concurrency - active_queue.len() {
            if let Some((tcb, descriptor)) = pending_queue.pop() {
                tcb.set_state(task::State::Running);
                let mm = self.model_manager.clone();
                let queues = self.queues.clone();
                let swap_file = self.swap_file.clone();
                let max_memory_size = self.max_memory_size;
                active_queue.push((
                    tcb.clone(),
                    tokio::spawn(async move {
                        tcb.set_state(task::State::Finished(
                            match descriptor.run(mm.as_ref(), "vlm", "lm").await {
                                Ok(bill) => Ok(task::Success(bill)),
                                Err(err) => Err(Arc::new(err)),
                            },
                        ));
                        let mut active_queue = queues.active.lock().await;
                        if let Some(index) = active_queue
                            .iter()
                            .position(|(task, _)| task.id() == tcb.id())
                        {
                            let (tcb, _) = active_queue.remove(index);
                            queues.finished.lock().await.push(tcb);

                            tokio::time::sleep(Duration::from_secs(10)).await;
                            if let Err(err) = queues.move_inactive_to_swap(&mut *swap_file.lock().await, max_memory_size).await {
                                log::error!(target: "scheduler", "swap failed, inactive queue now has a crowd of {}: {}", 
                                    queues.finished.lock().await.len(), err);
                            }
                        } else {
                            log::error!(target: "scheduler", "finished task {} not found in active queue", tcb.id());
                        }
                    }),
                ));
            }
        }

        active_queue.len() - original_active_tasks
    }

    pub async fn get_task(
        &self,
        task_id: impl AsRef<str>,
    ) -> anyhow::Result<Option<TaskControlBlock>> {
        let aq = self.queues.active.lock().await;
        let pq = self.queues.pending.lock().await;
        let fq = self.queues.finished.lock().await;
        let stream = stream::iter(aq.iter().map(|(task, _)| task).cloned())
            .chain(stream::iter(pq.iter().cloned().map(|(task, _)| task)))
            .chain(stream::iter(fq.iter().cloned()))
            .map(|task| Ok(task))
            .chain(self.in_disk_queue_iter());
        pin!(stream);
        while let Some(task) = stream.try_next().await? {
            if task.id() == task_id.as_ref() {
                return Ok(Some(task));
            }
        }
        Ok(None)
    }

    fn in_disk_queue_iter(&self) -> impl Stream<Item = anyhow::Result<TaskControlBlock>> {
        async fn get_next_chunk(file: &mut File) -> anyhow::Result<Option<Vec<TaskControlBlock>>> {
            let len = match file.read_u32().await {
                Ok(len) => len,
                Err(err) => {
                    if err.kind() == io::ErrorKind::UnexpectedEof {
                        log::debug!(target: "scheduler", "end of swap file");
                        return Ok(None);
                    } else {
                        return Err(anyhow!(err));
                    }
                }
            };
            log::debug!("len<in> = {}", len);
            let mut buf = vec![0u8; len as usize];
            file.read_exact(&mut buf).await?;
            let chunk: Vec<TaskControlBlock> = postcard::from_bytes(&buf)?;
            Ok(Some(chunk))
        }

        try_stream! {
            let mut swap_file = self.swap_file.lock().await;
            swap_file.rewind().await?;
            while let Some(chunk) = get_next_chunk(&mut *swap_file).await? {
                for task in chunk.into_iter() {
                    yield task;
                }
            }
        }
    }
}

impl ScheduleQueues {
    async fn move_inactive_to_swap(
        &self,
        fd: &mut File,
        max_memory_size: usize,
    ) -> anyhow::Result<usize> {
        fd.seek(SeekFrom::End(0)).await?;
        let mut finished_queue = self.finished.lock().await;
        let swap_amount = finished_queue.len() as i32 - max_memory_size as i32;
        if swap_amount <= 0 {
            log::debug!(target: "scheduler", "finished queue size {} <= max memory size {}, no need to swap", finished_queue.len(), max_memory_size);
            return Ok(0);
        }
        let items_left = finished_queue.split_off(swap_amount as usize);
        let items_swapped = finished_queue.len();
        let buf = postcard::to_allocvec(finished_queue.as_slice())?;
        log::debug!("len<out> = {}", buf.len());
        fd.write_u32(buf.len() as u32).await?;
        fd.write(buf.as_slice()).await?;
        fd.flush().await?;
        *finished_queue = items_left;
        Ok(items_swapped)
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(
            4,
            468_000, // approx. 50 megabytes
            Duration::from_mins(5),
            ModelProducer::new(default_vlm_model),
            ModelProducer::new(default_lm_model),
        )
    }
}

#[cfg(feature = "quantize")]
pub async fn default_lm_model() -> anyhow::Result<Model> {
    TextModelBuilder::new("ibm-granite/granite-4.0-micro")
        .with_auto_isq(IsqBits::Eight)
        .build()
        .await
}

pub async fn default_lm_model() -> anyhow::Result<Model> {
    TextModelBuilder::new("ibm-granite/granite-4.0-micro")
        .build()
        .await
}

#[cfg(feature = "quantize")]
pub async fn default_vlm_model() -> anyhow::Result<Model> {
    ModelBuilder::new("google/gemma-3-4b-it")
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .with_auto_isq(IsqBits::Four)
        .build()
        .await
}

pub async fn default_vlm_model() -> anyhow::Result<Model> {
    ModelBuilder::new("google/gemma-3-4b-it")
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await
}

impl Default for ScheduleQueues {
    fn default() -> Self {
        Self {
            active: Arc::new(Mutex::new(Vec::new())),
            pending: Arc::new(Mutex::new(Vec::new())),
            finished: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bill::Category;

    use super::*;
    #[tokio::test]
    async fn test_swap() {
        pretty_env_logger::init();
        Category::load_from_names(["No category"]);
        let scheduler = Scheduler::default();
        for i in 0..10 {
            let tcb = TaskControlBlock::new();
            tcb.set_state(task::State::Finished(Ok(task::Success(
                crate::bill::Bill {
                    notes: "No.".to_string(),
                    amount: i as f32 / 3f32,
                    category: Category::from_name("No category"),
                },
            ))));
            scheduler.queues.finished.lock().await.push(tcb);
        }
        let lookup_id = scheduler
            .queues
            .finished
            .lock()
            .await
            .first()
            .unwrap()
            .id()
            .to_string();
        scheduler
            .queues
            .move_inactive_to_swap(&mut *scheduler.swap_file.lock().await, 1)
            .await
            .unwrap();
        assert_eq!(scheduler.queues.finished.lock().await.len(), 1);
        assert!(scheduler.get_task(lookup_id).await.unwrap().is_some());
    }
}
