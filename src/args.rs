use std::time::Duration;

use clap::Parser;

use crate::key;

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
/// Client pulling based HTTP server to implement a VLM based bookkeeping workflow.
pub struct Cli {
    #[arg(short, long, default_value = "127.0.0.1:3100")]
    pub bind: String,
    /// Bearer token for authentication, empty to disable
    #[arg(short, long)]
    pub auth_key: Option<String>,
    #[arg(
        short, long,
        default_values_t = ["Gorceries".to_string(), "Transport".to_string(), "Rent".to_string(), "Entertainment".to_string(), "Shopping".to_string(), "Drink".to_string(), "Food".to_string(), "Drink".to_string()])]
    pub categories: Vec<String>,
    /// Use larger language models
    #[arg(long, default_value_t = false)]
    pub large_model: bool,
    /// Number of concurrent model executions
    #[arg(long, default_value_t = 4)]
    pub max_concurrency: usize,
    /// How many result records until swapping to disk
    #[arg(long, default_value_t = 468_000)]
    pub max_memory_size: usize,
    /// How long to wait for until an inactive model is removed from system memory
    #[arg(long, default_value_t = 5f32)]
    pub model_timeout_minutes: f32,
    /// Offline mode, use cached models only without reaching Hugging Face hub
    #[arg(long, default_value_t = false)]
    pub offline: bool,
}

#[derive(Debug, Clone)]
pub struct App {
    pub auth_key: String,
    pub large_model: bool,
    pub max_concurrency: usize,
    pub max_memory_size: usize,
    pub model_timeout: Duration,
    pub offline: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            auth_key: String::new(),
            large_model: false,
            max_concurrency: 4,
            max_memory_size: 468_000,
            model_timeout: Duration::from_mins(5),
            offline: false
        }
    }
}

impl From<Cli> for App {
    fn from(value: Cli) -> Self {
        Self {
            auth_key: match value.auth_key {
                Some(key) => key.clone(),
                None => match std::env::var("AUTH_KEY") {
                    Ok(key) => key,
                    Err(_) => {
                        let random_key = key::generate_random_key();
                        log::error!("missing authorization key, using a random one: {random_key}");
                        random_key
                    }
                },
            },
            large_model: value.large_model,
            max_concurrency: value.max_concurrency,
            max_memory_size: value.max_memory_size,
            model_timeout: Duration::from_secs_f32(value.model_timeout_minutes * 60f32),
            offline: value.offline,
        }
    }
}
