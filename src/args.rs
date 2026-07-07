use std::time::Duration;

use clap::Parser;

use crate::{key, task::ollama::GEMMA_4_E4B_Q4KM};

#[derive(Debug, Parser)]
#[command(version = option_env!("APP_VERSION"), about, long_about = None)]
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
    /// Caption model for describing screenshots
    #[arg(long, default_value = GEMMA_4_E4B_Q4KM)]
    pub caption_model: String,
    /// Extract model for amount & category analysis
    #[arg(long, default_value = GEMMA_4_E4B_Q4KM)]
    pub extract_model: String,
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
    pub caption_model: String,
    pub extract_model: String,
    pub max_concurrency: usize,
    pub max_memory_size: usize,
    pub model_timeout: Duration,
    pub offline: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            auth_key: String::new(),
            caption_model: GEMMA_4_E4B_Q4KM.into(),
            extract_model: GEMMA_4_E4B_Q4KM.into(),
            max_concurrency: 4,
            max_memory_size: 468_000,
            model_timeout: Duration::from_mins(5),
            offline: false,
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
                        println!("missing authorization key, using a random one: {random_key}");
                        random_key
                    }
                },
            },
            caption_model: value.caption_model,
            extract_model: value.extract_model,
            max_concurrency: value.max_concurrency,
            max_memory_size: value.max_memory_size,
            model_timeout: Duration::from_secs_f32(value.model_timeout_minutes * 60f32),
            offline: value.offline,
        }
    }
}
