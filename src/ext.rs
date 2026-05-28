use ollama_rs::Ollama;
use reqwest::Url;

pub trait FromEnvVars {
    fn from_env_vars() -> Self;
}

impl FromEnvVars for Ollama {
    fn from_env_vars() -> Self {
        if let Ok(var) = std::env::var("OLLAMA_ENDPOINT") {
            let url = Url::parse(&var).expect("invalid OLLAMA_ENDPOINT");
            return Ollama::from_url(url);
        }
        Ollama::new("http://127.0.0.1", 11434)
    }
}
