use ollama_rs::Ollama;

pub trait FromEnvVars {
    fn from_env_vars() -> Self;
}

impl FromEnvVars for Ollama {
    fn from_env_vars() -> Self {
        if let Ok(var) = std::env::var("OLLAMA_HOST") {
            let (host, port) = var.split_once(':').expect("invalid OLLAMA_HOST");
            return Ollama::new(host, port.parse().expect("invalid port in OLLAMA_HOST"));
        }
        Ollama::new("127.0.0.1", 11434)
    }
}
